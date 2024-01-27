# %%
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from datasets import load_dataset
from transformers import PreTrainedTokenizerFast, PhiForCausalLM, TrainingArguments, TrainerCallback
import pandas as pd
import time
import torch 
from trl import DPOTrainer

# %% [markdown]
# # 1. Define sft model path and dpo data

# %%
dpo_file = './data/dpo_train_data.json'
tokenizer_dir = './model_save/tokenizer/'
sft_from_checkpoint_file = './model_save/sft/'
model_save_dir = './model_save/dpo/'
max_seq_len = 320

# %% [markdown]
# ## 2. Load dataset

# %%
tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
print(f"vocab size: {len(tokenizer)}")

# %%
dataset = load_dataset(path='json', data_files=dpo_file, split='train', cache_dir='.cache')

# %%
dataset[0]

# %% [markdown]
# # 3. Dataset token formatting
# Add `eos` token to the three columns of dpo dataset, `bos` can be added or not

# %%
def split_prompt_and_responses(samples: dict[str, str]) -> Dict[str, str]:
        prompts, chosens, rejects = [], [], []
        batch_size = len(samples['prompt'])
        for i in range(batch_size):
            # add an eos token for signal that end of sentence, using in generate.
            prompts.append(f"[BOS]{samples['prompt'][i]}[EOS]")
            chosens.append(f"[BOS]{samples['chosen'][i]}[EOS]")
            rejects.append(f"[BOS]{samples['rejected'][i]}[EOS]")

        return {
              'prompt': prompts,
              'chosen': chosens,
              'rejected': rejects,
        }

dataset = dataset.map(split_prompt_and_responses, batched=True,).shuffle(2333)

# %% [markdown]
# # 4. Load model
# `model` and `model_ref` start as the same model, only train `model` parameters, keep `model_ref` parameters unchanged

# %%

model = PhiForCausalLM.from_pretrained(sft_from_checkpoint_file)
model_ref = PhiForCausalLM.from_pretrained(sft_from_checkpoint_file)

model_size = sum(t.numel() for t in model.parameters())
print(f"Phi-2 size: {model_size / 1000**2:.1f}M parameters")

# %% [markdown]
# # 5. Define training callbacks
# Clear CUDA cache, dpo loads two models, consuming more memory, this effectively mitigates slow memory growth on low memory machines

# %%
class EmptyCudaCacheCallback(TrainerCallback):
    log_cnt = 0
    def on_log(self, args, state, control, logs=None, **kwargs):
        self.log_cnt += 1
        if self.log_cnt % 5 == 0:
            torch.cuda.empty_cache()
            
empty_cuda_cache = EmptyCudaCacheCallback()

# %% [markdown]
# # 6. Define training parameters

# %%
args = TrainingArguments(
    output_dir=model_save_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    num_train_epochs=4,
    weight_decay=0.1,
    warmup_steps=1000,
    learning_rate=2e-5,
    save_steps=2000,
    save_total_limit=3,
    report_to='tensorboard',
    bf16=True,
    logging_steps=10,
    log_level='info',
    logging_first_step=True,
    optim="adafactor",
    remove_unused_columns=False,
    group_by_length=True,
)

trainer = DPOTrainer(
    model,
    model_ref,
    args=args,
    beta=0.1,
    train_dataset=dataset,
    tokenizer=tokenizer,
    callbacks=[empty_cuda_cache],
    max_length=max_seq_len * 2 + 16, # 16 for eos bos
    max_prompt_length=max_seq_len,
)

# %% [markdown]
# # 7. Training

# %%
trainer.train()

# %% [markdown]
# # 8. Save logs and model

# %%
loss_log = pd.DataFrame(trainer.state.log_history)
loss_log.to_csv(f"./logs/dpo_train_log_{time.strftime('%Y%m%d-%H%M')}.csv")


trainer.save_model(model_save_dir)

