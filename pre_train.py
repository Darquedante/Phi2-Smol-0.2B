# %%
import os, platform, time
from typing import Optional

from transformers import PreTrainedTokenizerFast, DataCollatorForLanguageModeling, PhiConfig, PhiForCausalLM, Trainer, TrainingArguments, TrainerCallback
from datasets import load_dataset, Dataset  
import pandas as pd
from transformers.trainer_callback import TrainerControl, TrainerState
import numpy as np 
from dataclasses import dataclass, field
import torch

# %%
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


attn_implementation = 'flash_attention_2'
try:
    from flash_attn import flash_attn_func 
except Exception as e:
    attn_implementation = 'eager'

# %% [markdown]  
# # 1. Training data source

TRAIN_FILES = [
    './data/wiki_chunk_320_2.2M.parquet',
    './data/bell_pretrain_400_3M.parquet',  
]

EVAL_FILE = './data/pretrain_eval_400_1w.parquet'

# %%

@dataclass  
class PretrainArguments:
    tokenizer_dir: str = './model_save/tokenizer/'
    model_save_dir: str = './model_save/pre/'
    logs_dir: str = './logs/'
    train_files: list[str] = field(default_factory=lambda: TRAIN_FILES)
    eval_file: str = EVAL_FILE
    max_seq_len: int = 512  

    # Use default attention implementation on Windows
    attn_implementation: str = 'eager' if platform.system() == 'Windows' else attn_implementation

pretrain_args = PretrainArguments()  

# %% [markdown]
# # 2. Load pretrained tokenizer  
# If you added your own tokens with `add_tokens`, you must use `len(tokenizer)` to get the length. `tokenizer.vocab_size` does not include tokens you added.  

# %%
tokenizer = PreTrainedTokenizerFast.from_pretrained(pretrain_args.tokenizer_dir)  

# %% [markdown]
# # 5. Define the model  
# Define from `config`, not `from_pretrained`.  
# For ease of cuda computation, pay attention to the size of the vocabulary. If it is not a multiple of 64, manually round up to a multiple of 64, or another power of 2 like 32, 128, 256.  

# %%
vocab_size = len(tokenizer)
if vocab_size % 64 != 0:
    vocab_size = (vocab_size // 64 + 1) * 64
print(f"final vocab sieze: {vocab_size}")  

# %% [markdown]  
# ## Cache token to id mapping to file, so no need to tokenize again during use
# Use uint16 if vocab size < 65535 to save space, else uint32

# %%
map_dtype = np.uint16 if vocab_size < 65535 else np.uint32

def token_to_id(samples: dict[str, list]) -> dict:

    batch_txt = samples['text']
    outputs = tokenizer( 
        batch_txt,  
        truncation=False,
        padding=False,
        return_attention_mask=False, 
    )

    input_ids = [np.array(item, dtype=map_dtype) for item in outputs["input_ids"]]  

    return {  
            "input_ids": input_ids
        }  


# %% [markdown]  
# # 3. Load dataset and map

# %%
def get_maped_dataset(files: str|list[str]) -> Dataset:
    dataset = load_dataset(path='parquet', data_files=files, split='train', cache_dir='.cache')
    maped_dataset = dataset.map(token_to_id, batched=True, batch_size=1_0000, remove_columns=dataset.column_names)
    return maped_dataset  

train_dataset = get_maped_dataset(pretrain_args.train_files)
eval_dataset = get_maped_dataset(pretrain_args.eval_file)

print(train_dataset, eval_dataset)

# %% [markdown]
# # 4. Define data_collator 
# Set `mlm=False` for CLM, `mlm=True` for MLM

# %% 
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)  

# %% [markdown]
# ## Validate data, check padding, input/output meets requirements   

# %%
if pretrain_args.attn_implementation == 'flash_attention_2':
    torch.set_default_dtype(torch.bfloat16)


# %%
phi_config = PhiConfig(
    vocab_size=vocab_size,
    bos_token_id=tokenizer.bos_token_id, 
    eos_token_id=tokenizer.eos_token_id,
    hidden_size=960,
    num_attention_heads=16,
    num_hidden_layers=22,
    max_position_embeddings=512,
    intermediate_size=4096,
    attn_implementation=pretrain_args.attn_implementation,  
)

model = PhiForCausalLM(phi_config)
# model = model.to_bettertransformer()  

# Another way to use flash_attention_2
# model = PhiForCausalLM.from_pretrained('./model_save/300m', torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2") 
# model = model.to('cuda')  

model_size = sum(t.numel() for t in model.parameters())
print(f"Phi-2 size: {model_size / 1000**2:.1f}M parameters")

# %% [markdown]
# # 6. cuda cache callback  

# %%
class MyTrainerCallback(TrainerCallback):
    log_cnt = 0
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        '''
        Clear cuda cache every n logs, for low memory devices, prevent OOM
        '''
        self.log_cnt += 1 
        if self.log_cnt % 2 == 0:
            torch.cuda.empty_cache()  
        
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        '''
        Save model on epoch end.  
        epoch and steps in TrainingArguments save_strategy are not compatible. To save every save_steps, consider disk space and keep max last 3 checkpoints.  
        '''
        # Set should_save=True  
        control.should_save = True
        return control
    
my_trainer_callback = MyTrainerCallback()  

# %% [markdown]  
# # 6. Define training args  

# %%
args = TrainingArguments(  
    output_dir=pretrain_args.model_save_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=32, 
    num_train_epochs=4,
    weight_decay=0.1,
    warmup_steps=1000,
    learning_rate=5e-4,
    evaluation_strategy='steps',
    eval_steps=2000,  
    save_steps=2000,
    save_strategy='steps',
    save_total_limit=3,
    report_to='tensorboard',   
    optim="adafactor",   
    bf16=True,  
    logging_steps=5,
    log_level='info',
    logging_first_step=True,  
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[my_trainer_callback],  
)

# %% [markdown]  
# # 7. Start training
# Set `resume_from_checkpoint=True` to resume from last checkpoint

# %%
trainer.train( 
    # resume_from_checkpoint=True
)

# %% [markdown]
# Calculate perplexity

# %%
eval_results = trainer.evaluate()
print(f"Perplexity: {np.exp(eval_results['eval_loss']):.2f}")  

# %% [markdown]
# # 8. Save final loss log and model  

# %%  

loss_log = pd.DataFrame(trainer.state.log_history)
loss_log.to_csv(f"./logs/pre_train_log_{time.strftime('%Y%m%d-%H%M')}.csv")  


trainer.save_model(pretrain_args.model_save_dir)
