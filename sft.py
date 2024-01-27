# Import necessary libraries and modules
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast, PhiForCausalLM, TrainingArguments, Trainer, TrainerCallback
import pandas as pd
import numpy as np
import time
import torch
from trl import DataCollatorForCompletionOnlyLM

# Define training data, tokenizer, pretrained model path, and maximum sequence length
sft_file = './data/sft_train_data.parquet'
tokenizer_dir = './model_save/tokenizer/'
sft_from_checkpoint_file = './model_save/pre/'
model_save_dir = './model_save/sft/'
max_seq_len = 320

# Load the training dataset from a parquet file
dataset = load_dataset(path='parquet', data_files=sft_file, split='train', cache_dir='.cache')

# Initialize a tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
print(f"Vocabulary size: {len(tokenizer)}")

# Define special characters for the sft data_collator
# The commented code block below is used to manually add 'instruction_template_ids' and 'response_template_ids'
# to input_ids because with a byte-level tokenizer, ':' and the following characters may get combined,
# causing issues in finding 'instruction_template_ids' and 'response_template_ids'.
# You can also manually add '\n' before and after '#' and ':' as shown in the following sections.

# %%
instruction_template = "## Question:"
response_template = "## Answer:"

# %%
# The commented-out code block below is used to manually add 'instruction_template_ids' and 'response_template_ids' to input_ids.

# template_ids = tokenizer([instruction_template, response_template])['input_ids']
# instruction_template_ids, response_template_ids = template_ids[0], template_ids[1]
# print(instruction_template_ids, response_template_ids)
# def formatting_prompts_func(example: list[dict]) -> list[str]:
#     batch_prompt,  batch_response = [], []
#     n = len(example['instruction'])
#     for i in range(n):
#         batch_prompt.append(example['instruction'][i])
#         batch_response.append(example['output'][i])
        
#     prompt_ids = tokenizer(batch_prompt, return_attention_mask=False)['input_ids']
#     resopnse_ids = tokenizer(batch_response, return_attention_mask=False)['input_ids']

#     input_ids = []
#     for i in range(n):
#         cur_input_ids = [tokenizer.bos_token_id] + instruction_template_ids + prompt_ids[i] \
#                         + response_template_ids + resopnse_ids[i] + [tokenizer.eos_token_id]
#         input_ids.append(cur_input_ids)
    
#     return {'input_ids': input_ids}

# from typing import List, Union


# class Phi2DataCollatorForCompletionOnlyLM(DataCollatorForCompletionOnlyLM):
#     def __init__(self, response_template: str | List[int], instruction_template: str | List[int] = None, *args, mlm: bool = False, ignore_index: int = -100, **kwargs):
#         super().__init__(response_template, instruction_template, *args, mlm=mlm, ignore_index=ignore_index, **kwargs)
    
#     def __call__(self, features, return_tensors=None):
#         '''
#         After executing formatting_prompts_func map, the dataset's __getitem__ method will return batch_size input_ids.
#         '''
#         batch_size = len(features)
#         paded_input_ids = tokenizer.pad(
#             {'input_ids': features['input_ids']},
#             padding=True,
#             return_attention_mask=False,
#         )['input_ids']

#         data = []
#         for i in range(batch_size):
#             data.append(
#                 {'input_ids': }
#             )

#         # Let the parent class execute LM mask in the end.
#         return super().__call__(data, return_tensors)

# %%


map_dtype = np.uint16 if len(tokenizer) < 65535 else np.uint32

def batched_formatting_prompts_func(example: list[dict]) -> list[str]:
    batch_txt = []
    for i in range(len(example['instruction'])):
        text = f"{instruction_template}\n{example['instruction'][i]}\n{response_template}\n{example['output'][i]}[EOS]"
        batch_txt.append(text)

    # token to id 
    outputs = tokenizer(batch_txt, return_attention_mask=False)
    input_ids = [np.array(item, dtype=map_dtype) for item in outputs["input_ids"]]

    return {
            "input_ids": input_ids
        }

# print(batched_formatting_prompts_func(samples))

# %%
# Shuffle the dataset with batched formatting prompts and remove columns
dataset = dataset.map(batched_formatting_prompts_func, batched=True, remove_columns=dataset.column_names).shuffle(23333)

# %% [markdown]
# ## 2.2 Define data_collator

# %%
# Set mlm=False to indicate training a CLM model
data_collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, response_template=response_template, tokenizer=tokenizer, mlm=False)

# %% [markdown]
# # 4. Load pretrained model

# %%

model = PhiForCausalLM.from_pretrained(sft_from_checkpoint_file)

model_size = sum(t.numel() for t in model.parameters())
print(f"Phi2 size: {model_size / 1000**2:.2f}M parameters")

# %% [markdown]
# ## Define callback functions during training
# Clear the CUDA cache after N log iterations to effectively reduce memory growth on low-memory GPUs

# %%
class EmptyCudaCacheCallback(TrainerCallback):
    log_cnt = 0
    def on_log(self, args, state, control, logs=None, **kwargs):
        self.log_cnt += 1
        if self.log_cnt % 5 == 0:
            torch.cuda.empty_cache()
            
empty_cuda_cahce = EmptyCudaCacheCallback()

# %% 
my_datasets =  dataset.train_test_split(test_size=4096)

# %% [markdown]
# # 5. Define training parameters

# %%
class EmptyCudaCacheCallback(TrainerCallback):
    log_cnt = 0
    def on_log(self, args, state, control, logs=None, **kwargs):
        self.log_cnt += 1
        if self.log_cnt % 5 == 0:
            torch.cuda.empty_cache()
            
empty_cuda_cahce = EmptyCudaCacheCallback()

# %% 
my_datasets =  dataset.train_test_split(test_size=4096)

# %% [markdown]
# # 5. Define Training Parameters

# %%
args = TrainingArguments(
    output_dir=model_save_dir,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    weight_decay=0.1,
    warmup_steps=1000,
    learning_rate=5e-5,
    evaluation_strategy='steps',
    eval_steps=2000,
    save_steps=2000,
    save_total_limit=3,
    report_to='tensorboard',
    optim="adafactor",
    bf16=True,
    logging_steps=10,
    log_level='info',
    logging_first_step=True,
    group_by_length=True,
)
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=my_datasets['train'],
    eval_dataset=my_datasets['test'],
    callbacks=[empty_cuda_cahce],
)


# %% [markdown]
# # 6. Start Training

# %%
trainer.train(
    # resume_from_checkpoint=True
)

# %% [markdown]
# Compute Perplexity

# %%
eval_results = trainer.evaluate()
print(f"Perplexity: {np.exp(eval_results['eval_loss']):.2f}")

# %% [markdown]
# # 7. Save Logs and Model

# %%
loss_log = pd.DataFrame(trainer.state.log_history)
loss_log.to_csv(f"./logs/sft_train_log_{time.strftime('%Y%m%d-%H%M')}.csv")


trainer.save_model(model_save_dir)

# %%



