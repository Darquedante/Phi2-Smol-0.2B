# Phi2-Tiny-0.2B Train your own Phi2 smpl model from scratch

**This project is experimental, with open-source code and model weights. The pre-training data is limited.

- Supports acceleration with flash attention 2


# 1. ⚗️ Data Cleaning
For example, adding periods at the end of sentences, removing duplicate punctuation (e.g., many dialogue corpora have a lot of `"。。。。。"`), etc.   



# 2. 🗨️ Tokenizer Training
Code: [tokenizer.ipynb](./1.tokenizer.ipynb)
This project uses a `byte level` `BPE` tokenizer. Training code for both `char level` and `byte level` tokenizers is provided.

After training the tokenizer, remember to check if common special symbols like `\t`, `\n`, etc., are in the vocabulary. You can try encoding and decoding a sentence with special characters to see if it can be restored. If these special symbols are not included, add them using the `add_tokens` function. Use `len(tokenizer)` to get the vocabulary size; `tokenizer.vocab_size` does not count characters added through the `add_tokens` function.

Tokenizer training is memory-intensive:

- `byte level` training with 100 million characters needs at least `32G` of memory (actually, `32G` is still not quite enough, frequent swapping may occur), taking about 1 hour on `13600k`.

- `char level` training with 650 million characters (exactly the size of the Chinese Wikipedia corpus) needs at least 32G of memory. Due to multiple swaps, the actual usage far exceeds 32G, taking about half an hour on `13600k`.

Therefore, for large datasets (GB level), it is recommended to sample from the dataset when training the `tokenizer`.



# 3. ⛏️ CLM Causal Model Pre-training

Conduct unsupervised pre-training with a large amount of text, mainly using the `bell open source` dataset [BELLE](https://github.com/LianjiaTech/BELLE).

Dataset format: one sentence per sample, longer sentences can be truncated and divided into multiple samples.

During CLM pre-training, the model's input and output are the same. When calculating the cross-entropy loss, it needs to be shifted by one position (`shift`).

When processing encyclopedia corpus, it is recommended to add an `'[EOS]'` marker at the end of each entry. Similar processing for other corpora, an end of a `doc` (which could be the end of an article or a paragraph) should also be marked with `'[EOS]'`. The start marker `'[BOS]'` can be added or not.



# 4. ⚒️ SFT Instruction Fine-tuning

Mainly using the `bell open source` dataset. Thanks to the contributor [BELLE](https://github.com/LianjiaTech/BELLE).

The format for SFT training data is as follows:  
```python
text = f"##Question:\n{example['instruction']}\n##Answer:\n{example['output'][EOS]"
```
The model ignores parts before the marker "##Answer:" (including "##Answer:" itself) when calculating the loss, starting from the text after "##Answer:".

Remember to add EOS, the special marker for sentence ending; otherwise, the model won't know when to stop decoding. The BOS sentence start marker is optional.



# 5. 📝 RLHF Optimization

Adopt a simpler, more memory-efficient DPO preference optimization method.

Code: [dpo.ipynb](./4.dpo.ipynb)

Fine-tune the SFT model according to personal preferences. The dataset should have three columns: `prompt`, `chosen`, and `rejected`. Some `rejected` data is generated from an early version of the SFT model (e.g., take a model checkpoint from 0.5 of 4 `epoch` training in SFT). If the similarity between the generated `rejected` and `chosen` is above 0.9, discard that data.

During the DPO process, there should be two models: one for training and one for reference. At loading, they are actually the same model, but the reference model does not participate in parameter updates.



# 6. 📑 Usage of This Project's Model
## 6.1 General Conversation Capabilities
Model weights in `huggingface` repository: [Phi2-smol-0.2B](https://huggingface.co/TBD/Phi2-Smol-0.2B)
```python
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained('charent/Phi2-Chinese-0.2B')
model = AutoModelForCausalLM.from_pretrained('charent/Phi2-Chinese-0.2B').to(device)

txt = 'What should I do if I catch a cold?'
prompt = f"##Question:\n{txt}\n##Answer:\n"

# greedy search
gen_conf = GenerationConfig(
    num_beams=1,
    do_sample=False,
    max_length=320,
    max_new_tokens=256,
    no_repeat_ngram_size=4,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)

tokened = tokenizer.encode_plus(text=prompt)
input_ids, attention_mask = torch.LongTensor([tokened.input_ids]).to(device), \
    torch.LongTensor([tokened.attention_mask]).to(device)

outputs = model.generate(
    inputs=input_ids,
    attention_mask=attention_mask,
    generation_config=gen_conf,
)

outs = tokenizer.decode(outputs[0].cpu().numpy(), clean_up_tokenization_spaces=True, skip_special_tokens=True,)
print(outs)


##Question:
What should I do if I catch a cold?
##Answer:
A cold is caused by a virus, and common colds are generally caused by viruses. Here are some common methods for dealing with a cold:
- Wash hands, especially after contacting other people or objects.
- Cover your mouth and nose with a tissue or elbow when coughing or sneezing.
- Avoid touching your mouth and nose, especially the throat and nose.
- If coughing or sneezing, use a tissue or handkerchief to cover your mouth and nose but stay away from other people.
- If you have a cold, it's best not to touch your eyes, nose, and mouth.
- During a cold, it's best to maintain adequate hydration and rest to relieve physical fatigue.
- If you already have a cold, you can drink some warm water or salt water to replenish body fluids.
- Also, if you catch a cold, it's advisable to seek medical attention promptly.
6.2 Retrieval-Enhanced Generation (RAG)
For detailed code, see rag_with_langchain.ipynb
```


rag



7. 🎓 Citations
If you find this project helpful, feel free to cite it.

conf
Copy code
@misc{Charent2023,
    author={Charent Chen},
    title={A small Chinese causal language model with 0.2B parameters base on Phi2},
    year={2023},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/charent/Phi2-mini-Chinese}},
}



8. 🤔 Other Matters
This project is not responsible for any data security, public opinion risks, or risks and responsibilities arising from the misuse, propagation, inappropriate use, or misguidance of the open-source model and code.

