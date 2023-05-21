from datasets import load_dataset
from transformers import AutoModelForCausalLM, LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-13b-hf")

model = AutoModelForCausalLM.from_pretrained("decapoda-research/llama-13b-hf")

data_files = {"train": "data/ratings_train.csv", "test": "data/ratings_test.csv"}
dataset = load_dataset("csv", data_files=data_files)


def tokenize_function(examples):
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer(examples["document"], padding=True, truncation=True, return_tensors="pt")


tokenized_dataset = dataset.map(tokenize_function, batched=True)
