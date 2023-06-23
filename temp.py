from tqdm.auto import tqdm

from datasets import load_dataset
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM, get_scheduler

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

data_files = {"train": "data/ratings_train.csv", "test": "data/ratings_test.csv"}
dataset = load_dataset("csv", data_files= data_files)

tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")

#
# def tokenize_function(examples):
# #     tokenizer.pad_token = tokenizer.eos_token
# #     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#     tokenizer.pad_token_id = (0)
# #     return tokenizer((examples["document"]), padding="max_length", return_tensors="pt")
#     return tokenizer(examples["document"], padding="max_length", truncation=True, max_length=32000, return_tensors="pt")
# #     print(examples["document"])
# #     return tokenizer(examples["document"])
#
#
# tokenized_dataset = dataset.map(tokenize_function)
#
# train_loader = DataLoader(tokenized_dataset["train"], shuffle=True, batch_size=256)
# eval_loader = DataLoader(tokenized_dataset["test"], shuffle=True, batch_size=256)
#
# model = LlamaForCausalLM.from_pretrained("decapoda-research/llama-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True)
#
# optimizer = AdamW(model.parameters(), lr=5e-5)
#
# num_epochs = 3
# num_training_steps = num_epochs * len(train_loader)
# lr_scheduler = get_scheduler(
#     name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
# )
#
# progress_bar = tqdm(range(num_training_steps))
#
# model.train()
# for epoch in range(num_epochs):
#     for batch in train_loader:
# #         print(batch)
# #         batch = batch["input_ids"]
#         print(batch)
#         batch = {k: v for k, v in batch}
#         outputs = model(**batch)
#         loss = outputs.loss
#         loss.backward()
#
#         optimizer.step()
#         lr_scheduler.step()
#         optimizer.zero_grad()
#         progress_bar.update(1)