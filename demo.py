from transformers import AutoModelForCausalLM, LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-13b-hf")

model = AutoModelForCausalLM.from_pretrained("decapoda-research/llama-13b-hf")