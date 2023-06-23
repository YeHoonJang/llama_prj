from transformers import AutoModelForCausalLM, AutoTokenizer

data_path = "/home/yehoon/workspace/llama_prj/data/ai_hub.json"

model = AutoModelForCausalLM.from_pretrained("EleutherAI/polyglot-ko-1.3b")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/polyglot-ko-1.3b")

