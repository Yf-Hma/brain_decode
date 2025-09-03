import os
DATA_PATH = "data/nsd"
MODELS_TRAIN_PATH =  "trained_models/nsd"

assert os.path.exists(MODELS_TRAIN_PATH), "MODELS_TRAIN_PATH does not exist."

LLM_PATH = "LLMs/llama3"
LLM_name = "llama"

# Transformers configs
d_model = 1024
emb_dim = d_model
d_ff = 512
heads = 16
N = 4
time_steps = 3
max_size = 40
type = 'caption'
vocab_len = 20702

# MLLM configs
llm_hidden_dim = 4096
max_txt_len=64
max_output_txt_len=64
use_nucleus_sampling=False
num_beams=3
max_new_tokens=12
min_length=1
top_p=0.9
repetition_penalty=1.5
length_penalty=0.9
num_captions=1
temperature=0.8

# Or, Describe this content:
fixed_instruction="Based on this content, provide a descriptive caption: "
