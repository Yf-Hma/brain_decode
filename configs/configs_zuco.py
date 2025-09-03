import os

DATA_PATH = 'data/ZuCo'
PROCESSED_DATA_PATH = "data/ZuCo/processed"
MODELS_TRAIN_PATH =  "trained_models/zuco"

LLM_PATH = "LLMs/llama3"
LLM_name = "llama"

# DATA parameters
time_steps=56

# Transformers configs
d_model = 256
d_ff = 512
heads = 4
N = 8
src_eeg_features = 840
max_size = 60
type = 'reading'

# MLLM configs
llm_hidden_dim = 4096
max_txt_len=32
max_output_txt_len=100
use_nucleus_sampling=False
num_beams=3
max_new_tokens = 60
min_length=1
top_p=0.9
repetition_penalty=1.5
length_penalty=1
num_captions=1
temperature=0.8
fixed_instruction="<|end_of_signal|>"
