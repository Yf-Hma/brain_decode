import os

DATA_PATH = 'data/zuco'
PROCESSED_DATA_PATH = "data/zuco/processed"
MODELS_TRAIN_DIR =  "trained_models/zuco"

LLM_DIR = "LLMs/llama3"
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

# MLLM configs
llm_hidden_dim = 4096
max_txt_len=32
max_output_txt_len=100
use_nucleus_sampling=False
num_beams=3
max_new_tokens = 40
min_length=1
repetition_penalty=1.5
length_penalty=1
num_captions=1
temperature=0.8
fixed_instruction="Based on this content, generate a coherente text: "
