import os

# Paths
DATA_PATH="data/NSD/"
RAW_FMRI_DATA_PATH="data/NSD"
MODELS_TRAIN_DIR =  "trained_models/NSD"


# Coco annotation
coco_annotation_file_path = "./tools/COCO_73k_annots.npy"

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
LLM_DIR = "llms/llama3"
LLM_name = "llama3"
llm_hidden_dim = 4096
fixed_instruction="Based on this content, provide a simple caption: " # Or, Describe this content:


max_txt_len=64
max_output_txt_len=64
use_nucleus_sampling=False
num_beams=2
max_new_tokens=10
min_length=1
top_p=0.9
repetition_penalty=1.5
length_penalty=0.9
num_captions=1
temperature=0.8
