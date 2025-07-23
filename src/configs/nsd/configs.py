import os

# Paths
DATA_PATH="data/nsd"
MODELS_TRAIN_PATH =  "trained_models/nsd"
LLM_PATH = "LLMs/llama3"
LLM_name = "llama"

assert os.path.exists (DATA_PATH), "DATA_PATH, specified in config file, does not exist!"

if not os.path.exists (MODELS_TRAIN_PATH):
    os.mkdir (MODELS_TRAIN_PATH)

coco_annotation_file_path = "./tools/COCO_73k_annots.npy"

voxels_per_subj = {1: 15724, 2: 14278, 3: 15226, 4: 13153, 5: 13039, 6: 17907, 7: 12682, 8: 14386}
src_fmri_features = 15724

# Transformers configs
d_model = 768
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
num_beams=2
max_new_tokens=10
min_length=1
top_p=0.9
repetition_penalty=1.5
length_penalty=0.9
num_captions=1
temperature=0.8

# Or, Describe this content:
fixed_instruction="Based on this content, provide a descriptive caption: "
