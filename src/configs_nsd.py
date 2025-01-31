import os
from tokenizers import Tokenizer

# Paths
DATA_PATH="data/NSD/"
RAW_FMRI_DATA_PATH="data/NSD/"
PROCESSED_FMRI_DATA_PATH = ""
MODELS_TRAIN_DIR =  "trained_models"

LLM_DIR = "llama3"
LLM_name = "llama"


# fMRI number of ROIS
voxels_per_subj = {1: 15724, 2: 14278, 3: 15226, 4: 13153, 5: 13039, 6: 17907, 7: 12682, 8: 14386}

# Transformers configs
d_model = 1024 #+ 256
d_ff = 512
heads = 8
N = 4
#src_fmri_features = 15724
src_fmri_features = 17910
time_steps = 3
max_size = 40
type = 'caption'
vocab_len = 20702
# vocab_len = Tokenizer.from_file("../tools/tokenizer_nsd.json").get_vocab_size()


# BrainClip configs
emb_dim = d_model
n_heads = 4
n_layers = 2
width = 512
max_seq_length = max_size
vocab_size = vocab_len
img_size = (time_steps, src_fmri_features)
patch_size = (time_steps, 16)
n_channels = 1
vit_width = 512
vit_layers = 4
vit_heads = 8

# MLLM configs
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
temperature=0.9

# Or, Describe this content:
fixed_instruction="Based on this content, provide a simple caption: "


# Vector quantizer hyperparameters
num_hiddens=d_model
embedding_dim=512
num_embeddings=d_model
