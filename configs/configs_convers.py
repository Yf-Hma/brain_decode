import os

RAW_FMRI_DATA_PATH="data/convers/raw_data/fmri"
DATA_PATH="data/convers"
MODELS_TRAIN_PATH = "trained_models/convers"

if not os.path.exists (MODELS_TRAIN_PATH):
    os.mkdir(MODELS_TRAIN_PATH)

PROCESSED_FMRI_DATA_PATH = os.path.join (DATA_PATH, "preprocessed_fmri_data")

LLM_PATH = "LLMs/vicuna-7b" # Or other LLMs
LLM_name = "vicuna"

# check if paths exist
assert os.path.exists(RAW_FMRI_DATA_PATH), "RAW_FMRI_DATA_PATH does not exist."

# DATA parameters
fmri_timestep=1.2 # fMRI scans are recorded with a timestep of 1.2 s
time_steps=10 # 10 time steps are considered (this is a fixe parameter in this experiment)
interval_length=12 # 10 time steps, equivalent to 12s

# Transformers configs
d_model = 256
d_ff = 512
heads = 8
N = 2
src_fmri_features = 200
max_size = 72
type = 'spoken'

# MLLM configs
llm_hidden_dim = 4096 # 3072 for llama3b

max_txt_len=128
max_output_txt_len=128
use_nucleus_sampling=False
num_beams=3
max_new_tokens = 70
min_length=1
top_p=0.9
repetition_penalty=1.5
length_penalty=0.9
num_captions=1
temperature=1

fixed_instruction= "En se basant sur ce contenu, réponds en Français : "
fixed_instruction_with_input_text="En se basant sur ce contenu, réponds en Français à la phrase suivante "
