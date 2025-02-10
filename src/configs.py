import os

# Paths
DATA_PATH="data"
JSON_DATA_PATH_OUT="data"
RAW_FMRI_DATA_PATH="data/raw_data/ds001740-2.2.0"
PROCESSED_FMRI_DATA_PATH = os.path.join (DATA_PATH, "raw_data/fmri_bold")
MODELS_TRAIN_DIR =  "trained_models"
LLM_DIR = "llm/vicuna-7b-v1.3"

# Transformers configs
d_model = 256
d_ff = 512
heads = 8
N = 2
src_fmri_features = 200
time_steps = 10
max_size = 100
type = 'spoken'

# DATA parameters
fmri_timestep=1.2 # fMRI scans are recorded with a timestep of 1.2 s
time_steps=10 # 10 time steps are considered (this is a fixe parameter in this experiment )
interval_length=12 # 10 time steps, equivalent to 12s
bold_lag=3 #There is a lag between bold function activation and a stimuli event, approximately 4s

# MLLM configs
max_txt_len=128
max_output_txt_len=256
use_nucleus_sampling=False
num_beams=3
max_new_tokens = 100
min_length=1
top_p=0.9
repetition_penalty=1.5
length_penalty=0.9
num_captions=1
temperature=1
fixed_instruction="En se basant sur ce contenu, réponds en Français à la phrase suivante: "
