import os

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
