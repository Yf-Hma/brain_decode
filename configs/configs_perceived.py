import os
import numpy as np

# paths
RAW_DATA_PATH = "data/perceived"
MODELS_TRAIN_PATH = "trained_models/perceived"

if not os.path.exists (MODELS_TRAIN_PATH):
    os.mkdir(MODELS_TRAIN_PATH)


assert os.path.exists(RAW_DATA_PATH), "RAW_DATA_PATH does not exist."

DATA_TRAIN_PATH = os.path.join(RAW_DATA_PATH, "data_train")
DATA_TEST_PATH = os.path.join(RAW_DATA_PATH, "data_test")
PROCESSED_DATA_PATH = os.path.join(RAW_DATA_PATH, "processed")

DATA_JSON_PATH = os.path.join(RAW_DATA_PATH, "processed")

if not os.path.exists (DATA_JSON_PATH):
    os.mkdir(DATA_JSON_PATH)

LLM_PATH = "LLMs/llama3"
LLM_name = "llama"
llm_hidden_dim = 4096

if not os.path.exists(PROCESSED_DATA_PATH):
    os.mkdir(PROCESSED_DATA_PATH)


RESULT_PATH = "results/perceived"

if not os.path.exists(RESULT_PATH):
    os.mkdir(RESULT_PATH)

# Chunking parameters
TRIM = 5
STIM_DELAYS = [1, 2, 3, 4]
RESP_DELAYS = [-4, -3, -2, -1]
ALPHAS = np.logspace(1, 3, 10)
NBOOTS = 50
VOXELS = 10000
CHUNKLEN = 10
WINDOW = 20

## Transformers configs
d_model = 1024
d_ff = 512
heads = 16
N = 4
src_fmri_features_S1 = 81126
src_fmri_features_S2 = 94251
src_fmri_features_S3 = 95556

src_fmri_features_max = 95556

time_steps = 10
wandb_log = False
max_size = 200

# MLLM configs
max_txt_len=32
max_output_txt_len=256
use_nucleus_sampling=False
num_beams=5
max_new_tokens = 200
min_length=1
top_p=0.9
repetition_penalty=1.5
length_penalty=0.5
num_captions=1
temperature=0.8
fixed_instruction=""
