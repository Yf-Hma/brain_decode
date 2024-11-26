import os
import numpy as np


STORAGE_DIR = "data/raw_data/fmri_bold"
BOLD_DATA_PATH = STORAGE_DIR
MODELS_TRAIN_DIR =  "trained_models"

LLM_DIR = "llm/vicuna-7b-v1.3"


# Transformers config
d_model = 256
d_ff = 512
heads = 8
N = 2
src_fmri_features = 200
time_steps = 10
max_size = 100
type = 'spoken'
