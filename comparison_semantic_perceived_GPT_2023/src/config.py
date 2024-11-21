import os
import numpy as np

# paths

REPO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_TRAIN_DIR = os.path.join(REPO_DIR, "data/semantic_decoding/data_train")
DATA_TEST_DIR = os.path.join(REPO_DIR, "data/semantic_decoding/data_test")
MODEL_DIR = os.path.join(REPO_DIR, "models")
RESULT_DIR = os.path.join(REPO_DIR, "results")
SCORE_DIR = os.path.join(REPO_DIR, "scores")

# GPT encoding model parameters

TRIM = 5
STIM_DELAYS = [1, 2, 3, 4]
RESP_DELAYS = [-4, -3, -2, -1]
ALPHAS = np.logspace(1, 3, 10)
NBOOTS = 50
VOXELS = 10000
CHUNKLEN = 10
GPT_LAYER = 9
GPT_WORDS = 5

# decoder parameters

RANKED = True
WIDTH = 200
NM_ALPHA = 2/3
LM_TIME = 8
LM_MASS = 0.9
LM_RATIO = 0.1
EXTENSIONS = 5

# evaluation parameters

WINDOW = 20

# devices

GPT_DEVICE = "cuda"
EM_DEVICE = "cuda"
SM_DEVICE = "cuda"

## Transformers configs
lr = 0.0001
#d_model = 256
d_model = 1024 + 512
d_ff = 512
heads = 8
N = 3
epochs = 200
batch_size = 64

src_fmri_features_S1 = 81126
src_fmri_features_S2 = 94251
src_fmri_features_S3 = 95556
time_steps = 10
wandb_log = False
max_size = 300
