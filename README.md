### BrainDEC: A Multimodal LLM for the Non-Invasive Decoding of Text from Brain Recordings


![model](figs/overview_1.png)

 <!-- In this work, we propose and end-to-end multimodal large language model for decoding the spoken text in a human-human or human-robot interactions. The proposed architecture is founded on  ($i$) an encoder derived from a specific transformer with the incorporation of an augmented embedding layer for the encoder and a better-adjusted attention mechanism than that present in the state of the art, and ($ii$) a frozen LLM adapted via instruction tuning to align the embedding of the input modalities to decode the output text. A benchmark in performed on two publicly available datasets where fMRI brain activity and conversational signals are recorded synchronously. -->

### Requirements
* Python >=3.12.2
* Required packages:  ```python install -r requirement.txt```
* vicuna-7b-v1.3 from https://huggingface.co/lmsys/vicuna-7b-v1.3 in the folder 'LLMs' (or others such as Meta-llama3.2-8b-Instruct)
* CLIP: git clone https://github.com/openai/CLIP

### Experiments
This repository contains three experiments associated to three different tasks and datasets:
 - Spoken text decoding (convers): Multimodal spoken text decoding during conversations (main task of this work).
 - Perceived speech decoding (perceived): Decoding the textual content of listened stories.
 - Brain captioning (NSD): Decoding the captions of viewed images using the NSD datasets.
 - Decoding reading text from EEG signals.

In the following, we detail the steps to reproduce the results of each experiment presented in the paper.

#### 1. Spoken Text Decoding
##### Configuration
- Update the [configuration file](configs/configs_convers.py) by specifying the following paths: __DATA_PATH__ (ex. data/convers), __RAW_FMRI_DATA_PATH__ (ex. data/fmri_convers), __MODELS_TRAIN_PATH__ (ex. trained_models/convers), __LLM_PATH__ (ex. LLMs/Meta-llama3.2-8b-Instruct)
- Download the the Convers datasets version 2.2.0 from the OpenNeuro platform [ds001740](https://openneuro.org/datasets/ds001740/versions/2.2.0)
 inside __RAW_FMRI_DATA_PATH__ specified in the config file.
- Create a folder named "raw_data/transcriptions" inside __DATA_PATH__ and upload  the raw Transcriptions from the Ortolang platform [convers/v2](https://www.ortolang.fr/market/corpora/convers/v2) into it:


With DATA_PATH set to "data/convers", you should obtain a structure similar to this after data preprocessing:

```
data
└── convers
    ├── preprocessed_fmri_data
    │   └── fMRI_data_200
    ├── processed_data
    │   ├── fMRI_data_split
    │   ├── interlocutor_text_data
    │   └── participant_text_data
    ├── raw_data
    │   ├── transcriptions
    │   └── fmri
    ├── test.json
    └── train.json
```

##### Preprocessing and evaluation
```bash
# Preprocessing raw data
python exps/convers/process_raw_bold_signal.py --n_rois 200 # Parcellation using 200 ROIs
python exps/convers/data_builder_tools/split_bold_files.py  # Processing raw 4D voxel BOLD signals and segmenting them into fixed-duration chunks
python exps/convers/data_builder_tools/textgrid_to_text.py # Processing transcription files (conversations) and segmenting them into fixed-duration text sequences

# Building training and test data
python exps/convers/data_builder_tools/build_data.py # Using json files to save paths of bold chunks and the [input, output] text for instruction tuning
python exps/convers/data_builder_tools/build_tokenizer.py # Building the tokenizer for the first stage of training

# Training and testing after each save_epoch
python  exps/convers/train_stage1.py -m DeconvBipartiteTransformerConv --batch_size 128 --epochs 200 # Stage-1: training the DeconvBipartite Transformer
python  exps/convers/train_stage2.py --batch_size 32 --epochs 100  -m BrainDEC_V0  --save_epochs 50 # Stage-2. Note: BrainDEC_V1 or BrainDEC_V2 converge quickly than V0, only 20 epochs are needed.

# Evaluate the results of the test set and save the scores
python exps/convers/evaluation.py   
```   

#### 2. Perceived Speech Decoding
##### Configuration
- Update the [configuration file](configs/configs_perceived.py) by specifying the following paths: __RAW_FMRI_DATA_PATH__ (ex. data/perceived), __MODELS_TRAIN_PATH__ (ex. trained_models/perceived), and __LLM_PATH__ (ex. LLMs/Meta-llama3.2-8b-Instruct).
- In the folders "DATA_TRAIN_DIR" and "DATA_TEST_DIR" (see the config file), download the training and test datasets as outlined in this project [semantic-decoding](https://github.com/HuthLab/semantic-decoding).

With DATA_PATH set to "data/perceived" for example, you should obtain a structure similar to this after data preprocessing:

```
data
└── perceived
    ├── data_test
    ├── data_train
    └── processed
        ├── S1
        ├── S2
        ├── S3
        ├── fMRI_data_test_split
        └── fMRI_data_train_split
```

##### Preprocessing and evaluation
```bash
# Data preparation
python exps/perceived/prepare_datasets.py -s $subject (for $subject  in ['S1', 'S2', 'S3'])

# Build tokenizer for stage 1
python exps/perceived/build_tokenizer.py

# Stage-1 training (in a cross-subject manner)
python exps/perceived/train_stage1.py --batch_size 128

# Stage-2 training (for each subject separately)
python exps/perceived/train_stage2.py --batch_size 32 -s $subject (for $subject  in ['S1', 'S2', 'S3'])

# Evaluate the results of the test set and save the scores
python exps/perceived/evaluation.py $subject (for $subject  in ['S1', 'S2', 'S3'])
```   

#### 3. Brain Captioning - BrainHub benchmark on NSD dataset
This is a comparison with brain understanding benchmark ([BrainHub](https://github.com/weihaox/BrainHub)), based on Natural Scenes Dataset [NSD](https://naturalscenesdataset.org/) and [COCO](https://cocodataset.org).

##### Configuration
- The processed datasets are available in [here](https://huggingface.co/datasets/pscotti/naturalscenesdataset).
- Download the datasets using this [script](https://github.com/weihaox/UMBRAE/blob/main/umbrae/download_data.sh).
- Download COCO annotations from this [link](https://huggingface.co/datasets/pscotti/naturalscenesdataset/blob/main/COCO_73k_annots_curated.npy) in the folder 'tools'
- Update the [configuration file](configs/configs_nsd.py) to specify the paths, and eventually to modify the hyperparameters.

 ```
 data
 └── nsd
     ├── webdataset_avg_split
     │   ├── test
     │   ├── train
     └── └── val
 ```
##### Traning
 - To train and evaluate the model:

 ```bash
 # Stage-1 training (in a cross-subject manner)
 python exps/zuco/build_tokenizer.py
 python exps/nsd/train_stage1.py --batch_size 128

 # Stage-2 training (for each subject separately)
 python exps/nsd/train_stage2.py --epochs 6 --save_epochs 1 --batch_size 32 -s $subject (choices=[1, 2, 5, 7])
 ```   


 To get the evaluation scores for each subject based on the generated files of the test set, refer to the Benchmark [project](https://github.com/weihaox/BrainHub).

 With DATA_PATH set to "data/nsd", you should obtain the following structure:


#### Results
The results presented here improve upon those reported in the paper by (1) training the first stage in a cross-subject manner, (2) using curated COCO annotations (COCO_73k_annots_curated.npy) and (3) adjusting the decoder LLM’s inference hyperparameters (see the [configuration file](configs/configs_nsd.py)).
Results may vary slightly due to initialization and non-deterministic algorithms, but the variation remains low. Reported BrainDEC values are averaged over three runs. We compare our model with existing methods from the [BrainHub benchmark](https://github.com/weihaox/BrainHub):

| Method    | Eval | BLEU1 | BLEU4 | METEOR | ROUGE | CIDEr | SPICE | CLIPS | RefCLIPS |
|-----------|------|-------|-------|--------|-------|-------|-------|-------|----------|
| UMBRAE    | S1   | 59.44 | 19.03 | 19.45  | 43.71 | 61.06 | 12.79 | 67.78 | 73.54    |
| UMBRAE-S1 | S1   | 57.63 | 16.76 | 18.41  | 42.15 | 51.93 | 11.83 | 66.44 | 72.12    |
| BrainDEC  | S1   | 61.29 | 19.68 | 17.99  | 44.47 | 53.82 | 10.67 | 63.09 | 69.60    |
| BrainCap  | S1   | 55.96 | 14.51 | 16.68  | 40.69 | 41.30 | 9.06  | 64.31 | 69.90    |
| OneLLM    | S1   | 47.04 | 9.51  | 13.55  | 35.05 | 22.99 | 6.26  | 54.80 | 61.28    |
| SDRecon   | S1   | 36.21 | 3.43  | 10.03  | 25.13 | 13.83 | 5.02  | 61.07 | 66.36    |

| Method    | Eval | BLEU1 | BLEU4 | METEOR | ROUGE | CIDEr | SPICE | CLIPS | RefCLIPS |
|-----------|------|-------|-------|--------|-------|-------|-------|-------|----------|
| UMBRAE    | S2   | 59.37 | 18.41 | 19.17  | 43.86 | 55.93 | 12.08 | 66.46 | 72.36    |
| UMBRAE-S2 | S2   | 57.18 | 17.18 | 18.11  | 41.85 | 50.62 | 11.50 | 64.87 | 71.06    |
| BrainDEC  | S2   | 59.28 | 17.99 | 17.75  | 43.60 | 51.53 | 9.88  | 62.86 | 69.27    |
| BrainCap  | S2   | 53.80 | 13.03 | 15.90  | 39.96 | 35.60 | 8.47  | 62.48 | 68.19    |
| SDRecon   | S2   | 34.71 | 3.02  | 9.60   | 24.22 | 13.38 | 4.58  | 59.52 | 65.30    |

| Method    | Eval | BLEU1 | BLEU4 | METEOR | ROUGE | CIDEr | SPICE | CLIPS | RefCLIPS |
|-----------|------|-------|-------|--------|-------|-------|-------|-------|----------|
| UMBRAE    | S5   | 60.36 | 19.03 | 20.04  | 44.81 | 61.32 | 13.19 | 68.39 | 74.11    |
| UMBRAE-S5 | S5   | 58.99 | 18.73 | 19.04  | 43.30 | 57.09 | 12.70 | 66.48 | 72.69    |
| BrainDEC  | S5   | 61.82 | 19.57 | 18.70  | 44.63 | 57.65 | 11.32 | 64.03 | 70.26    |
| BrainCap  | S5   | 55.28 | 14.62 | 16.45  | 40.87 | 41.05 | 9.24  | 63.89 | 69.64    |
| SDRecon   | S5   | 34.96 | 3.49  | 9.93   | 24.77 | 13.85 | 5.19  | 60.83 | 66.30    |

| Method    | Eval | BLEU1 | BLEU4 | METEOR | ROUGE | CIDEr | SPICE | CLIPS | RefCLIPS |
|-----------|------|-------|-------|--------|-------|-------|-------|-------|----------|
| UMBRAE    | S7   | 57.20 | 17.13 | 18.29  | 42.16 | 52.73 | 11.63 | 65.90 | 71.83    |
| UMBRAE-S7 | S7   | 55.71 | 15.75 | 17.51  | 40.64 | 47.07 | 11.26 | 63.66 | 70.09    |
| BrainDEC  | S7   | 59.07 | 17.97 | 17.48  | 43.07 | 49.22 | 9.90  | 61.52 | 68.06    |
| BrainCap  | S7   | 54.25 | 14.00 | 15.94  | 40.02 | 37.49 | 8.57  | 62.52 | 68.48    |
| SDRecon   | S7   | 34.99 | 3.26  | 9.54   | 24.33 | 13.01 | 4.74  | 58.68 | 64.59    |


#### 4. Decoding Reading Text from EEG Signals
##### Configuration and data preparation
The same raw data and preprocessing presented in [EEG-To-Text](https://github.com/MikeWangWZHL/EEG-To-Text) are employed here.

* Update the configuration files "configs/configs_zuco.py" by specifying the paths similarly to the previous experiments.
* Download the following folders from [ZuCo v1.0](https://osf.io/q3zws/files/) and place them in the `DATA_PATH` specified in the config file (e.g., `data/zuco/task1-SR/Matlab_files`, etc.).
* Download `task1-NR/Matlab_files` from [ZuCo v2.0](https://osf.io/2urht/files/) and place it as `task2-NR-2.0/Matlab_files` inside `DATA_PATH`.
* Generate the preprocessed data using the following instructions:


With DATA_PATH set to data/zuco, for example, you should obtain the following structure after data preprocessing:

```
data
└── zuco
    ├── processed
    │   ├── task1-SR
    │   ├── task2-NR
    │   ├── task2-NR-2.0
    │   └── task3-TSR
    ├── task1-SR
    │   └── Matlab_files
    ├── task2-NR
    │   └── Matlab_files
    ├── task2-NR-2.0
    │   └── Matlab_files
    └── task3-TSR
        └── Matlab_files
```

##### Preprocessing and evaluation
```bash
# Data preparation
python exps/zuco/preprocess_data.py -t task1-SR
python exps/zuco/preprocess_data.py -t task2-NR
python exps/zuco/preprocess_data.py -t task3-TSR
python exps/zuco/preprocess_data_v2.py

# Build tokenizer for stage-1
python exps/zuco/build_tokenizer.py

# Training and evaluation
python exps/zuco/train_stage1.py --batch_size 128 --epochs 20
python exps/zuco/train_stage2.py --batch_size 16 --epochs 4
python exps/zuco/evaluation.py
```   

### TODO
- [x] Apply the proposed methodology for NSD datasets.
- [x] Test other LLM decoders.
- [x] Add experiments for decoding text from EEG signals.
- [x] Cross-subject training for NSD dataset.

### Notes
* The structure of this repository is in work progress
* Some parts of the code of this project are adapted from [InstructBlip](https://github.com/salesforce/LAVIS/blob/main/projects/instructblip/README.md), we thank the authors for their great work.

* In the comparison on perceived speech decoding, we used the same datasets and configuration setup in this [article](https://www.nature.com/articles/s41593-023-01304-9). Data preprocessing and preparation scripts are taken from this [link](https://github.com/HuthLab/semantic-decoding). We thank the authors for their great work.


### Citation
```bibtex
@article{hmamouche2024multimodal,
  title={A multimodal LLM for the non-invasive decoding of spoken text from brain recordings},
  author={Hmamouche, Youssef and Chihab, Ismail and Kdouri, Lahoucine and Seghrouchni, Amal El Fallah},
  journal={arXiv preprint arXiv:2409.19710},
  year={2024}
}
```
