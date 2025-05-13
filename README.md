### A Multimodal LLM for the Non-Invasive Decoding of Spoken Text from Brain Recordings


![model](figs/overview_1.png)

 In this paper, we propose and end-to-end multimodal large language model for decoding the spoken text in a human-human or human-robot interactions. The proposed architecture is founded on  ($i$) an encoder derived from a specific transformer with the incorporation of an augmented embedding layer for the encoder and a better-adjusted attention mechanism than that present in the state of the art, and ($ii$) a frozen LLM adapted via instruction tuning to align the embedding of the input modalities to decode the output text. A benchmark in performed on two publicly available datasets where fMRI brain activity and conversational signals are recorded synchronously.

## Requirements
* python 3.9.19
* Required packages:  ```python install -r requirement.txt```
* vicuna-7b-v1.3 from https://huggingface.co/lmsys/vicuna-7b-v1.3 in the folder 'llms' (or other LLMs such as Meta-llama3.2-8b-Instruct)
* CLIP in the main folder: git clone https://github.com/openai/CLIP
* To use other data storage paths, change the configuration file: src/configs/convers/configs.py

## Experiments

This benchamrk contains three experiments associated to three different tasks and datasets:
* __Spoken text decoding (convers)__: Multimodal spoken text decoding during conversations (main task of this work).
* __Perceived speech decoding (perceived)__: Decoding the textual content of listened stories.
* __Brain captioning (nsd)__: Decoding the captions of viewed images using the Natural Scenes Dataset datasets.

In the following, we detail the steps to conduct or reproduce the results of each experiment.

### Spoken text decoding
#### Configuration
- Update the configuration files "srs/configs/perceived/configs.py" by specifying the following paths:
    * DATA_PATH (ex. data/convers)
    * RAW_FMRI_DATA_PATH (ex. data/fmri_convers)
    * MODELS_TRAIN_DIR (ex. trained_models/convers)
    * LLM_DIR (ex. LLMs/Meta-llama3.2-8b-Instruct)

- Download the version 2.2.0 of the Convers datasets from the OpenNeuro platform (https://openneuro.org/datasets/ds001740/versions/2.2.0)
 in the path 'RAW_FMRI_DATA_PATH' specified in the config file.

- Create a folder named $DATA_PATH/raw_data/transcriptions" and upload  the raw Transcriptions from the Ortolang platform into it:
https://www.ortolang.fr/market/corpora/convers/v2


#### Preprocessing raw data:
```bash
python exps/convers/process_raw_bold_signal.py --n_rois 200 # Parcellation using 200 ROIs
python exps/convers/data_builder_tools/split_bold_files.py  # Processing raw 4D voxel BOLD signals and segmenting them into fixed-duration chunks
python exps/convers/data_builder_tools/textgrid_to_text.py # Processing transcription files (conversations) and segmenting them into fixed-duration text sequences

python exps/convers/data_builder_tools/build_data.py # Using json files to save paths of bold chunks and the [input, output] text for instruction tuning
python exps/convers/data_builder_tools/build_tokenizer.py # Building the tokenizer for the first stage of training
```

#### Training and evaluation
```bash
python  exps/convers/train_stage1.py -m DeconvBipartiteTransformerConv --batch_size 128 # Stage1: training the DeconvBipartite Transformer
python  exps/convers/train_stage2.py --batch_size 32 # Stage1: training the overall MLLM with the pre-trained DeconvBipartite encoder.
python exps/convers/evaluation.py  # Evaluate the results of the test set and report the scores
```   


### Perceived speech decoding
#### Configuration
- Update the configuration files "srs/configs/perceived/configs.py" by specifying the following paths:

    * RAW_FMRI_DATA_PATH (ex. data/perceived)
    * MODELS_TRAIN_DIR (ex. trained_models/perceived)
    * LLM_DIR (ex. LLMs/Meta-llama3.2-8b-Instruct)


####  Data preparation
* In the folders "DATA_TRAIN_DIR" and "DATA_TEST_DIR" (see the config file), download the training and test datasets as outlined in the project [semantic-decoding](https://github.com/HuthLab/semantic-decoding).

#### Training and evaluation
* To reproduce the results of this experiment:
```bash
python exps/perceived/prepare_Tang2023_datasets.py -s $subject (choices=['S1', 'S2', 'S3'])
python exps/perceived/build_tokenizer.py
python exps/perceived/train_stage1.py --batch_size 128 -s $subject (choices=['S1', 'S2', 'S3'])
python exps/perceived/train_stage2.py --batch_size 32 --epochs 100 --save-epoch 20 -s $subject (choices=['S1', 'S2', 'S3'])
python exps/perceived/evaluation.py $subject (choices=['S1', 'S2', 'S3'])
```   

##  Brain captioning - BrainHub benchmark on NSD dataset
This a comparison with brain understanding benchmark ([BrainHub](https://github.com/weihaox/BrainHub)), based on Natural Scenes Dataset [NSD](https://naturalscenesdataset.org/) and [COCO](https://cocodataset.org).

#### Configuration
- The processed datasets are available in [here](https://huggingface.co/datasets/pscotti/naturalscenesdataset).
- Download the datasets using this [script](https://github.com/weihaox/UMBRAE/blob/main/umbrae/download_data.sh).
- Download COCO annotations from this [link](https://huggingface.co/datasets/pscotti/naturalscenesdataset/blob/main/COCO_73k_annots.npy), and put it in the folder 'tools'
- Update the configuration file 'src/configs.nsd/configs_nsd.py' to specify the paths, and eventually to modify the hyperparameters.
- To train and evaluate the model:
```bash
python exps/nsd/main.py --epochs 6 --save_epochs 1 --batch_size 32 -s $subject (choices=[1, 2, 5, 7])
```   
- To get the evaluation scores for each subject based on the generated files of the test set, refer to the Benchmark [project](https://github.com/weihaox/BrainHub).
- Please be aware that the training process involves non-deterministic algorithms even with fixed seed, which can lead to slightly different results on each run.
In our case, we ran the training procedure 5 times for 6 epochs each, and selected the best result.
 **TODO**: Retrain the models in a fully deterministic mode to reproduce the same exact results, but this may impact the overall performance.


#### Results
We adapted the previous architecture to work with Llama-3.2-8B-Instruct and Lora finetuning for brain captioning. We also added Mistral-8b-Instruct for comparison. Unlike existing methods, our model uses only brain fMRI signals and text during training, without leveraging VLMs or brain-image alignment. We trained our models for each subject, and the results are promising.  The model achieved very competitive results, yielding, in several cases, to the first or second best scores based on  BLEU1, BLEU4, ROUGE, and METEOR. The generated caption on the test are in the folder "results/nsd". As future work, we aim to train our model in a cross-subject manner. Let BrainDEC be the abreviation of our method. The following table compares the results obtained with existing methods.


| Method            | Eval | BLEU1 | BLEU4 | METEOR | ROUGE |  RefCLIPS |
|-------------------|------|-------|-------|--------|-------|----------|
| UMBRAE            | S1   | 59.44 | 19.03 | __19.45__  | __43.71__  | __73.54__    |
| UMBRAE-S1         | S1   | 57.63 | 16.76 | 18.41  | 42.15 |  72.12    |
| BrainCap          | S1   | 55.96 | 14.51 | 16.68  | 40.69  |  69.90    |
| OneLLM            | S1   | 47.04 | 9.51  | 13.55  | 35.05 |  61.28    |
| SDRecon           | S1   | 36.21 | 3.43  | 10.03  | 25.13  | 66.36    |
| BrainDEC-llama3-8b (ours) | S1   | __59.93__ | __20.05__ | 18.57  | __43.71__ |  69.66    |
| BrainDEC-mistral-8b (ours)| S1   | 58.92 | 18.25 | 42.61  | __43.71__ |  68.20    |

| Method            | Eval | BLEU1 | BLEU4 | METEOR | ROUGE | RefCLIPS |
|-------------------|------|-------|-------|--------|-------|----------|
| UMBRAE            | S2   | __59.37__ | 18.41 | __19.17__  | __43.86__ |  __72.36__    |
| UMBRAE-S2         | S2   | 57.18 | 17.18 | 18.11  | 41.85 |   71.06    |
| BrainCap          | S2   | 53.80 | 13.03 | 15.90  | 39.96 |   68.19    |
| SDRecon           | S2   | 34.71 | 3.02  | 9.60   | 24.22 |  65.30    |
| BrainDEC-llama3-8b | S2   | 57.85 | __18.52__ | 17.61  | 43.11 | 68.01    |
| BrainDEC-mistral-8b | S2   | 58.26 |17.68  | 17.21  | 42.52 | 67.18   |


| Method            | Eval | BLEU1 | BLEU4 | METEOR | ROUGE | RefCLIPS |
|-------------------|------|-------|-------|--------|-------|----------|
| UMBRAE            | S5   | 60.36 | 19.03 | 20.04  | 44.81 |   __74.11__    |
| UMBRAE-S5         | S5   | 58.99 | 18.73 | 19.04  | 43.30 |   72.69    |
| BrainCap          | S5   | 55.28 | 14.62 | 16.45  | 40.87 | 69.64    |
| SDRecon           | S5   | 34.96 | 3.49  | 9.93   | 24.77 |  66.30    |
| BrainDEC-llama3-8b | S5   | 60.52 | __20.19__ | __18.82__  | __44.69__ | 69.57    |
| BrainDEC-mistral-8b | S5   | __60.92__ | 20.12 | __18.83__ | 44.47 | 70.14   |


| Method            | Eval | BLEU1 | BLEU4 | METEOR | ROUGE |  RefCLIPS |
|-------------------|------|-------|-------|--------|-------|----------|
| UMBRAE            | S7   | 57.20 | 17.13 | __18.29__  | __42.16__ | __71.83__    |
| UMBRAE-S7         | S7   | 55.71 | 15.75 | 17.51  | 40.64 | 70.09    |
| BrainCap          | S7   | 54.25 | 14.00 | 15.94  | 40.02 |  68.48    |
| SDRecon           | S7   | 34.99 | 3.26  | 9.54   | 24.33 |   64.59    |
| BrainDEC-llama3-8b | S7   | 56.98 | __17.77__ | 17.42  | 42.01 | 67.08    |
| BrainDEC-mistral-8b | S7   |__57.45__  |17.40  | 16.90 | 41.78 | 66.71   |


#### TODO
- [x] Apply the proposed methodology for NSD datasets - Brain captioning benchmark.
- [x] Test other LLM decoders
- [ ] Cross-subject training for NSD dataset.
- [ ] Add experiments for decoding text from EEG signals.


## Notes
* The structure of this repository is in work progress
* Some parts of the code of this project are adapted from [InstructBlip](https://github.com/salesforce/LAVIS/blob/main/projects/instructblip/README.md), we thank the authors for their great work.

* In the comparison on perceived speech decoding, we used the same datasets and configuration setup in this [article](https://www.nature.com/articles/s41593-023-01304-9). Data preprocessing and preparation scripts are taken from this [link](https://github.com/HuthLab/semantic-decoding). We thank the authors for their great work.


## Citation
```bibtex
@article{hmamouche2024multimodal,
  title={A multimodal LLM for the non-invasive decoding of spoken text from brain recordings},
  author={Hmamouche, Youssef and Chihab, Ismail and Kdouri, Lahoucine and Seghrouchni, Amal El Fallah},
  journal={arXiv preprint arXiv:2409.19710},
  year={2024}
}
```
