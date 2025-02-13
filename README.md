### A Multimodal LLM for the Non-Invasive Decoding of Spoken Text from Brain Recordings


![model](figs/overview_1.png)

 In this paper, we propose and end-to-end multimodal large language model for decoding the spoken text in a human-human or human-robot interactions. The proposed architecture is founded on  ($i$) an encoder derived from a specific transformer with the incorporation of an augmented embedding layer for the encoder and a better-adjusted attention mechanism than that present in the state of the art, and ($ii$) a frozen LLM adapted via instruction tuning to align the embedding of the input modalities to decode the output text. A benchmark in performed on two publicly available datasets where fMRI brain activity and conversational signals are recorded synchronously.

## Requirements

* python 3.9.19
* Required packages:  ```python install -r requirement.txt```
* vicuna-7b-v1.3 from https://huggingface.co/lmsys/vicuna-7b-v1.3 in the folder 'llm'
* CLIP in the main folder: git clone https://github.com/openai/CLIP
* To use other data storage paths, change the configuration file: src/config.py

## Preprocessing raw data:
#### Preprocessing raw BOLD signal:

* Create a folder named “data/raw_data/fmri_bold” and upload the version 2.2.0 of the datasets repository from the OpenNeuro platform into it:
https://openneuro.org/datasets/ds001740/versions/2.2.0

* Create a folder named “data/raw_data/transcriptions" and upload  the raw Transcriptions from the Ortolang platform into it:
https://www.ortolang.fr/market/corpora/convers/v2


* Processing raw 4D voxel BOLD signals and segmenting them into fixed-duration sequences:
```bash
python src/process_raw_bold_signal.py --n_rois 200
python src/data_builder_tools/split_bold_files.py
```

#### Processing transcription files
* Processing transcription files (conversations) and segmenting them into fixed-duration text sequences:
```bash
python src/data_builder_tools/textgrid_to_text.py
```

#### Building training and test data
Using json files to associate paths of bold chunks and the associated input and response text sentences for instruction tuning:
```bash
python src/data_builder_tools/build_data.py
python src/data_builder_tools/build_tokenizer.py
```


## Stage 1: Train Transformer-based models

Example: training and testing DeconvBipartiteTransformer
```bash
python  train_transformers.py -m DeconvBipartiteTransformerConv
python  train_transformers.py -m DeconvBipartiteTransformerConv --test
```   


## Stage 2: Train and test the MLLM using the trained fmri encoder
Note that this requires a trained DeconvBipartiteTransformer (ex. MllmBrainToTextV0).
```bash
python  trainer.py -m MllmBrainToTextV0
python  trainer.py -m MllmBrainToTextV0 -s trained_models/MllmBrainToTextV0_200_spoken_300.pth --test
```

* To use multiple GPUs, use trainer_dist.py instead of trainer.py
* To use the version with a quantized LLM for inference, include the 'load_in_4bit' parameter. Example:

```bash
python  trainer.py -m MllmBrainToTextV0 -s trained_models/MllmBrainToTextV0_200_spoken_300.pth --test --batch_size 16 --load_in_4bit
```   

## Evaluation
```bash
python src/evaluation.py
```

##  Perceived Speech Decoding
* Download training and test datasets in the folder "data/semantic_decoding" as outlined in the project [semantic-decoding](https://github.com/HuthLab/semantic-decoding).
* To run the experiments on this dataset, run the following commands from "comparison_semantic_perceived_GPT_2023" folder:
```bash
python src/prepare_Tang2023_datasets.py -s subject (choices=['S1', 'S2', 'S3'])
python src/build_tokenizer.py
python train_transformers.py s subject
python trainer.py -s subject
python trainer.py --test -s subject --saving_path trained_model_path
```   

##  Brain captioning - BrainHub benchmark on NSD dataset
This a comparison with brain understanding benchmark ([BrainHub](https://github.com/weihaox/BrainHub)), based on Natural Scenes Dataset [NSD](https://naturalscenesdataset.org/) and [COCO](https://cocodataset.org).

#### Experimental setup
- The processed datasets are available in [here](https://huggingface.co/datasets/pscotti/naturalscenesdataset).
- Download the datasets using this [script](https://github.com/weihaox/UMBRAE/blob/main/umbrae/download_data.sh).
- Download COCO annotations from this [link](https://huggingface.co/datasets/pscotti/naturalscenesdataset/blob/main/COCO_73k_annots.npy), and put it in the folder 'tools'
- Update the configuration file 'src/configs_nsd' to specify the datasets and llama3.2 paths, and eventually to modify the hyperparameters.
- To train and evaluate the model, from the folder 'comparison_NSD' execute the following scripts (example for subject 1): ```python main.py --subj 1```.  
- Trained models are available in this [link](https://drive.google.com/file/d/1bzSz4oQY3YDEq7jh7JfsioZ_mgBceBDU/view?usp=sharing).
- To get the evaluation scores for each subject based on the generated captions of the test set, refer to the Benchmark [project](https://github.com/weihaox/BrainHub).

#### Results
We adapted the previous architecture to work with Llama-3.2-8B-Instruct and Lora finetuning (instead of frozen Vicuna-7b) for brain captioning. Unlike existing methods, the model uses only brain fMRI signals and text during training, without leveraging VLMs or brain-image alignment. We trained our models for each subject, and the results are promising.  The model achieved very competitive results, yielding, in several cases, to the first or second best scores based on  BLEU1, BLEU4, ROUGE, and METEOR. The generated caption on the test are in the folder "comparison_NSD/results". As future work, we aim to train our model in a cross-subject manner. Let BrainDEC be the abreviation of our method. The following table compares the results obtained with existing methods.



| Method            | Eval | BLEU1 | BLEU4 | METEOR | ROUGE | CIDEr | SPICE | CLIPS | RefCLIPS |
|-------------------|------|-------|-------|--------|-------|-------|-------|-------|----------|
| UMBRAE            | S1   | 59.44 | 19.03 | 19.45  | 43.71 | 61.06 | 12.79 | 67.78 | 73.54    |
| UMBRAE-S1         | S1   | 57.63 | 16.76 | 18.41  | 42.15 | 51.93 | 11.83 | 66.44 | 72.12    |
| BrainCap          | S1   | 55.96 | 14.51 | 16.68  | 40.69 | 41.30 | 9.06  | 64.31 | 69.90    |
| OneLLM            | S1   | 47.04 | 9.51  | 13.55  | 35.05 | 22.99 | 6.26  | 54.80 | 61.28    |
| SDRecon           | S1   | 36.21 | 3.43  | 10.03  | 25.13 | 13.83 | 5.02  | 61.07 | 66.36    |
| BrainDEC-S1 (ours)| S1   | 59.93 | 20.05 | 18.57  | 43.71 | 55.90 | 10.48 | 63.51 | 69.66    |

| Method            | Eval | BLEU1 | BLEU4 | METEOR | ROUGE | CIDEr | SPICE | CLIPS | RefCLIPS |
|-------------------|------|-------|-------|--------|-------|-------|-------|-------|----------|
| UMBRAE            | S2   | 59.37 | 18.41 | 19.17  | 43.86 | 55.93 | 12.08 | 66.46 | 72.36    |
| UMBRAE-S2         | S2   | 57.18 | 17.18 | 18.11  | 41.85 | 50.62 | 11.50 | 64.87 | 71.06    |
| BrainCap          | S2   | 53.80 | 13.03 | 15.90  | 39.96 | 35.60 | 8.47  | 62.48 | 68.19    |
| SDRecon           | S2   | 34.71 | 3.02  | 9.60   | 24.22 | 13.38 | 4.58  | 59.52 | 65.30    |
| BrainDEC-S2 (ours)| S2   | 57.85 | 18.52 | 17.61  | 43.11 | 48.31 | 9.74  | 61.55 | 68.01    |


| Method            | Eval | BLEU1 | BLEU4 | METEOR | ROUGE | CIDEr | SPICE | CLIPS | RefCLIPS |
|-------------------|------|-------|-------|--------|-------|-------|-------|-------|----------|
| UMBRAE            | S5   | 60.36 | 19.03 | 20.04  | 44.81 | 61.32 | 13.19 | 68.39 | 74.11    |
| UMBRAE-S5         | S5   | 58.99 | 18.73 | 19.04  | 43.30 | 57.09 | 12.70 | 66.48 | 72.69    |
| BrainCap          | S5   | 55.28 | 14.62 | 16.45  | 40.87 | 41.05 | 9.24  | 63.89 | 69.64    |
| SDRecon           | S5   | 34.96 | 3.49  | 9.93   | 24.77 | 13.85 | 5.19  | 60.83 | 66.30    |
| BrainDEC-S5 (ours)| S5   | 60.52 | 20.19 | 18.82  | 44.69 | 55.67 | 10.53 | 63.22 | 69.57    |


| Method            | Eval | BLEU1 | BLEU4 | METEOR | ROUGE | CIDEr | SPICE | CLIPS | RefCLIPS |
|-------------------|------|-------|-------|--------|-------|-------|-------|-------|----------|
| UMBRAE            | S7   | 57.20 | 17.13 | 18.29  | 42.16 | 52.73 | 11.63 | 65.90 | 71.83    |
| UMBRAE-S7         | S7   | 55.71 | 15.75 | 17.51  | 40.64 | 47.07 | 11.26 | 63.66 | 70.09    |
| BrainCap          | S7   | 54.25 | 14.00 | 15.94  | 40.02 | 37.49 | 8.57  | 62.52 | 68.48    |
| SDRecon           | S7   | 34.99 | 3.26  | 9.54   | 24.33 | 13.01 | 4.74  | 58.68 | 64.59    |
| BrainDEC-S7 (ours)| S7   | 56.98 | 17.77 | 17.42  | 42.01 | 45.53 | 9.38  | 60.25 | 67.08    |


## TODO
- [x] Provide pretrained checkpoints.
- [x] Apply the proposed methodology for NSD datasets - Brain captioning benchmark.
- [ ] Cross-subject training for NSD dataset.
- [ ] Test other LLM decoders


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
