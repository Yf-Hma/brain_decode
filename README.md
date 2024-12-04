# Implementation code of the paper: A Multimodal LLM for the Non-Invasive Decoding of Spoken Text from Brain Recordings

![model](figs/MLLM_V2.png)

## Requirements

* python 3.9.19
* Required packages:  ```python install -r requirement.txt```
* vicuna-7b-v1.3 from https://huggingface.co/lmsys/vicuna-7b-v1.3 in the folder 'llm'
* CLIP in the main folder: git clone https://github.com/openai/CLIP
* To use other data storage paths, change the configuration file: src/config.py

## Preprocessing raw:
#### Preprocessing raw fMRI data:

* Create a folder named “data/raw_data/fmri_bold” and upload the version 2.2.0 of the datasets repository from the OpenNeuro platform into it:
https://openneuro.org/datasets/ds001740/versions/2.2.0

* Create a folder named “data/raw_data/transcriptions" and upload  the raw Transcriptions from the Ortolang platform into it:
https://www.ortolang.fr/market/corpora/convers/v2?path=%2FTranscriptions


* Processing raw 4D voxel BOLD signals and segmenting them into fixed-duration sequences:
```bash
python src/process_raw_bold_signal.py --n_rois 200 --data_path data/raw_data/fmri_bold/ds001740-2.2.0  -o data/raw_data/fmri_bold
python src/data_builder_tools/split_bold_files.py --fmri_data_path data/raw_data/fmri_bold/fMRI_data_200
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


## Train and test Transformer-based models

```bash
python  train_transformers.py [-h] [-seed SEED] [--test] [--retrain] [--load]
                             [--model_name {Transformer,CNNTransformer,DuplexTransformerConv,BipartiteTransformerConv,DeconvBipartiteTransformerConv}]
```   
Example: training and testing DeconvBipartiteTransformer
```bash
python  train_transformers.py -m DeconvBipartiteTransformerConv
python  train_transformers.py -m DeconvBipartiteTransformerConv --test
```   


## Train and test MLLMs
Note that this requires a trained DeconvBipartiteTransformer.
```bash
  python trainer.py
  Arguments:
    --model_name {MllmBrainToTextV0, MllmBrainToTextV1, MllmBrainToTextV2}   name of the model to train.
    --test                To test the model
    --retrain             retrain from existing checkpoint
    --starting_epoch      starting epoch in case of retrain is True
    --save_epochs         number of epochs before saving the checkpoint
    --epochs              number of training epochs (default 300)
    --batch_size          Batch size (default 32)
    --saved_checkpoint    Path of the trained model in case of “retrain“ or “test“ is True
    --load_in_4bit       to load the llm quantized in 4 bits for inference.")
```

Example
```bash
python  trainer.py -m MllmBrainToTextV0
python  trainer.py -m MllmBrainToTextV0 -s trained_models/MllmBrainToTextV0_200_spoken_300.pth --test
```

* To use multiple GPUs, use trainer_dist.py instead of trainer.py
* To use the version with a quantized LLM for inference, include the 'load_in_4bit' parameter. Example:

Example
```bash
python  trainer.py -m MllmBrainToTextV0 -s trained_models/MllmBrainToTextV0_200_spoken_300.pth --test --batch_size 16 --load_in_4bit
```   

## Evaluation and Benchmarking
```bash
python src/evaluation.py
```


## Benchmark on perceived speech decoding:
* Download training and test datasets in the folder "data/semantic_decoding" as outlined in: https://github.com/HuthLab/semantic-decoding
* To run the experiments on this dataset, run the following commands from "comparison_semantic_perceived_GPT_2023" folder:
```bash
python src/prepare_Tang2023_datasets.py -s subject (choices=['S1', 'S2', 'S3'])
python src/build_tokenizer.py
python train_transformers.py s subject
python trainer.py -s subject
python trainer.py --test -s subject --saving_path trained_model_path
```   

## TODO
- [ ] Provide pretrained checkpoints.
- [ ] Add NSD datasets.
- [ ] Test other LLMs (Mistral, LLama3, etc.)


## Notes
* The structure of this repository is in work progress
* Some parts of the code of this project are adapted from InstructBlip (https://github.com/salesforce/LAVIS/blob/main/projects/instructblip/README.md), we thank the authors for their great work.

* In the comparison on perceived speech decoding, we used the same datasets and configuration setup in https://www.nature.com/articles/s41593-023-01304-9. Data preprocessing and preparation scripts are taken from :https://github.com/HuthLab/semantic-decoding. We thank the authors for their great work.
