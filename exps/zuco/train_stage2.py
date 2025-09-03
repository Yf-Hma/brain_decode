import argparse
import torch
import os, sys
from transformers import set_seed
from tokenizers import Tokenizer

import trainer
sys.path.insert(0, os.getcwd())

from src.loaders.dataloader_zuco import get_dada_loaders_all as zuco_loader
from configs import configs_zuco as configs
from src.model import BrainDEC_V0
from src.transformers_src.Transformer import DeconvBipartiteTransformerConv

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--val_batch_size", default=128, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--version", default=2, type=int, choices=[1, 2], help='ZuCo dataset version.')
    parser.add_argument("--model_name", "-m", help="Name of the model to train.")
    parser.add_argument('--test', action='store_true', help="test the model")
    parser.add_argument('--use_lora', action='store_true', help="using lora to fintune the decoder.")
    parser.add_argument('--retrain', action='store_true', help="retrain from existing checkpoint")
    parser.add_argument("--lr", default=0.00001, type=float)
    parser.add_argument("--starting_epoch", default=1, type=int)
    parser.add_argument("--save_epochs", default=1, type=int)
    parser.add_argument("--epochs", default=4, type=int)
    parser.add_argument("--saved_checkpoint", "-s", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--type", "-t", type=str, choices=['normal', "deconv", "cnn"], default="normal")
    parser.add_argument('--load_in_4bit', action='store_true', help="to load the llm quantized in 4 bits for inference.")
    parser.add_argument('--gpu', default=None, help='GPU id to use.')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:3456', type=str, help='')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='')
    parser.add_argument('--world_size', default=1, type=int, help='')
    parser.add_argument('--distributed', action='store_true', default=False)
    parser.add_argument('--num_workers', type=int, default=1, help='')

    args=parser.parse_args()

    ################## Parameters ######################
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed_data_path=configs.PROCESSED_DATA_PATH
    args.saving_path=configs.MODELS_TRAIN_PATH
    set_seed(args.seed)

    if not os.path.exists ("results/zuco"):
        os.mkdir ("results/zuco")

    ################## Loading data ######################
    train_set, test_set = zuco_loader (configs.PROCESSED_DATA_PATH, args.batch_size,args.batch_size)
    
    # for item in train_set:
    #     print (len (item["text_output"]), len (item["signal"]))

    # for item in test_set:
    #     print (len (item["text_output"]), item["signal"].shape)
        
    # exit ()
 
    ############# Init Brain Signal Encoder #############
    tokenizer=Tokenizer.from_file("./tools/tokenizer-zuco.json")
    vocab_len=tokenizer.get_vocab_size()

    encoder_model=DeconvBipartiteTransformerConv(configs.time_steps, 
                                         configs.src_eeg_features, 
                                         configs.max_size,\
                                         vocab_len, 
                                         configs.d_model, 
                                         configs.d_ff, 
                                         configs.N, 
                                         configs.heads, 
                                         args.device).float()
    
    encoder_path=f'{configs.MODELS_TRAIN_PATH}/DeconvBipartiteTransformerConv_v{args.version}.pt'
    assert os.path.exists(encoder_path), "Encoder path does not exist."
    
    encoder_model.load_state_dict(torch.load(encoder_path, weights_only=True))
    encoder_model = encoder_model.encoder

    ################ Model Init ##############
    mllm_model=BrainDEC_V0(encoder_model, 
                           configs, 
                           configs.src_eeg_features, 
                           device=args.device, 
                           load_in_4bit=args.load_in_4bit, 
                           lora=args.use_lora)

    ################## Model Training/Testing ######################
    if args.retrain or args.test:
        assert os.path.exists(args.saved_checkpoint), "Path to checkpoint does not exist."
        model_name=args.saved_checkpoint.split('/')[-1].split('.')[0]
        checkpoint=torch.load(args.saved_checkpoint, map_location=device)
        mllm_model.load_state_dict(checkpoint["model"], strict=False)

    if args.test:
        results_fname=args.saved_checkpoint.split ('/')[-1].split ('.')[0]
        trainer.test (mllm_model, results_fname, test_set)
    else:
        out_name=f"BrainDEC_zuco_{configs.LLM_name}_v{args.version}"
        trainer.train_single(mllm_model, out_name, train_set, test_set, args, configs.MODELS_TRAIN_PATH)
