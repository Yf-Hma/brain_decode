import argparse
import torch
import os, sys
from transformers import set_seed

current=os.path.dirname(os.path.realpath(__file__))
parent=os.path.dirname(current)
main=os.path.dirname(parent)
sys.path.append(main)

from exps.zuco.load_zuco_data import get_loaders
from src.models.models_eeg import BrainDEC_V0
import trainer
import src.configs.zuco.configs as configs

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int)
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
    parser.add_argument("--epochs_max", default=10, type=int)
    parser.add_argument("--saved_checkpoint", "-s", type=str)
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
    args.saving_path=configs.MODELS_TRAIN_DIR
    set_seed(args.seed)

    ################## Loading data ######################
    train_set, test_set=get_loaders (processed_data_path, args.version, args.batch_size,args.val_batch_size)

    ################ Model Init ##############
    eeg_encoder_checkpoint=f'{configs.MODELS_TRAIN_DIR}/DeconvBipartiteTransformerConv_v{args.version}.pt'
    model=BrainDEC_V0(configs, eeg_encoder_checkpoint, device=device, load_in_4bit=args.load_in_4bit, use_lora=args.use_lora)

    ################## Model Training/Testing ######################
    if args.retrain or args.test:
        assert os.path.exists(args.saved_checkpoint), "Path to checkpoint does not exist."
        model_name=args.saved_checkpoint.split('/')[-1].split('.')[0]
        checkpoint=torch.load(args.saved_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"], strict=False)
                              
    if args.test:
        results_fname=args.saved_checkpoint.split ('/')[-1].split ('.')[0]
        trainer.test (model, results_fname, test_set)
    else:
        out_name=f"BrainDEC_zuco_{configs.LLM_name}_v{args.version}"
        trainer.train_single(model, out_name, train_set, test_set, args, configs.MODELS_TRAIN_DIR)


