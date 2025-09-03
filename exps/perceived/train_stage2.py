import os
from glob import glob
import sys
import argparse
import torch
from torch.optim import Adam
from tokenizers import Tokenizer

sys.path.insert(0, os.getcwd())

#from src.load_perceived_data import data_builder_from_file

from src.loaders.dataloader_perceived import data_builder_from_file
from src.loaders.dataloader_perceived import get_loaders as semantic_loader

import configs.configs_perceived as configs
from src.model import BrainDEC_V0
from src.transformers_src.Transformer import *


def save_checkpoint(model, model_name, cur_epoch, is_best=False):
    param_grad_dic = {
        k: v.requires_grad for (k, v) in model.named_parameters()
    }
    state_dict = model.state_dict()
    for k in list(state_dict.keys()):
        if k in param_grad_dic.keys() and not param_grad_dic[k]:
            # delete parameters that do not require gradient
            del state_dict[k]

    save_obj = {"model": state_dict,"epoch": cur_epoch}

    save_to = "%s/%s_%s.pth"%(configs.MODELS_TRAIN_PATH, model_name, ("best" if is_best else str (cur_epoch)))
    print("Saving checkpoint at epoch {} to {}.".format(cur_epoch, save_to))
    torch.save(save_obj, save_to)


def load_from_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    try:
        model.load_state_dict(checkpoint["model"])
    except RuntimeError as e:
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


def train (model, model_name, data_loader, batch_size, epochs = 10, save_epochs = 5, starting_epoch = 1):

    model.train()
    optim = Adam(model.parameters(), lr=0.001)

    len_data_batched = data_loader.size // batch_size
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR (optim, max_lr=0.001, epochs=epochs, steps_per_epoch=len_data_batched)

    best_loss = 100000

    for epoch in range(starting_epoch, epochs + 1):
        print ('-------- Epoch: ', epoch)
        mean_loss = 0

        for sample in data_loader:
            loss = model(sample)
            mean_loss += loss
            optim.zero_grad()
            loss.backward()
            optim.step()

        lr_scheduler.step()

        print (mean_loss / data_loader.size)
        if epoch % save_epochs == 0:
            best_loss = mean_loss
            save_checkpoint(model, model_name, epoch)
            test (model, model_name, epoch)


@torch.no_grad()
def test (model, model_name, epoch = ''):
    model.eval()

    test_files = glob("%s/%s/*test_perceived_speech*"%(configs.DATA_JSON_PATH, args.subject))
    print (test_files)

    for test_file in test_files:
        finename = os.path.basename(test_file).split('.')[0]
        finename_out = "results/perceived/%s/%s_%s.txt"%(args.subject, model_name + "_" + str(epoch), finename)
        f = open(finename_out, "w")

        data_loader = data_builder_from_file (test_file)
        f.write('chunk_number'+ ';###;' + 'predicted' + ';###;' + 'target' + '\n')
        for sample in data_loader:
            output_text = model.generate (sample)
            for chunk_number, predicted, target in zip (sample["chunk_number"], output_text, sample["text_output"]):
                f.write(str(chunk_number) + ';###;' + predicted.replace('\n', ' ') + ';###;' + target + "\n")
        f.close()


############ MAIN ##################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default = 64, type = int)
    parser.add_argument("-seed", default = 42)
    parser.add_argument("--model_name", "-m", help="Name of the model to train.", choices = ["V0"], default = "BrainDEC_V0")
    parser.add_argument('--test', action='store_true', help = "test the model")
    parser.add_argument('--retrain', action='store_true', help = "retrain from existing checkpoint")
    parser.add_argument("--device", default = "cuda", type = str)
    parser.add_argument("--starting_epoch", default = 1, type = int)
    parser.add_argument("--save_epochs", default = 1, type = int)
    parser.add_argument("--epochs", default = 10, type = int)
    parser.add_argument("--subject", '-s', choices=['S1', 'S2', 'S3'])
    parser.add_argument("--saving_path", default = "trained_models")
    parser.add_argument("--saved_checkpoint", "-c", type = str)
    parser.add_argument('--use_lora', action='store_true', help = "To use LoRA on the decoder LLM.")
    parser.add_argument('--load_in_4bit', action='store_true', help = "To load the llm quantized in 4 bits for inference.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)


    ############### Results paths ##############
    if not os.path.exists ("results/perceived"):
        os.mkdir ("results/perceived")

    if not os.path.exists ("results/perceived/%s"%args.subject):
        os.mkdir ("results/perceived/%s"%args.subject)


    ############### Different ROIs for each participant ##############
    if args.subject == "S1":
        src_fmri_features = 81126
    elif args.subject == "S2":
        src_fmri_features = 94251
    elif args.subject == "S3":
        src_fmri_features = 95556

    ############### Loading data ##############
    train_set, test_set = semantic_loader(args.subject, batch_size = args.batch_size, val_batch_size = args.batch_size)

    ################# Init fMRI Encoder #######################
    encoder_path = os.path.join (configs.MODELS_TRAIN_PATH, "DeconvBipartiteTransformerConv.pt")
    
    assert os.path.exists(encoder_path), "Encoder path does not exist."

    tokenizer = Tokenizer.from_file("./tools/tokenizer-perceived.json")
    vocab_len = tokenizer.get_vocab_size()
    encoder_model = DeconvBipartiteTransformerConv(configs.time_steps, 
                                                   configs.src_fmri_features_max, 
                                                   configs.max_size,\
                                                   vocab_len, 
                                                   configs.d_model, 
                                                   configs.d_ff, 
                                                   configs.N, 
                                                   configs.heads, 
                                                   args.device).float()
    
    encoder_model.load_state_dict(torch.load(encoder_path, weights_only=True))
    encoder_model = encoder_model.encoder
    
    ################ Model Init ##############
    mllm_model = BrainDEC_V0(encoder_model, 
                             configs, 
                             configs.src_fmri_features_max, 
                             load_in_4bit = args.load_in_4bit, 
                             lora = args.use_lora)
    
    args.model_name = args.model_name + "_" + args.subject + "_" + configs.LLM_name

    if args.use_lora:
         args.model_name = args.model_name + "_lora"


    ################ Training / Testing ##############
    if args.test:
        mllm_model = load_from_checkpoint(mllm_model, args.saved_checkpoint)
        args.model_name = args.saved_checkpoint.split ('/')[-1].split ('.')[0]
        test (mllm_model, args.model_name)
    else:
        if args.retrain:
            mllm_model = load_from_checkpoint(mllm_model, args.saved_checkpoint)

        train (mllm_model,
               args.model_name,
               train_set,
               args.batch_size,
               epochs = args.epochs,
               save_epochs = args.save_epochs,
               starting_epoch = args.starting_epoch)
