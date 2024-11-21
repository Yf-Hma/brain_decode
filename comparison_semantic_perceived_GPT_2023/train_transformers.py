import torch
import argparse
from tokenizers import Tokenizer
from src.load_data import data_builder, data_builder_from_file

import os
import sys
import inspect
from glob import glob

import src.config as config

# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0, parentdir) 

from src.transformers_src.Transformer import *
from src.transformers_src.Train_Function import train_model
from src.transformers_src.Inference import inference


models_dict = {
'Transformer':Transformer,
'CNNTransformer':CNNTransformer,
'DuplexTransformerConv':DuplexTransformerConv,
'BipartiteTransformerConv':BipartiteTransformerConv,
'DeconvBipartiteTransformerConv':DeconvBipartiteTransformerConv,
}
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-m", help="Name of the model to train.", choices = models_dict.keys(), default = "DeconvBipartiteTransformerConv")
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--retrain", action='store_true')
    parser.add_argument("--subject", '-s', choices=['S1', 'S2', 'S3'])
    parser.add_argument("-seed", default = 3)
    parser.add_argument("--saving_path", default = "trained_models")
    args = parser.parse_args()

    name = args.model_name + "_" + args.subject
    torch.manual_seed(args.seed)

    lr = config.lr
    d_model = config.d_model
    d_ff = config.d_ff
    heads = config.heads
    N = config.N
    epochs = config.epochs
    batch_size = config.batch_size
    time_steps = config.time_steps
    wandb_log = config.wandb_log
    max_size = config.max_size


    src_fmri_features = config.src_fmri_features_S1

    if args.subject == "S2":
        src_fmri_features = config.src_fmri_features_S2
    elif args.subject == "S3":
        src_fmri_features = config.src_fmri_features_S3
        

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pad_token_id, sos_token_id, eos_token_id = 0, 1, 2

    tokenizer = Tokenizer.from_file("tools/tokenizer-trained.json")
    vocab_len = tokenizer.get_vocab_size()

    ################ Datasets ##############
    data_loader = data_builder(args.subject, batch_size)

    ################ Model Init ##############
    model_class = models_dict[args.model_name]
    model = model_class(time_steps, src_fmri_features, max_size, vocab_len, d_model, d_ff, N, heads, device).to(device)
    model = model.float()
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    ################ Model Training/Testing ##############

    checkpoint_filename = '%s/%s_%s.pt'%(args.saving_path, args.model_name, args.subject)

    if args.test:
        # TODO: increase batch size in model testing
        batch_size = 1

        test_files = glob("data/%s/*test_*"%args.subject)
        model.load_state_dict(torch.load(checkpoint_filename, weights_only=True))

        for test_file in test_files:
            finename = os.path.basename(test_file).split('.')[0]
            finename_out = "results/%s_%s_%s.txt"%(args.model_name, args.subject, finename)
            os.system(f'rm {finename_out}')
            data_loader = data_builder_from_file (test_file, batch_size=1)
            inference(model, finename_out, tokenizer, vocab_len, data_loader, sos_token_id, eos_token_id, pad_token_id, max_size, device)

    else:
        if args.retrain:
            model.load_state_dict(torch.load(checkpoint_filename, weights_only=True))
        train_model(args.saving_path, name, model, data_loader["train"], batch_size, optim, epochs, lr, N, sos_token_id, eos_token_id, pad_token_id,max_size, tokenizer, device, wandb_log)
