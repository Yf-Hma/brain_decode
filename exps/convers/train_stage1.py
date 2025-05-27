import torch
import argparse, os
from tokenizers import Tokenizer
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
main = os.path.dirname(parent)
sys.path.append(main)

import src.configs.convers.configs as configs
from src.load_convers_data import data_builder
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
    parser.add_argument("--seed", default = 3)
    parser.add_argument("--batch_size", default = 128, type = int)
    parser.add_argument("--epochs", default = 200, type = int)
    parser.add_argument("--lr", default = 0.0001, type = float)
    parser.add_argument("--type", "-t", type = str, default = 'spoken', choices = ['spoken', 'perceived'])
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    if not os.path.exists('results/convers'):
        os.mkdir('results/convers')
    
    batch_size = args.batch_size
    wandb_log = False
    epochs = args.epochs
    lr = args.lr
    
    src_fmri_features = configs.src_fmri_features
    time_steps = configs.time_steps
    max_size = configs.max_size
    d_model = configs.d_model
    heads = configs.heads
    d_ff = configs.d_ff
    N = configs.N

    pad_token_id, sos_token_id, eos_token_id = 0, 1, 2

    tokenizer = Tokenizer.from_file("./tools/tokenizer-convers.json")
    vocab_len = tokenizer.get_vocab_size()

    # TODO: increase batch size in testing
    if args.test:
        batch_size = 1

    ################ Datasets ##############
    data_loader = data_builder(batch_size=batch_size)

    ################ Model Init ##############
    model_class = models_dict[args.model_name]
    model = model_class(time_steps, src_fmri_features, max_size, vocab_len, d_model, d_ff, N, heads, device).to(device)
    model = model.float()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    ################ Model Training/Testing ##############
    name = args.model_name  + '_' + str(src_fmri_features) + '_' + args.type
    out_name = os.path.join (configs.MODELS_TRAIN_PATH, name)

    if args.test:
        model.load_state_dict(torch.load('%s.pt'%(out_name), weights_only=True))
        saving_file = 'results/convers/%s.txt'%(name)
        inference(model, saving_file, tokenizer, vocab_len, data_loader["test"], sos_token_id, eos_token_id, pad_token_id, max_size, device)
    else:
        if args.retrain:
            model = torch.load('%s.pt'%(out_name), map_location=torch.device(device))
        model = train_model(out_name, model, data_loader["train"], batch_size, optim, epochs, lr, N, sos_token_id, eos_token_id, pad_token_id,max_size, tokenizer, device, wandb_log)

