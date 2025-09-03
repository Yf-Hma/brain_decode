import os
import argparse
import torch
import sys
import json

from tokenizers import Tokenizer
from transformers import set_seed

sys.path.insert(0, os.getcwd())

from src.loaders.dataloader_convers import get_loaders as convers_loader
from src.model import BrainDEC_V0, BrainDEC_V1, BrainDEC_V2
import configs.configs_convers as configs
from src.transformers_src.Transformer import *


def save_checkpoint(model, model_name, cur_epoch, saving_path, is_best=False):

    param_grad_dic = {
        k: v.requires_grad for (k, v) in model.named_parameters()
    }
    state_dict = model.state_dict()
    for k in list(state_dict.keys()):
        if k in param_grad_dic.keys() and not param_grad_dic[k]:
            # delete parameters that do not require gradient
            del state_dict[k]

    save_obj = {"model": state_dict,"epoch": cur_epoch}

    #os.system ("rm %s/%s*"%(saving_path, model_name))
    save_to = "%s/%s_%s.pth"%(saving_path, model_name, ("best" if is_best else str (cur_epoch)))
    print("Saving checkpoint at epoch {} to {}.".format(cur_epoch, save_to))
    torch.save(save_obj, save_to)


def load_from_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    try:
        model.load_state_dict(checkpoint["model"])
    except RuntimeError as e:
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


def train (model, model_name, type, data_loader, data_loader_test, saving_path, epochs, save_epochs, starting_epoch = 1, lr = 0.0001):

	model.train()
	optim = torch.optim.Adam (model.parameters(), lr = lr)

	for epoch in range(starting_epoch, epochs + 1):
		print ('-------- Epoch: ', epoch)
		mean_loss = 0

		for sample in data_loader:
			loss = model(sample)
			mean_loss += loss
			optim.zero_grad()
			loss.backward()
			optim.step()

		if epoch % save_epochs == 0:
			print(model_name + "_" + type, epoch, saving_path)
			save_checkpoint(model, model_name, epoch, saving_path)
			test (model, data_loader_test, f"{model_name}_{str(epoch)}")


def test (model, data_loader, model_name):
    model.eval()

    if not os.path.exists (f"results/convers/{configs.LLM_name}"):
         os.mkdir(f"results/convers/{configs.LLM_name}")

    results = []
    for sample in data_loader:
         output_text = model.generate (sample)
         for predicted, target in zip (output_text, sample["text_output"]):
              results.append({"Generated": predicted, "Real": target})

    with open(f"results/convers/{configs.LLM_name}/{model_name}.json", 'w') as out_file:
        json.dump(results, out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default = 32, type = int)
    parser.add_argument("--seed", default = 1, type=int)
    parser.add_argument("--model_name", "-m", help="Name of the model to train.", choices = ["BrainDEC_V0", "BrainDEC_V1", "BrainDEC_V2"], default = "BrainDEC_V0")
    parser.add_argument('--test', action='store_true', help = "test the model")
    parser.add_argument('--retrain', action='store_true', help = "To retrain the model from an existing checkpoint")
    parser.add_argument("--lr", default = 0.0001, type = float)
    parser.add_argument("--starting_epoch", default = 1, type = int)
    parser.add_argument("--save_epochs", default = 50, type = int)
    parser.add_argument("--device", default = "cuda", type = str)
    parser.add_argument("--epochs", default = 200, type = int)
    parser.add_argument("--saved_checkpoint", "-s", type = str)
    parser.add_argument('--use_lora', action='store_true', help = "To use LoRA on the decoder LLM.")
    parser.add_argument('--load_in_4bit', action='store_true', help = "To load the llm quantized in 4 bits for inference.")

    args = parser.parse_args()
    args.saving_path = configs.MODELS_TRAIN_PATH

    models_dict = {
    'BrainDEC_V0':BrainDEC_V0,
    'BrainDEC_V1':BrainDEC_V1,
    'BrainDEC_V2':BrainDEC_V2,
    }

    set_seed(args.seed)
    assert os.path.exists(configs.LLM_PATH), "LLM_PATH does not exist."

    ################# Load Datasets #######################
    train_set_convers, test_set_convers = convers_loader(batch_size = args.batch_size, 
                                                         val_batch_size = args.batch_size, 
                                                         version = args.model_name.split("_")[-1])

    ################# Init fMRI Encoder #######################
    encoder_path = os.path.join (configs.MODELS_TRAIN_PATH, "DeconvBipartiteTransformerConv_%d.pt"%(configs.src_fmri_features))
    
    assert os.path.exists(encoder_path), "Encoder path does not exist."

    tokenizer = Tokenizer.from_file("./tools/tokenizer-convers.json")
    vocab_len = tokenizer.get_vocab_size()
    encoder_model = DeconvBipartiteTransformerConv(configs.time_steps, 
                                                   configs.src_fmri_features, 
                                                   configs.max_size,\
                                                   vocab_len, 
                                                   configs.d_model, 
                                                   configs.d_ff, 
                                                   configs.N, 
                                                   configs.heads, 
                                                   args.device).float()
    
    encoder_model.load_state_dict(torch.load(encoder_path, weights_only=True))
    encoder_model = encoder_model.encoder
    
    ################# Init BrainDEC Model #######################
    mllm_model = models_dict[args.model_name](encoder_model, 
                                              configs, 
                                              configs.src_fmri_features, 
                                              load_in_4bit = args.load_in_4bit, 
                                              lora = args.use_lora, 
                                              device=args.device)

    ################# Checkpoint Filename #######################
    if args.use_lora:
        name = f"{args.model_name}_{str(configs.src_fmri_features)}_{configs.LLM_name}_lora"
    else:
        name = f"{args.model_name}_{str(configs.src_fmri_features)}_{configs.LLM_name}"

    ################# Model Training / Testing #######################
    if args.test:
        name = args.saved_checkpoint.split('/')[-1].split('.')[0]
        mllm_model = load_from_checkpoint(mllm_model, args.saved_checkpoint)
        test (mllm_model, test_set_convers, name)
    else:
        if args.retrain:
            mllm_model = load_from_checkpoint(mllm_model, args.saved_checkpoint)

        train (mllm_model,
            name,
            configs.type,
            train_set_convers,
            test_set_convers,
            configs.MODELS_TRAIN_PATH,
            epochs = args.epochs,
            save_epochs = args.save_epochs,
            starting_epoch = args.starting_epoch,
            lr = args.lr)
