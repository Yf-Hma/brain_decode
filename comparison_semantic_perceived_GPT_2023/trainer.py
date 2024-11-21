import argparse
import torch
from torch.optim import Adam

import os
from glob import glob

from src.load_data import data_builder, data_builder_from_file
from src.models_semantic import *



def save_checkpoint(model, model_name, saving_path, cur_epoch, is_best=False):

    param_grad_dic = {
        k: v.requires_grad for (k, v) in model.named_parameters()
    }
    state_dict = model.state_dict()
    for k in list(state_dict.keys()):
        if k in param_grad_dic.keys() and not param_grad_dic[k]:
            # delete parameters that do not require gradient
            del state_dict[k]

    save_obj = {"model": state_dict,"epoch": cur_epoch}

    os.system ("rm %s/%s*"%(saving_path,model_name))
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


def train (model, model_name, data_loader, saving_path, epochs = 100, save_epochs = 10, starting_epoch = 1):

	model.train()
	optim = Adam(model.parameters(), lr=0.0001)

	#lr_scheduler = torch.optim.lr_scheduler.OneCycleLR (optim, max_lr=0.001, epochs=epochs, steps_per_epoch=len(data_loader))

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

		print (mean_loss / len (data_loader))
		if epoch % save_epochs == 0 and mean_loss < best_loss:
			best_loss = mean_loss
			save_checkpoint(model, model_name, saving_path, epoch)

		#lr_scheduler.step()

@torch.no_grad()
def test (model, model_name):
    model.eval()
    #f = open("results/%s.txt"%args.model_name, "w")

    test_files = glob("data/%s/*test_*"%args.subject)
    print (test_files)

    for test_file in test_files:
        finename = os.path.basename(test_file).split('.')[0]
        finename_out = "results/%s_%s.txt"%(args.model_name, finename)
        f = open(finename_out, "w")

        data_loader = data_builder_from_file (test_file)
        f.write('chunk_number'+ ';###;' + 'predicted' + ';###;' + 'target' + '\n')
        for sample in data_loader:
            output_text = model.generate (sample)
            for chunk_number, predicted, target in zip (sample["chunk_number"], output_text, sample["text_output"]):
                f.write(str(chunk_number.item()) + ';###;' + predicted.replace('\n', ' ') + ';###;' + target + "\n")
        f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default = 32, type = int)
    parser.add_argument("-seed", default = 3)
    parser.add_argument("--model_name", "-m", help="Name of the model to train.", choices = ["MllmBrainToTextV0"], default = "MllmBrainToTextV0")
    parser.add_argument('--test', action='store_true', help = "test the model")
    parser.add_argument('--retrain', action='store_true', help = "retrain from existing checkpoint")
    parser.add_argument("--starting_epoch", default = 1, type = int)
    parser.add_argument("--save_epochs", default = 10, type = int)
    parser.add_argument("--epochs", default = 300, type = int)
    parser.add_argument("--subject", '-s', choices=['S1', 'S2', 'S3'])
    parser.add_argument("--saving_path", default = "trained_models")


    parser.add_argument("--saved_checkpoint", "-c", type = str)

    args = parser.parse_args()

    if args.subject == "S1":
        src_fmri_features = 81126
    elif args.subject == "S2":
        src_fmri_features = 94251
    elif args.subject == "S3":
        src_fmri_features = 95556

    models_dict = {
    'MllmBrainToTextV0':MllmBrainToTextV0
    }

    torch.manual_seed(args.seed)

    data_loader = data_builder(args.subject, args.batch_size)

    model_base_filename =  "Transformer_" + args.subject + '.pt'
    fmri_encoder_path = os.path.join (args.saving_path, model_base_filename)

    if not os.path.exists(fmri_encoder_path):
        print ("fMRI encoder model does not exists!")
        print (fmri_encoder_path)
        exit ()

    llm = models_dict[args.model_name](fmri_encoder_path, src_fmri_features)

    args.model_name = args.model_name + "_" + args.subject

    if args.test:
        llm = load_from_checkpoint(llm, args.saved_checkpoint)
        test (llm, args.model_name)
    else:
        if args.retrain:
            llm = load_from_checkpoint(llm, args.saved_checkpoint)

        train (llm,
               args.model_name,
               data_loader["train"],
               args.saving_path,
               epochs = args.epochs,
               save_epochs = args.save_epochs,
               starting_epoch = args.starting_epoch)
