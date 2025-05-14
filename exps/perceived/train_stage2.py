import argparse
import torch
from torch.optim import Adam

import os
from glob import glob
import sys


current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
main = os.path.dirname(parent)
sys.path.append(main)


from src.load_perceived_data import data_builder, data_builder_from_file
import src.configs.perceived.configs as configs
from src.models.models_semantic import BrainDEC_V0

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

    save_to = "%s/%s_%s.pth"%(configs.TRAINED_MODELS_PATH, model_name, str (cur_epoch))
    print("Saving checkpoint at epoch {} to {}.".format(cur_epoch, save_to))
    torch.save(save_obj, save_to)


def load_from_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    try:
        model.load_state_dict(checkpoint["model"])
    except RuntimeError as e:
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


def train (model, model_name, data_loader, saving_path, epochs = 10, save_epochs = 5, starting_epoch = 1):

	model.train()
	optim = Adam(model.parameters(), lr=0.001)

	lr_scheduler = torch.optim.lr_scheduler.OneCycleLR (optim, max_lr=0.001, epochs=280, steps_per_epoch=len(data_loader))

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

		print (mean_loss / len (data_loader))
		if epoch % save_epochs == 0 and mean_loss < best_loss:
			best_loss = mean_loss
			save_checkpoint(model, model_name, saving_path, epoch)
			test (model, model_name, epoch)




@torch.no_grad()
def test (model, model_name, epoch):
    model.eval()

    test_files = glob("%s/%s/*test_*"%(configs.PROCESSED_DATA_PATH, args.subject))

    for test_file in test_files:
        finename = os.path.basename(test_file).split('.')[0]
        finename_out = "results/perceived/%s/%s_%s.txt"%(args.subject, model_name + "_" + str(epoch), finename)
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
    parser.add_argument("--batch_size", default = 64, type = int)
    parser.add_argument("-seed", default = 42)
    parser.add_argument("--model_name", "-m", help="Name of the model to train.", choices = ["V0"], default = "BrainDEC_V0")
    parser.add_argument('--test', action='store_true', help = "test the model")
    parser.add_argument('--retrain', action='store_true', help = "retrain from existing checkpoint")
    parser.add_argument("--starting_epoch", default = 1, type = int)
    parser.add_argument("--save_epochs", default = 1, type = int)
    parser.add_argument("--epochs", default = 10, type = int)
    parser.add_argument("--subject", '-s', choices=['S1', 'S2', 'S3'])
    parser.add_argument("--saving_path", default = "trained_models")


    parser.add_argument("--saved_checkpoint", "-c", type = str)

    args = parser.parse_args()

    if not os.path.exists ("results/perceived"):
        os.mkdir ("results/perceived")

    if not os.path.exists ("results/perceived/%s"%args.subject):
        os.mkdir ("results/perceived/%s"%args.subject)


    if args.subject == "S1":
        src_fmri_features = 81126
    elif args.subject == "S2":
        src_fmri_features = 94251
    elif args.subject == "S3":
        src_fmri_features = 95556


    torch.manual_seed(args.seed)

    data_loader = data_builder(args.subject, args.batch_size)

    model_base_filename =  "DeconvBipartiteTransformerConv_" + args.subject + '.pt'
    fmri_encoder_path = os.path.join (configs.TRAINED_MODELS_PATH, model_base_filename)

    if not os.path.exists(fmri_encoder_path):
        print ("fMRI encoder path does not exists!", fmri_encoder_path)
        print ()
        exit ()

    llm = BrainDEC_V0(fmri_encoder_path, src_fmri_features)

    args.model_name = args.model_name + "_" + args.subject + "_" + configs.LLM_name

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
