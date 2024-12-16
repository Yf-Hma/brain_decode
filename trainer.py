import argparse
import torch

from src.load_data import data_builder
from src.models import *
import src.configs as configs

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

    os.remove ("%s/%s*"%(saving_path, model_name))
    save_to = "%s/%s_%s.pth"%(saving_path, model_name, ("best" if is_best else str (cur_epoch)))
    print("Saving checkpoint at epoch {} to {}.".format(cur_epoch, save_to))
    torch.save(save_obj, save_to)


def load_from_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=True)
    try:
        model.load_state_dict(checkpoint["model"])
    except RuntimeError as e:
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


def train (model, model_name, type, data_loader, saving_path, epochs = 100, save_epochs = 10, starting_epoch = 1, lr = 0.0001):
	model.train()
	optim = torch.optim.Adam (model.parameters(), lr = lr)
	best_loss = 100000

	for epoch in range(starting_epoch, epochs + 1):
		print ('-------- Epoch: ', epoch)
		mean_loss = 0

		if epoch > (epochs - 20) and epochs > 20:
			save_epochs = 2

		for sample in data_loader:
			loss = model(sample)
			mean_loss += loss
			optim.zero_grad()
			loss.backward()
			optim.step()

		if epoch % save_epochs == 0 and (mean_loss / len (data_loader)) < best_loss:
			best_loss = mean_loss
			print(model_name + "_" + type, epoch, saving_path)
			save_checkpoint(model, model_name, epoch, saving_path)

def test (model, data_loader, model_name):
    model.eval()
    f = open("results/%s.txt"%model_name, "w")
    for sample in data_loader:
        output_text = model.generate (sample)
        for predicted, target in zip (output_text, sample["text_output"]):
            f.write("The predicted Conversation :")
            f.write(predicted + "\n")
            f.write("The target Conversation :")
            f.write(target + "\n")

    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default = 32, type = int)
    parser.add_argument("--seed", default = 3)
    parser.add_argument("--model_name", "-m", help="Name of the model to train.", choices = ["MllmBrainToTextV0", "MllmBrainToTextV1", "MllmBrainToTextV2"], default = "MllmBrainToTextV0")
    parser.add_argument('--test', action='store_true', help = "test the model")
    parser.add_argument('--retrain', action='store_true', help = "retrain from existing checkpoint")
    parser.add_argument("--lr", default = 0.0001, type = float)
    parser.add_argument("--starting_epoch", default = 1, type = int)
    parser.add_argument("--save_epochs", default = 5, type = int)
    parser.add_argument("--epochs", default = 300, type = int)
    parser.add_argument("--saved_checkpoint", "-s", type = str)
    parser.add_argument("--type", "-t", type = str, default = 'spoken', choices = ['spoken', 'perceived'])
    parser.add_argument('--load_in_4bit', action='store_true', help = "to load the llm quantized in 4 bits for inference.")

    args = parser.parse_args()
    args.saving_path = configs.MODELS_TRAIN_DIR

    models_dict = {
    'MllmBrainToTextV0':MllmBrainToTextV0,
    'MllmBrainToTextV1':MllmBrainToText,
    'MllmBrainToTextV2':MllmBrainToTextV2,
    }

    torch.manual_seed(args.seed)

    data_loader = data_builder(args.batch_size)
    llm = models_dict[args.model_name](load_in_4bit = args.load_in_4bit)

    name = args.model_name + "_" + str (configs.src_fmri_features) + '_' + args.type

    if args.test:
        llm = load_from_checkpoint(llm, args.saved_checkpoint)
        test (llm, data_loader["test"], name)
    else:
        if args.retrain:
            llm = load_from_checkpoint(llm, args.saved_checkpoint)

        train (llm,
               name,
               configs.type,
               data_loader["train"],
               configs.MODELS_TRAIN_DIR,
               epochs = args.epochs,
               save_epochs = args.save_epochs,
               starting_epoch = args.starting_epoch,
               lr = args.lr)
