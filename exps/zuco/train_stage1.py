import argparse
import torch
import os, sys
import os
import sys
import torch
import argparse
from transformers import set_seed
from tokenizers import Tokenizer

sys.path.insert(0, os.getcwd())

from exps.zuco.load_zuco_data import get_loaders
import src.configs.zuco.configs as configs

from src.transformers_src.Transformer import DeconvBipartiteTransformerConv

models_dict = {
'DeconvBipartiteTransformerConv':DeconvBipartiteTransformerConv
}

def add_filling_tokens_convert_to_tensor(token_id_list, sos_token_id, eos_token_id, pad_token_id, max_size):
    token_ids_tensors = []
    for id_list in token_id_list:
        id_list = [sos_token_id] + id_list + [eos_token_id]
        token_id = [pad_token_id for _ in range (max_size)]
        for i in range (len (id_list)):
            if i < len (token_id):
                token_id[i] = id_list[i]
        if len (token_id) < len (id_list):
            token_id[-1] = eos_token_id
        token_ids_tensors.append(token_id)
    return torch.Tensor(token_ids_tensors).type(torch.int64)


def train_model(name, model, train_dataset, optimizer, num_epochs, sos_token_id, eos_token_id, pad_token_id, max_seq_len, tokenizer, device):

    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()

        total_loss_train = 0.0
        total_samples_training = 0

        # Iterate over batches
        for batch in train_dataset:
            src, trg_sentences = batch["signal"], batch["text_output"]

            trg = []
            for a in trg_sentences:
                trg.append(tokenizer.encode(a, add_special_tokens=False).ids)

            input_decoder = [a[:-1] for a in trg]
            input_decoder = add_filling_tokens_convert_to_tensor(input_decoder, sos_token_id, eos_token_id, pad_token_id, max_seq_len - 1)
            label_decoder = add_filling_tokens_convert_to_tensor(trg, sos_token_id, eos_token_id, pad_token_id, max_seq_len)
            label_decoder = label_decoder[:,1:]

            src, input_decoder = src.to(device), input_decoder.to(device)
            label_decoder = label_decoder.to(device)
            optimizer.zero_grad()  # Clear gradients

            # Forward pass
            output, _ = model(src.float(),input_decoder.float())

            # Compute loss
            loss = criterion(output.reshape(-1, output.size(-1)), label_decoder.reshape(-1))

            # Backward pass
            loss.backward()

            # Update model parameters
            optimizer.step()

            total_loss_train += loss.item() * label_decoder.size(0)  # Accumulate loss
            total_samples_training += label_decoder.size(0)  # Accumulate number of samples

        epoch_loss = total_loss_train / total_samples_training

        print(f"Epoch {epoch + 1}/{num_epochs} Loss: {epoch_loss:.4f}")

        if (epoch + 1) % 20 == 0 or (epoch + 1) == num_epochs:
            torch.save(model.state_dict(), '%s/%s_%d.pt'%(configs.MODELS_TRAIN_PATH, name, epoch + 1))
            torch.save(model.state_dict(), '%s/%s.pt'%(configs.MODELS_TRAIN_PATH, name))

    print("Training completed!")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-m", help="Name of the model to train.", choices = models_dict.keys(), default = "DeconvBipartiteTransformerConv")
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--version", default=2, type=int, choices=[1, 2])
    parser.add_argument("--retrain", action='store_true')
    parser.add_argument("--seed", default = 42)
    parser.add_argument("--batch_size", default = 128, type = int)
    parser.add_argument("--epochs", default = 20, type = int)
    parser.add_argument("--lr", default = 0.0001, type = float)
    parser.add_argument("--saving_path", default = "trained_models/zuco")
    args = parser.parse_args()

    set_seed(args.seed)
    ################ Parameters ##############
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    time_steps = configs.time_steps
    max_size = configs.max_size
    d_model = configs.d_model
    heads = configs.heads
    d_ff = configs.d_ff
    N = configs.N
    src_eeg_features = configs.src_eeg_features
    pad_token_id, sos_token_id, eos_token_id = 0, 1, 2
    tokenizer = Tokenizer.from_file("./tools/tokenizer-zuco.json")
    vocab_len = tokenizer.get_vocab_size()
    args.saving_path = configs.MODELS_TRAIN_PATH
    processed_data_path = configs.PROCESSED_DATA_PATH

    ################ Loading data ##############
    train_set, _ = get_loaders (processed_data_path, args.version, args.batch_size,args.batch_size)

    ################ Model Init ##############
    model_class = models_dict[args.model_name]
    model = model_class(time_steps, src_eeg_features, max_size, vocab_len, d_model, d_ff, N, heads, device).to(device)
    model = model.float()
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    ################ Model Training ##############
    if args.retrain:
        checkpoint_filename = '%s/%s.pt'%(args.saving_path, args.model_name)
        assert os.path.exists(checkpoint_filename), "Path to the checkpoint does not exists."
        model.load_state_dict(torch.load(checkpoint_filename, weights_only=True))
    train_model(args.model_name + "_v" + str(args.version), model, train_set, optim, epochs, sos_token_id, eos_token_id, pad_token_id,max_size, tokenizer, device)
