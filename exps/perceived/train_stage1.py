import torch
import argparse
from tokenizers import Tokenizer

import os
import sys

sys.path.insert(0, os.getcwd())

from src.load_perceived_all_subject import data_builder
import src.configs.perceived.configs as configs
from src.transformers_src.Transformer import Transformer, CNNTransformer, DuplexTransformerConv, BipartiteTransformerConv, DeconvBipartiteTransformerConv

models_dict = {
'Transformer':Transformer,
'CNNTransformer':CNNTransformer,
'DuplexTransformerConv':DuplexTransformerConv,
'BipartiteTransformerConv':BipartiteTransformerConv,
'DeconvBipartiteTransformerConv':DeconvBipartiteTransformerConv,
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
            src, trg_sentences = batch["bold_signal"], batch["text_output"]

            #print (src.shape)
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
            output, softmax_output = model(src.float(),input_decoder.float())

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
            torch.save(model.state_dict(), '%s/%s_%d.pt'%(configs.TRAINED_MODELS_PATH, name, epoch))
            torch.save(model.state_dict(), '%s/%s.pt'%(configs.TRAINED_MODELS_PATH, name))


    print("Training completed!")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-m", help="Name of the model to train.", choices = models_dict.keys(), default = "DeconvBipartiteTransformerConv")
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--retrain", action='store_true')
    parser.add_argument("--seed", default = 3)
    parser.add_argument("--batch_size", default = 128, type = int)
    parser.add_argument("--epochs", default = 60, type = int)
    parser.add_argument("--lr", default = 0.0001, type = float)
    parser.add_argument("--saving_path", default = "trained_models/perceived")
    args = parser.parse_args()

    name = args.model_name
    torch.manual_seed(args.seed)

    ################ Parameters ##############
    lr = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    time_steps = configs.time_steps
    max_size = configs.max_size
    d_model = configs.d_model
    heads = configs.heads
    d_ff = configs.d_ff
    N = configs.N
    src_fmri_features = configs.src_fmri_features_max

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pad_token_id, sos_token_id, eos_token_id = 0, 1, 2
    tokenizer = Tokenizer.from_file("./tools/tokenizer-perceived.json")
    vocab_len = tokenizer.get_vocab_size()

    ################ Datasets ##############
    data_loader = data_builder(batch_size)


    ################ Model Init ##############
    model_class = models_dict[args.model_name]
    model = model_class(time_steps, src_fmri_features, max_size, vocab_len, d_model, d_ff, N, heads, device).to(device)
    model = model.float()
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    ################ Model Training/Testing ##############
    checkpoint_filename = '%s/%s.pt'%(args.saving_path, args.model_name)

    if args.retrain:
        model.load_state_dict(torch.load(checkpoint_filename, weights_only=True))
    train_model(name, model, data_loader["train"], optim, epochs, sos_token_id, eos_token_id, pad_token_id,max_size, tokenizer, device)
