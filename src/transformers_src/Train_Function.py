#import wandb
import torch
import torch.nn as nn
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from torch.distributions import MultivariateNormal
from .Metric import word_overlap_percentage, jaccard_similarity, detokenize, remove_word, LPTS
from .Inference import generate_sentence_ids
# from .LPTS import LPTS
import os


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

        #id_tensor = torch.tensor(token_id)
        token_ids_tensors.append(token_id)
    return torch.Tensor(token_ids_tensors).type(torch.int64)


def train_model(out_path, model, train_dataset, batch_size, optimizer, num_epochs, lr, N, sos_token_id, eos_token_id, pad_token_id, max_seq_len, tokenizer, device, wandb_log):

    criterion = nn.CrossEntropyLoss()
    nlp_lpips = LPTS()
    #train_num_samples = len(train_dataset)
    smoothie = SmoothingFunction().method1
    word_to_remove1 = '[PAD]'
    word_to_remove2 = '[EOS]'
    word_to_remove3 = '[SOS]'
    for epoch in range(num_epochs):
        model.train()

        total_loss_train = 0.0
        total_bleu_train = 0
        total_word_overlap_train = 0
        total_lpips_train = 0
        total_jaccard_train = 0
        total_samples_training = 0

        total_loss_val = 0.0
        total_bleu_val = 0
        total_word_overlap_val = 0
        total_lpips_val = 0
        total_jaccard_val = 0
        total_samples_val = 0

        # Iterate over batches
        for batch in train_dataset:
            src, trg_sentences = batch["bold_signal"], batch["text_output"]

            trg = []
            for a in trg_sentences:
                trg.append(tokenizer.encode(a, add_special_tokens=False).ids)

            #trg = add_filling_tokens_convert_to_tensor(trg, sos_token_id, eos_token_id, pad_token_id, max_seq_len)

            input_decoder = [a[:-1] for a in trg]
            input_decoder = add_filling_tokens_convert_to_tensor(input_decoder, sos_token_id, eos_token_id, pad_token_id, max_seq_len - 1)
            label_decoder = add_filling_tokens_convert_to_tensor(trg, sos_token_id, eos_token_id, pad_token_id, max_seq_len)
            label_decoder = label_decoder[:,1:]

            src, input_decoder = src.to(device), input_decoder.to(device)
            label_decoder = label_decoder.to(device)
            optimizer.zero_grad()  # Clear gradients

            # Forward pass
            output, softmax_output = model(src.float(),input_decoder.float())

            # print (output.shape, label_decoder.shape)
            # Compute loss
            loss = criterion(output.reshape(-1, output.size(-1)), label_decoder.reshape(-1))

            # Backward pass
            loss.backward()

            # Update model parameters
            optimizer.step()

            total_loss_train += loss.item() * label_decoder.size(0)  # Accumulate loss
            total_samples_training += label_decoder.size(0)  # Accumulate number of samples

            # Add references and hypotheses for BLEU calculation

            for i in range(label_decoder.size(0)):
                # Get the reference sentence
                ref = label_decoder[i].tolist()
                ref_sentence = tokenizer.decode(ref, skip_special_tokens = True)

                # Get the hypothesis
                hyp = softmax_output.argmax(dim=-1)[i].tolist()
                hyp_sentence = tokenizer.decode(hyp, skip_special_tokens = True)

                # Calculate BLEU score for this sentence pair
                bleu_score = sentence_bleu([ref_sentence.split()], hyp_sentence.split(), smoothing_function=smoothie)
                total_bleu_train += bleu_score

                # Calculate word overlap
                overlap_score = word_overlap_percentage(ref_sentence, hyp_sentence)
                total_word_overlap_train += overlap_score

                # Calculate LPIPS distance (assuming you have the function nlp_lpips.distance)
                lpips_dist = nlp_lpips.distance(ref_sentence, hyp_sentence)
                total_lpips_train += lpips_dist.item()

                # Calculate Jaccard similarity
                jaccard_score = jaccard_similarity(ref_sentence, hyp_sentence)
                total_jaccard_train += jaccard_score

        epoch_loss = total_loss_train / total_samples_training
        epoch_bleu = total_bleu_train / total_samples_training
        epoch_word_overlap = total_word_overlap_train / total_samples_training
        epoch_lpips = total_lpips_train / total_samples_training
        epoch_jaccard = total_jaccard_train / total_samples_training

        print(f"Epoch {epoch + 1}/{num_epochs} Loss: {epoch_loss:.4f} BLEU: {epoch_bleu:.4f} Word Overlap: {epoch_word_overlap:.4f}% Jaccard Similarity: {epoch_jaccard:.4f} Train LPIPS: {epoch_lpips:.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save(model, '%s_%d.pt'%(out_path, epoch))
            torch.save(model, '%s.pt'%(out_path))
            if epoch > 10:
                os.system ("rm %s_%d.pt"%(out_path, epoch - 10))

    print("Training completed!")
    return model
