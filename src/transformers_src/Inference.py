import torch
import torch.nn.functional as F
import torch.nn as nn
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
#import wandb
from .LPTS import LPTS
from .Metric import word_overlap_percentage, jaccard_similarity, detokenize, remove_word



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



def inference(model, saving_file, tokenizer, vocab_len, test_dataset, sos_token_id, eos_token_id, pad_token_id, max_seq_len, device):

    total_loss = 0
    total_bleu = 0
    total_word_overlap_test = 0
    total_lpips_test = 0
    total_jaccard_test = 0
    criterion = nn.CrossEntropyLoss()
    nlp_lpips = LPTS()
    #vocab = {v: k for k, v in vocab.items()}
    total_samples = len(test_dataset)



    for batch in test_dataset:
        # batch_end = min(batch_start + batch_size, train_num_samples)
        # batch = train_dataset[batch_start:batch_end]
        src, trg_sentences = batch["bold_signal"], batch["text_output"]

        trg = []
        for a in trg_sentences:
            trg.append(tokenizer.encode(a, add_special_tokens=False).ids)

        trg = add_filling_tokens_convert_to_tensor(trg, sos_token_id, eos_token_id, pad_token_id, max_seq_len)
        src, trg = src.to(device), trg.to(device)




        output_ids, _, output = generate_sentence_ids(model, src.float(), sos_token_id,
                                                      eos_token_id, pad_token_id,
                                                      max_seq_len - 1, vocab_len, device)
        output = output.float()
        loss = criterion(output.view(-1, output.size(-1)), trg[:,1:].view(-1))
        bleu_score, lpips_dist, overlap_score, jaccard_score = print_output_sentence(output_ids, trg, saving_file,
                                                                                     tokenizer, pad_token_id,
                                                                                     eos_token_id, nlp_lpips)
        total_loss += loss
        total_bleu += bleu_score
        total_lpips_test += lpips_dist.item()
        total_word_overlap_test += overlap_score
        total_jaccard_test += jaccard_score

    Loss = total_loss / total_samples
    bleu_score = total_bleu / total_samples
    word_overlap = total_word_overlap_test / total_samples
    lpips = total_lpips_test / total_samples
    jaccard = total_jaccard_test / total_samples

    print(f"Sequence to sequence: BLEU: {bleu_score:.4f} Word Overlap: {word_overlap:.4f}% Jaccard Similarity: {jaccard:.4f} Test NLPIPS: {lpips:.4f}")

def generate_sentence_ids(model, src, sos_token_id, eos_token_id, pad_token_id, max_length, vocab_len, device):
    model.eval()
    bs = src.size(0)

    # Preallocate memory
    sentences = torch.full((bs, max_length), pad_token_id, dtype=torch.float32, device=device)
    sentences[:, 0] = sos_token_id
    logits = torch.full((bs, max_length, vocab_len), 0, dtype=torch.float32, device=device)
    prob = torch.full((bs, max_length, vocab_len), 0, dtype=torch.float32, device=device)

    # Encoder once
    e_outputs, src_mask = model.encoder(src.float())
    with torch.no_grad():
        for t in range(1, max_length): #max_length
            current_tokens = sentences[:, :t]
            current_tokens_padded = torch.nn.functional.pad(current_tokens, (0, max_length - t), value=pad_token_id)
            d_output = model.decoder(current_tokens_padded.float(), e_outputs, src_mask)
            output = model.out(d_output)

            next_logit = output[:, t, :]
            logits[:, t, :] = next_logit

            softmax_output = F.softmax(output, dim=2)

            next_prob = softmax_output[:, t, :]
            prob[:, t, :] = next_prob

            next_tokens = next_prob.argmax(dim=-1)
            sentences[:, t] = int (next_tokens)

    return sentences.to(device), prob.to(device), logits.to(device)

def print_output_sentence(output, trg_des, saving_file, tokenizer, pad_token_id, eos_token_id, nlp_lpips):
    smoothie = SmoothingFunction().method1
    trg_des = trg_des.flatten().tolist()
    # trg_desc =[x for x in trg_des if x != pad_token_id]
    # trg_desc = [x for x in trg_desc if x != eos_token_id]
    # trg_desc = [x for x in trg_desc if x != 1]
    # trg_desc = [x for x in trg_desc if x != 0]

    output_words = tokenizer.decode(output[0].type(torch.int64).tolist(), skip_special_tokens = True).split (' ')

    unique_list = list(set(output_words))


    desc_words = tokenizer.decode(trg_des, skip_special_tokens = True).split (' ')

    output_sentence = ' '.join(output_words)
    output_sentence1 = ' '.join(unique_list)
    desc_sentence = ' '.join(desc_words)

    bleu_score = sentence_bleu([desc_sentence.split()], output_sentence.split(), smoothing_function=smoothie)
    lpips_dist = nlp_lpips.distance(desc_sentence, output_sentence)
    overlap_score = word_overlap_percentage(desc_sentence, output_sentence)
    jaccard_score = jaccard_similarity(desc_sentence, output_sentence)

    with open(saving_file, 'a') as f:
        f.write(f'The predicted Conversation : {output_sentence1}\nThe target Conversation : {desc_sentence}\n\n')
    return bleu_score, lpips_dist, overlap_score, jaccard_score
