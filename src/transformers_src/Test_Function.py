import torch
import torch.nn as nn
#import wandb
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from .Metric import word_overlap_percentage, jaccard_similarity, detokenize, remove_word
from .Inference import generate_sentence_ids
from .LPTS import LPTS



def add_filling_tokens_convert_to_tensor(token_id_list, sos_token_id, eos_token_id, pad_token_id, max_size):

    token_ids_tensors = []

    for id_list in token_id_list:
        id_list = [sos_token_id] + id_list + [eos_token_id]
        token_id = [pad_token_id for _ in range (max_size)]
        for i in range (len (id_list)):
            token_id[i] = id_list[i]

        #id_tensor = torch.tensor(token_id)
        token_ids_tensors.append(token_id)
    return torch.Tensor(token_ids_tensors).type(torch.int64)

def test_model(name, model, test_dataset, batch_size, lr, N, sos_token_id,
              eos_token_id, pad_token_id, max_seq_len, tokenizer, vocab_len, device, wandb_log):

    model.eval()  # Set the model to evaluation mode

    criterion = nn.CrossEntropyLoss()
    nlp_lpips = LPTS()
    #num_samples = len(dataset)
    word_to_remove1 = '[PAD]'
    word_to_remove2 = '[EOS]'
    word_to_remove3 = '[SOS]'
    total_loss = 0.0
    total_bleu = 0
    total_word_overlap_test = 0
    total_lpips_test = 0
    total_jaccard_test = 0
    total_samples = 0
    smoothie = SmoothingFunction().method1
    # Iterate over batches


    for batch in test_dataset:
        # batch_end = min(batch_start + batch_size, train_num_samples)
        # batch = train_dataset[batch_start:batch_end]
        src, trg_sentences = batch["bold_signal"], batch["text_output"]

        trg = []
        for a in trg_sentences:
            trg.append(tokenizer.encode(a, add_special_tokens=False).ids)

        trg = add_filling_tokens_convert_to_tensor(trg, sos_token_id, eos_token_id, pad_token_id, max_seq_len)
        src, trg = src.to(device), trg.to(device)

        sentences, softmax_output, output_ids = generate_sentence_ids(model, src.float(),
                                                                      sos_token_id, eos_token_id,
                                                                      pad_token_id, max_seq_len - 1,
                                                                      vocab_len, device)

        output = output_ids.float()

        # Compute loss
        loss = criterion(output.view(-1, output.size(-1)), trg[:,1:].view(-1))

        total_loss += loss.item()  # Accumulate loss
        total_samples += trg.size(0)  # Accumulate number of samples

        # Add references and hypotheses for BLEU calculation
        for i in range(trg.size(0)):
                # Get the reference sentence
                ref = trg[i].tolist()

                ref_sentence = tokenizer.decode(ref, skip_special_tokens = True)

                hyp = sentences[i].type(torch.int64).tolist()


                hyp_sentence = tokenizer.decode(hyp, skip_special_tokens = True)


                # Calculate BLEU score for this sentence pair
                bleu_score = sentence_bleu([ref_sentence.split()], hyp_sentence.split(), smoothing_function=smoothie)
                print(bleu_score)
                total_bleu += bleu_score

                # Calculate word overlap
                overlap_score = word_overlap_percentage(ref_sentence, hyp_sentence)
                total_word_overlap_test += overlap_score

                # Calculate LPIPS distance (assuming you have the function nlp_lpips.distance)
                lpips_dist = nlp_lpips.distance(ref_sentence, hyp_sentence)
                total_lpips_test += lpips_dist.item()

                # Calculate Jaccard similarity
                jaccard_score = jaccard_similarity(ref_sentence, hyp_sentence)
                total_jaccard_test += jaccard_score

    avg_loss = total_loss / total_samples
    bleu_score = total_bleu / total_samples
    word_overlap = total_word_overlap_test / total_samples
    lpips = total_lpips_test / total_samples
    jaccard = total_jaccard_test / total_samples


    print(f"Loss: {avg_loss:.4f} BLEU: {bleu_score:.4f} Word Overlap: {word_overlap:.4f}% Jaccard Similarity: {jaccard:.4f} Test NLPIPS: {lpips:.4f}")


    return avg_loss, bleu_score
