import glob
import os
import torch
from transformers import FlaubertModel, FlaubertTokenizer

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

from nltk.translate import meteor
from nltk import word_tokenize

import json

class LPTS:
    def __init__(self):
        # Load Flaubert tokenizer and model
        self.tokenizer = FlaubertTokenizer.from_pretrained('flaubert/flaubert_base_uncased')
        self.model = FlaubertModel.from_pretrained('flaubert/flaubert_base_uncased')
        self.model.eval()

        # If you have a GPU, move the model to GPU
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def encode(self, sentence):
        """Tokenizes and gets embeddings for the given sentence."""
        with torch.no_grad():
            inputs = self.tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=100)
            if torch.cuda.is_available():
                for k, v in inputs.items():
                    inputs[k] = v.cuda()
            outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1)  # Average pooling over sequence dimension

    def distance(self, sentence1, sentence2):
        """Computes 'LPIPS' distance between two sentences."""
        embedding1 = self.encode(sentence1)
        embedding2 = self.encode(sentence2)
        dist = torch.nn.functional.cosine_similarity(embedding1, embedding2)
        return 1 - dist  # Convert similarity to distance


def word_overlap_percentage(sentence_A, sentence_B):
    # Tokenize the sentences
    tokens_A = set(sentence_A.split())
    tokens_B = set(sentence_B.split())

    # Check the overlap
    overlap = tokens_A.intersection(tokens_B)

    # Calculate the percentage
    if len(tokens_A) == 0:
        return 0.0
    else:
        return len(overlap) / len(tokens_A) * 1

def gatherup(list):
  preds = []
  for i in range(len(list)):
    pred_sentences = [' '.join(inner_list) for inner_list in list[i]]
    preds.append(pred_sentences)
  return preds

def jaccard_similarity(sentence_A, sentence_B):
    tokens_A = set(sentence_A)
    tokens_B = set(sentence_B)
    intersection = tokens_A.intersection(tokens_B)
    union = tokens_A.union(tokens_B)
    return len(intersection) / len(union) if union else 0.0

if __name__ == "__main__":

  models_results = sorted (glob.glob (f"results/convers/*"))

  for model_dir_results in models_results:
    if not os.path.isdir(model_dir_results):
      continue

    model_name = model_dir_results.split ('/')[-1]
    filenames = sorted (glob.glob (f"results/convers/{model_name}/*.json"))

    f = open(f"results/convers/final_results_{model_name}.csv", "w")
    f.write ("Model ; BLUE score ; meteor score ; jaccard similarity ; word overlap ;  LPIPS\n")

    for filename in filenames:

      predictions = []
      target = []

      with open(filename, 'r') as d:
        file = json.load(d)
        for line in file:
            sentence_txt = line["Generated"].replace('<unk>', ' ').strip()
            predictions.append(sentence_txt)

            sentence_txt = line["Real"].strip()
            target.append(sentence_txt)

      pred_conversations = [predictions[x:x+5] for x in range(0, len(predictions), 5)]
      true_conversations = [target[x:x+5] for x in range(0, len(target), 5)]

      full_preds = [' '.join(inner_list)for inner_list in pred_conversations]
      full_targs = [' '.join(inner_list) for inner_list in true_conversations]

      total_bleu = 0
      total_meteor = 0
      total_word_overlap = 0
      total_lpips = 0
      total_jaccard = 0
      nlp_lpips = LPTS()
      smoothie = SmoothingFunction().method2

      for pred, targ in zip(full_preds, full_targs):
        bleu_score = sentence_bleu([targ.split()], pred.split(),  smoothing_function=smoothie, weights = (0.25, 0.25, 0.25, 0.25))
        meteor_score = meteor([word_tokenize(targ)], word_tokenize(pred))
        total_bleu += bleu_score
        total_meteor += meteor_score
        jaccard_score = jaccard_similarity(targ.split(), pred.split())
        total_jaccard += jaccard_score
        lpips_dist = nlp_lpips.distance(targ, pred)
        total_lpips += lpips_dist
        overlap_score = word_overlap_percentage(targ, pred)
        total_word_overlap += overlap_score

      final_bleu = total_bleu / len(full_preds)
      final_meteor = total_meteor / len(full_preds)
      final_jaccard = total_jaccard / len(full_preds)
      final_word_overlap = total_word_overlap / len(full_preds)
      final_lpips = total_lpips / len(full_preds)

      f.write ("%s ; %s ; %s ; %s ; %s ; %s\n"%(filename.split('.txt')[0], round (final_bleu * 100, 2),
                                                                          round (final_meteor * 100, 2),
                                                                          round (final_jaccard*100, 2),
                                                                          round (final_word_overlap, 4),
                                                                          round (final_lpips[0].cpu().detach().item(), 4))
              )
    f.close()
