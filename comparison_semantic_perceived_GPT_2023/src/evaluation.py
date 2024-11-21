import glob
import torch
from transformers import FlaubertModel, FlaubertTokenizer
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

from nltk.translate import meteor
from nltk import word_tokenize

from utils.utils_eval import WER, BLEU, METEOR, BERTSCORE
from jiwer import wer

import pandas as pd

# from bert_score import BERTScorer
# BERTSCORE_metric = BERTScorer(model_type='bert-base-uncased')


WER_metric = WER()
BLEU_metric = BLEU()
METEOR_metric = METEOR()
BERTSCORE_metric = BERTSCORE()

def calculate_wer(ref_words, hyp_words):
	# ref_words = reference.split()
	# hyp_words = hypothesis.split()
	# Counting the number of substitutions, deletions, and insertions
	substitutions = sum(1 for ref, hyp in zip(ref_words, hyp_words) if ref != hyp)
	deletions = len(ref_words) - len(hyp_words)
	insertions = len(hyp_words) - len(ref_words)
	# Total number of words in the reference text
	total_words = len(ref_words)
	# Calculating the Word Error Rate (WER)
	wer = (substitutions + deletions + insertions) / total_words
	return wer



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


def jaccard_similarity(sentence_A, sentence_B):
    tokens_A = set(sentence_A)
    tokens_B = set(sentence_B)
    intersection = tokens_A.intersection(tokens_B)
    union = tokens_A.union(tokens_B)
    return len(intersection) / len(union) if union else 0.0


if __name__ == "__main__":


  filenames = sorted (glob.glob ("results/*test_perceived*"))

  f = open("results/final_results.csv", "w")
  f_stories = open("results/final_results_stories.csv", "w")
  f.write ("Model ; BLUE score ; METEOR score ; BertScore, WER\n")
  f_stories.write ("Model ; BLUE score ; METEOR score ; BertScore, WER\n")


  for filename in filenames:

    content = pd.read_csv(filename, sep=";###;", header=0)
    content.sort_values(by=['chunk_number'], inplace=True)

    predictions = content["predicted"].values.tolist()
    target = content["target"].values.tolist()
    
    N = len (predictions)
    temp = '{} ' * N
    print (len (predictions))
    predictions = [temp.format(*ele) for ele in zip(*[iter(predictions)] * N)]
    target = [temp.format(*ele) for ele in zip(*[iter(target)] * N)]

    total_bleu = 0
    total_meteor = 0
    total_bert_score = 0
    total_word_erro_rate = 0

    for pred, targ in zip(predictions, target):
      pred = str (pred).replace('\_', '').replace('\n', ' ').strip()
      targ = targ.strip()
      bleu_score = BLEU_metric.score([targ.split()], [pred.split()])[0]
      meteor_score = METEOR_metric.score([targ.split()], [pred.split()])[0]
      bert_score = BERTSCORE_metric.score([pred], [targ])[0]
      
      word_erro_rate = calculate_wer(targ.split(), pred.split())
      total_bleu += bleu_score
      total_meteor += meteor_score
      total_bert_score += bert_score
      total_word_erro_rate += word_erro_rate
    final_bleu = total_bleu / len(predictions)
    final_meteor = total_meteor / len(predictions)


    f_stories.write ("%s ; %s ; %s ; %s ; %s\n"%(filename.split('.txt')[0], round(total_bleu / len(predictions), 4),
                                                                         round(total_meteor / len(predictions), 4), 
                                                                         round(total_bert_score / len(predictions), 4), 
                                                                         round(total_word_erro_rate / len(predictions), 4), 
                                              ))
    
  f.close()
  f_stories.close()
