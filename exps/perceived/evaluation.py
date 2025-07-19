import os, glob
from utils.utils_eval import WER, BLEU, METEOR, BERTSCORE
import pandas as pd
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
main = os.path.dirname(parent)
sys.path.append(main)

import src.configs.perceived.configs as configs

WER_metric = WER()
BLEU_metric = BLEU(n=1)
METEOR_metric = METEOR()
BERTSCORE_metric = BERTSCORE(rescale = False, score = "recall")

def calculate_wer(ref_words, hyp_words):
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

  filenames = sorted (glob.glob ("results/perceived/%s/*test_perceived_speech*"%sys.argv[1]))

  f_stories = open("results/perceived/final_results_%s.csv"%sys.argv[1], "w")
  f_stories.write ("Model ; BLUE score ; METEOR score ; BertScore, WER\n")

  CHUNKLEN = configs.CHUNKLEN
  WINDOW = configs.WINDOW

  for filename in filenames:

    content = pd.read_csv(filename, sep=";###;", header=0, engine='python')
    content.sort_values(by=['chunk_number'], inplace=True)

    predictions = content["predicted"].values.tolist()
    reals = content["target"].values.tolist()
    

    # print (predictions[0])
    

    N = WINDOW // CHUNKLEN
    #N = 1
    # temp = '{} ' * N
    # segmented_sentences_pred = [temp.format(*ele) for ele in zip(*[iter(predictions)] * N)]
    # segmented_sentences_real = [temp.format(*ele) for ele in zip(*[iter(reals)] * N)]

    segmented_sentences_pred = []
    segmented_sentences_real = []
    for i in range (0, len (predictions), N):
        segment_pred = ""
        segment_real = ""
        for j in range (i, min (i + N, len (predictions))):
            segment_pred += predictions[j] + " "
            segment_real += reals[j] + " "
            
        segmented_sentences_pred.append (segment_pred)
        segmented_sentences_real.append (segment_real)

    #print (segmented_sentences_pred[0])

    #exit ()
    # temp = '{} ' * N
    # print (len (predictions))
    # predictions = [temp.format(*ele) for ele in zip(*[iter(predictions)] * N)]
    # targets = [temp.format(*ele) for ele in zip(*[iter(targets)] * N)]


    # print (len (segmented_list))
    # exit ()


    total_bleu = 0
    total_meteor = 0
    total_bert_score = 0
    total_word_erro_rate = 0

    for pred, targ in zip(segmented_sentences_pred, segmented_sentences_real):
      pred = str (pred).replace('\_', '').replace('\n', ' ').strip()
      targ = targ.strip()


      bleu_score = BLEU_metric.score([targ], [pred])[0]
      meteor_score = METEOR_metric.score([targ], [pred])[0]

    #   bleu_score = BLEU_metric.score([targ.split()], [pred.split()])[0]
    #   meteor_score = METEOR_metric.score([targ.split()], [pred.split()])[0]


      bert_score = BERTSCORE_metric.score([pred], [targ])[0]
      
      word_erro_rate = calculate_wer(targ.split(), pred.split())
      total_bleu += bleu_score
      total_meteor += meteor_score
      total_bert_score += bert_score
      total_word_erro_rate += word_erro_rate


    f_stories.write ("%s ; %s ; %s ; %s ; %s\n"%(filename.split('.txt')[0], round(total_bleu / len(segmented_sentences_pred), 4),
                                                                         round(total_meteor / len(segmented_sentences_pred), 4), 
                                                                         round(total_bert_score / len(segmented_sentences_pred), 4), 
                                                                         round(total_word_erro_rate / len(segmented_sentences_pred), 4), 
                                              ))

  f_stories.close()
