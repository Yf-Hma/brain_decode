import glob, json
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge
#from bert_score import score
import numpy as np
import nltk
import os, sys

nltk.download('punkt')
from nltk.tokenize import word_tokenize

from evaluate import load

wer_metric = load("wer")

files =  glob.glob("results/zuco/*.json")
files.sort()

if os.path.exists ("results/zuco/eval_results.csv"):
    os.remove("results/zuco/eval_results.csv")

f_eval = open("results/zuco/eval_results.csv", "w")
f_eval.write ("Model , BLUE-1 , BLUE-2 , BLUE-3, BLUE-4 , Rouge1-p , Rouge1-r , Rouge1-f , WER \n")


for json_path in files:
    print (f'EVALs of file: {json_path}')
    model_name = json_path.split('/')[-1].split('.')[0]

    # === Load JSON Data ===
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    f_eval.write(model_name)

    # === Prepare data ===
    target_string_list = [entry['text_target'] for entry in data]

    print (max([len (a.split()) for a in target_string_list]))
    pred_string_list = [entry['text_predicted'] for entry in data]

    # Tokenize for BLEU
    target_tokens_list = [[word_tokenize(t)] for t in target_string_list]  # wrapped in an extra list for corpus_bleu
    pred_tokens_list = [word_tokenize(p) for p in pred_string_list]

    # === Calculate BLEU scores ===
    weights_list = [(1.0,),(0.5,0.5),(1./3.,1./3.,1./3.),(0.25,0.25,0.25,0.25)]

    for weight in weights_list:
        corpus_bleu_score = corpus_bleu(target_tokens_list, pred_tokens_list, weights=weight)
        #print(f'corpus BLEU-{len(weight)} score:', corpus_bleu_score)
        f_eval.write (f" , {np.round (corpus_bleu_score, 6)}")
    # print()

    # === Calculate ROUGE scores ===
    rouge = Rouge()
    rouge_scores = rouge.get_scores(pred_string_list, target_string_list, avg=True, ignore_empty=True)
    #print("ROUGE scores:", rouge_scores["rouge-1"])

    for a in ["p", "r", "f"]:
        f_eval.write (f" , {np.round (rouge_scores['rouge-1'][a], 5)}")
                
    """ calculate WER score """
    #wer = WordErrorRate()
    wer_scores = wer_metric.compute(predictions=pred_string_list, references=target_string_list)

    f_eval.write (f" , {np.round (wer_scores, 5)}")

    f_eval.write ("\n")
    
f_eval.close()
