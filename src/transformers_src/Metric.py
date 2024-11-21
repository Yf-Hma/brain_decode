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
        return len(overlap) / len(tokens_A) * 100

def jaccard_similarity(sentence_A, sentence_B):
    tokens_A = set(sentence_A.split())
    tokens_B = set(sentence_B.split())
    intersection = tokens_A.intersection(tokens_B)
    union = tokens_A.union(tokens_B)
    return len(intersection) / len(union) if union else 0.0

def remove_word(sentence, word_to_remove):
    words = sentence.split()
    words = [word for word in words if word != word_to_remove]
    return ' '.join(words)

def detokenize(token_ids, vocab_dict):
    word_list = []
    for tid in token_ids:
        word = vocab_dict[int(tid)]
        word_list.append(word)
    sentence = ' '.join(word_list)
    return sentence
