import torch
from transformers import FlaubertModel, FlaubertTokenizer

class LPTS:
    def __init__(self):
        # Load Flaubert tokenizer and model
        self.tokenizer = FlaubertTokenizer.from_pretrained('flaubert/flaubert_base_cased')
        self.model = FlaubertModel.from_pretrained('flaubert/flaubert_base_cased')
        self.model.eval()

        # If you have a GPU, move the model to GPU
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def encode(self, sentence):
        """Tokenizes and gets embeddings for the given sentence."""
        with torch.no_grad():
            inputs = self.tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=512)
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
