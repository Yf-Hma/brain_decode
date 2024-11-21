import torch
from transformers import FlaubertModel, FlaubertTokenizer
from .Metric import remove_word
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
