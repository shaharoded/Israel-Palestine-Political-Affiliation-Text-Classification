import numpy as np
import torch
from torch.nn.functional import normalize
from transformers import DistilBertTokenizer, DistilBertModel
import joblib
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

# Local Code
from Config.dataset_config import *

class Embedder:
    def __init__(self):
        """
        Initializes the Embedder object with a DistilBERT model and a TF-IDF vectorizer.,
        both loaded from pretrained files.
        """
        # Load the DistilBERT tokenizer and model
        self.tokenizer = DistilBertTokenizer.from_pretrained(EMBEDDING_PATH)
        self.distilbert_model = DistilBertModel.from_pretrained(EMBEDDING_PATH)
        
        # Load the TF-IDF vectorizer
        self.tfidf_vectorizer = joblib.load(TFIDF_PATH)


    def embed(self, input_text, method="distilbert"):
        """
        Generate embeddings for the input text using the specified method.
        Generate embeddings by sentence chunking + mean pooling per sentence.
        Longer comments benefit from dividing into smaller semantic units.

        Parameters:
        - input_text (str): The text to be embedded.
        - method (str): The embedding method ("distilbert" or "tf-idf").

        Returns:
        - embedding (np.ndarray): The embedding for the full input text.
        """
        sentences = sent_tokenize(input_text)
        sentence_embeddings = []

        for sent in sentences:
            if method == "distilbert":
                sent_emb = self._distilbert_embedding(sent)
            elif method == "tf-idf":
                sent_emb = self._tfidf_embedding(sent)
            else:
                raise ValueError("Unsupported embedding method. Use 'distilbert' or 'tf-idf'.")
            sentence_embeddings.append(sent_emb)
        
        # Mean-pool sentence-level embeddings
        if len(sentence_embeddings) == 0:
            return np.zeros_like(self._distilbert_embedding(""))  # fallback for empty input
        
        stacked = np.stack(sentence_embeddings)  # shape: (n_sentences, emb_dim)
        mean_embedding = stacked.mean(axis=0)
        return mean_embedding


    def _distilbert_embedding(self, input_text):
        """
        Generate embeddings using the DistilBERT model.

        Parameters:
        - input_text (str): The text to be embedded.

        Returns:
        - embedding (torch.Tensor): A mean pooling vector from the model over the entire input -> Embedding for the input text (comment).
        """
        # Tokenize and encode the input text
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        # Pass the input through the DistilBERT model
        outputs = self.distilbert_model(**inputs)
        last_hidden = outputs.last_hidden_state  # shape: (1, seq_len, hidden_size)
        attention_mask = inputs["attention_mask"]
        
        # Mean pooling over all non-padding tokens
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = torch.sum(last_hidden * mask_expanded, dim=1)
        sum_mask = mask_expanded.sum(dim=1)

        mean_pooled = sum_embeddings / sum_mask
        return mean_pooled.detach().numpy().flatten()


    def _tfidf_embedding(self, input_text):
        """
        Generate embeddings using the TF-IDF model.

        Parameters:
        - input_text (str): The text to be embedded.

        Returns:
        - embedding (np.ndarray): The embedding for the input text.
        """
        # Transform the input text using the TF-IDF vectorizer
        embedding = self.tfidf_vectorizer.transform([input_text])
        return embedding.toarray().flatten()


if __name__ == "__main__":
    embedder = Embedder()
    print("Embedding using DistilBERT:")
    print(embedder.embed("israel palestine conflict", method="distilbert"))
    print("Embedding using TF-IDF:")
    print(embedder.embed("israel palestine conflict", method="tf-idf"))
