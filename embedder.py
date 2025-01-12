from transformers import DistilBertTokenizer, DistilBertModel
import joblib
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

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

        Parameters:
        - input_text (str): The text to be embedded.
        - method (str): The embedding method ("distilbert" or "tf-idf").

        Returns:
        - embedding (np.ndarray or torch.Tensor): The embedding for the input text.
        """
        if method == "distilbert":
            return self._distilbert_embedding(input_text)
        elif method == "tf-idf":
            return self._tfidf_embedding(input_text)
        else:
            raise ValueError("Unsupported embedding method. Use 'distilbert' or 'tf-idf'.")


    def _distilbert_embedding(self, input_text):
        """
        Generate embeddings using the DistilBERT model.

        Parameters:
        - input_text (str): The text to be embedded.

        Returns:
        - embedding (torch.Tensor): The [CLS] vector -> Embedding for the input text.
        """
        # Tokenize and encode the input text
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        # Pass the input through the DistilBERT model
        outputs = self.distilbert_model(**inputs)
        
        # Extract the [CLS] token embedding (first token)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # First token's embedding
        return cls_embedding.detach().numpy().flatten()


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
