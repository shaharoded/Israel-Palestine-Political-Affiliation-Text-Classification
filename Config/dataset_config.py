import os

# Data & Embeddings
# Get the project root directory (move up one level from the 'Config' folder)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Base directories (relative to project root)
DATA_DIR_PATH = os.path.join(project_root, 'Data')
EMBEDDING_DIR_PATH = os.path.join(project_root, 'Embedding')
DATA_PATH = os.path.join(DATA_DIR_PATH, 'full_research_data_tagged.csv')  # Full path to the data file
EMBEDDING_PATH = os.path.join(EMBEDDING_DIR_PATH, "distilbert-finetuned")  # Full path to the embedding directory
TFIDF_PATH = os.path.join(EMBEDDING_DIR_PATH, 'tfidf/tfidf_vectorizer.pkl')  # Full path to the TF-IDF vectorizer

# Relevant Columns
ID_COLUMN_IDX = 0
COMMENT_COLUMN_IDX = 1   # The column where the raw text is in the file to be tagged.
LABEL_COLUMN_IDX = 7    # The column where the label for the comment is (on tagged data only).
SUBSET_COLUMN_IDX = 8   #   The column where the subset mark (A, B) is.

# Augmentation
AUGMENTED_CLASSES = ['Pro-Israel', 'Pro-Palestine'] # Classes to augment.
AUGMENTATION_RATIO = 0    # Increase in the comments number, int. Meaning -> 1 comments turns to 1 + AUGMENTATION_RATIO comments.
AUGMENTATION_METHODS = ['deletion', 'swap', 'wordnet']    # Add from 'deletion', 'swap', 'wordnet'
ADVERSATION_RATIO = 0.2 # Replacement ratio within the comment.

# Undersampling (maximal number of records per class)
UNDERSAMPLING_TARGETS = {
    "Pro-Palestine": 5500,
    "Pro-Israel": 5500,
    "Undefined": 5500
}

# Embedding
EMBEDDING_METHOD = "distilbert"     # "distilbert" or "tf-idf"

# Dataloader
BATCH_SIZE = 128
DATALOADER_SHAPE = 'embedding'   # Choose from 'text' or 'embedding'
LABELS_ENCODER = {
    "Pro-Palestine": 0,
    "Pro-Israel": 1,
    "Undefined": 2
}