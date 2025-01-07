# Data & Embeddings
DATA_PATH = 'Data/full_research_data_tagged.csv'
EMBEDDING_PATH = 'Embedding/BERT_Weights...'
TFIDF_PATH = 'Embedding/TFIDF_File.pkl'
SUBSET = 'B'    # Choose between A or B, one for embedding finetune and the other for classification.

# Relevant Columns
ID_COLUMN_IDX = 0
COMMENT_COLUMN_IDX = 1   # The column where the raw text is in the file to be tagged.
LABEL_COLUMN_IDX = 7    # The column where the label for the comment is (on tagged data only).
SUBSET_COLUMN_IDX = 8   #   The column where the subset mark (A, B) is.

# Augmentation
AUGMENTED_CLASSES = ['Pro-Israel', 'Pro-Palestine'] # Classes to augment.
AUGMENTATION_RATIO = 5    # Increase in the comments number, int.
ADVERSATION_RATIO = 0.1 # Replacement ratio within the comment.

# Dataloader
BATCH_SIZE = 32
DATALOADER_SHAPE = 'text'   # Choose from text or embedding