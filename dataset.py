import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
import random
import re
import contractions
import nltk
from nltk.corpus import wordnet, stopwords
import torch
from torch.utils.data import Dataset, DataLoader
import xgboost as xgb

STOPWORDS = set(stopwords.words('english'))

# Local Code
from Config.dataset_config import *
from embedder import *


class TextAugmenter:
    def __init__(self, adversation_ratio=0.1, methods=None):
        self.adversation_ratio = adversation_ratio
        self.methods = methods or ['wordnet']
    
    def random_deletion(self, sentence):
        '''
        Randomally delete words from the sentence
        '''
        words = nltk.word_tokenize(sentence)
        return " ".join([word for word in words if random.random() > self.adversation_ratio])
    
    def random_swap(self, sentence):
        '''
        Randomally swap 2 words of the sentence
        '''
        words = nltk.word_tokenize(sentence)
        for _ in range(int(len(words) * self.adversation_ratio)):
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        return " ".join(words)
    
    def get_wordnet_synonyms(self, word, pos=None):
        '''
        Randomally replace a word with a synonim
        '''
        synonyms = set()
        for syn in wordnet.synsets(word, pos=pos):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().replace('_', ' '))
        synonyms.discard(word)
        return list(synonyms)
    
    def get_wordnet_pos(self, treebank_tag):
        '''
        Use POS to choose the word to replace
        '''
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None
    
    def synonym_replacement(self, sentence):
        '''
        Randomally replace a word with a synonim
        '''
        words = nltk.word_tokenize(sentence)
        pos_tags = nltk.pos_tag(words)
        augmented_words = []
        for word, pos in pos_tags:
            if word.lower() in STOPWORDS:
                augmented_words.append(word)
                continue
            wordnet_pos = self.get_wordnet_pos(pos)
            if wordnet_pos and random.random() < self.adversation_ratio:
                synonyms = self.get_wordnet_synonyms(word, pos=wordnet_pos)
                if synonyms:
                    augmented_words.append(random.choice(synonyms))
                else:
                    augmented_words.append(word)
            else:
                augmented_words.append(word)
        return " ".join(augmented_words)

    def augment_comment(self, comment):
        method = random.choice(self.methods)
        if method == 'deletion':
            return self.random_deletion(comment)
        elif method == 'swap':
            return self.random_swap(comment)
        elif method == 'wordnet':
            return self.synonym_replacement(comment)
        else:
            return comment  # Fallback


class TextDataset(Dataset):
    '''
    Creates a dataset object to interact with different ML and DL models.
    Will load the text comments and allow a dataloader that will get them as vectors or embeddings.
    '''
    def __init__(self, data_path, subset, id_column_idx, comment_column_idx, label_column_idx, subset_column_idx,
                 augmented_classes=None, augmentation_ratio=3.0, augmentation_methods = ['wordnet'], adversation_ratio=0.1):
        self.data_path = data_path
        self.subset = subset
        self.id_column_idx = id_column_idx
        self.comment_column_idx = comment_column_idx
        self.label_column_idx = label_column_idx
        self.subset_column_idx = subset_column_idx
        self.augmented_classes = augmented_classes or []
        self.augmentation_ratio = augmentation_ratio        

        # Load data
        self.data = self.__load_and_filter_data()
        
        # Process data (text, remove nan, remove short comments)
        self.data = self.__preprocess_data()
        
        # Augment data
        if self.augmented_classes and self.augmentation_ratio > 0 and adversation_ratio > 0:
            self.augmenter = TextAugmenter(adversation_ratio=adversation_ratio, 
                                        methods=augmentation_methods)
            self.data = self.__augment_data()
        else:
            print(f'[Dataset Status]: No Augmentation was chosen (augmentation/ adversation ratio == 0 or no augmented_classes). Moving on...')

    def __load_and_filter_data(self):
        print(f'[Dataset Status]: Loading the dataset...')
        try:
            # Try reading with utf-8 encoding
            df = pd.read_csv(self.data_path, encoding='utf-8')
        except UnicodeDecodeError:
            # Fallback to another encoding if utf-8 fails
            df = pd.read_csv(self.data_path, encoding='ISO-8859-1')
        subset_data = df[df.iloc[:, self.subset_column_idx] == self.subset]
        return subset_data

    def __preprocess_data(self):
        """
        Apply text preprocessing to the comment column with a progress bar.
        In addition, drop irrelevant comments and nans.
        """
        # Wrap the progress bar around the column iteration
        tqdm.pandas(desc="Preprocessing comments")
        self.data.iloc[:, self.comment_column_idx] = self.data.iloc[:, self.comment_column_idx].progress_apply(self.preprocess_text)
        self.data = self.data.dropna(subset=[self.data.columns[self.comment_column_idx]])
        self.data = self.data[
        self.data[self.data.columns[self.comment_column_idx]].apply(lambda x: len(x.split()) >= 2)
        ]
        return self.data

    @staticmethod
    def preprocess_text(text):
        """
        Perform basic text normalization:
        - Replace URLs with <URL>.
        - Replace user mentions with <USER>.
        - Clean hashtags, retaining the word only.
        - Remove irrelevant characters (e.g., special symbols, emojis).
        - Normalize whitespace.
        """
        replacements = {
            '“': '"', '”': '"', '‘': "'", '’': "'",
            'â\x80\x9c': '"', 'â\x80\x9d': '"', 'â\x80\x99': "'"
        }

        for bad_char, good_char in replacements.items():
            text = text.replace(bad_char, good_char)

        text = contractions.fix(text)  # Expand contractions
        text = " ".join(text.split())  # Normalize whitespace
        text = re.sub(r'http\S+|www\S+|https\S+', '<URL>', text, flags=re.MULTILINE)  # Replace URLs
        text = re.sub(r'@\w+', '<USER>', text)  # Replace user mentions
        text = re.sub(r'#(\w+)', r'\1', text)  # Clean hashtags, retain the word
        text = re.sub(r'[^\w\s.,!?\'"\{\(\[\-\\/:;]', '', text)  # Remove irrelevant characters
        return text

    def __augment_data(self):
        """
        Augment the data with a progress bar to track augmentation progress.
        """
        augmented_data = []
        for idx, row in tqdm(self.data.iterrows(), total=len(self.data), desc="Augmenting data", unit="row"):
            original_id = row.iloc[self.id_column_idx]
            comment = row.iloc[self.comment_column_idx]
            label = row.iloc[self.label_column_idx]
            if label in self.augmented_classes:
                for copy_number in range(1, int(self.augmentation_ratio) + 1):
                    augmented_comment = self.augmenter.augment_comment(comment)
                    augmented_row = row.copy()
                    augmented_row.iloc[self.comment_column_idx] = augmented_comment
                    augmented_row.iloc[self.id_column_idx] = f"{original_id}_Augmented_{copy_number}"
                    augmented_data.append(augmented_row)
        return pd.concat([self.data, pd.DataFrame(augmented_data, columns=self.data.columns)])

    def save_to_csv(self, output_path):
        """
        Save the dataset to a CSV file for inspection.
        """
        self.data.to_csv(output_path, index=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        comment_id = row.iloc[self.id_column_idx]
        comment = row.iloc[self.comment_column_idx]
        label = row.iloc[self.label_column_idx]
        encoded_label = LABELS_ENCODER.get(label)
        return comment_id, comment, encoded_label
   


class EmbeddingDataset(Dataset):
    '''
    Dataset class to handle embedding generation.
    The embeddings are precomputed on init to support non NN models that cannot handle
    a dataloader, so beware of memory usage.
    If a strict NN based model is selected, there is no need for the pre-computation.
    Takes a TextDataset object.
    '''
    def __init__(self, text_dataset, embedder, embedding_method):
        """
        Args:
            text_dataset (TextDataset): The original text dataset object.
            embedder (Embedder): Embedder instance for generating embeddings.
            embedding_method (str): Method for embedding generation (e.g., 'distilbert', 'tf-idf').
        """
        self.text_dataset = text_dataset
        self.embedder = embedder
        self.embedding_method = embedding_method
        self.subset = text_dataset.subset
        self.data_dir = os.path.dirname(text_dataset.data_path)
        # Ensure the data directory exists
        os.makedirs(self.data_dir, exist_ok=True)

        # File path for the precomputed embeddings
        self.cache_file = os.path.join(self.data_dir, f"subset {self.subset}_augmentation={text_dataset.augmentation_ratio}_embeddings_{self.embedding_method}.pkl")

        # Load or generate embeddings
        if os.path.exists(self.cache_file):
            print(f"[EmbeddingDataset]: Loading precomputed embeddings from {self.cache_file}...")
            with open(self.cache_file, "rb") as f:
                data = pickle.load(f)
            self.embeddings = data["embeddings"]
            self.labels = data["labels"]
        else:
            print(f"[EmbeddingDataset]: Precomputing embeddings and saving to {self.cache_file}...")
            self.embeddings, self.labels = self.__precompute_embeddings()
            print("[EmbeddingDataset Status]: Embedding generation complete.")

    def __precompute_embeddings(self):
        """
        Generate embeddings and save them to a pickle file for future use.
        """
        embeddings = []
        labels = []

        for _, (comment_id, comment, label) in tqdm(
            enumerate(self.text_dataset),
            total=len(self.text_dataset),
            desc=f"Generating embeddings for {self.subset} dataset"
        ):
            embedding = self.embedder.embed(comment, method=self.embedding_method)
            # Convert embedding to a PyTorch tensor if it's a numpy array or other type
            if isinstance(embedding, np.ndarray):
                embedding = torch.tensor(embedding, dtype=torch.float32)
            embeddings.append(embedding)
            labels.append(label)

        embeddings = torch.stack(embeddings)  # Stack embeddings into a single tensor
        labels = torch.tensor(labels, dtype=torch.long)

        # Save to pickle
        with open(self.cache_file, "wb") as f:
            pickle.dump({"embeddings": embeddings, "labels": labels}, f)

        return embeddings, labels

    def __len__(self):
        return len(self.text_dataset)  # Length is based on the original TextDataset

    def __getitem__(self, idx):
        """
        Return the precomputed embedding and label for the given index.
        """
        return self.embeddings[idx], self.labels[idx]
        
    
def get_dataloader(dataset, embedder=None, datashape='text', embedding_method="distilbert", batch_size=32, shuffle=True, num_workers=2):
    '''
    Will create the DataLoader object.
    Assumption is that is embedder object is passed, this is to be used for feeding in a model.
    For that purpose, if embedder, this function will return a Dataloader, DMatrix (for xgboost) 
    and a (X, y) tuple for other scikit models.
    Args:
        dataset (TextDataset): The original dataset object.
        embedder (Embedder): The embedder class that generates embeddings (DistilBERT or TF-IDF).
        datashape (str): Choose between 'text' or 'embedding'.
        embedding_method (str): Choose between 'distilbert' or 'tf-idf'.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the dataset.
        num_workers (int): Number of subprocesses for data loading.
    Returns:
        if datashape == 'text':
            DataLoader: A PyTorch DataLoader object for text output.
        elif datashape == 'embedding':
            1. DataLoader: A PyTorch DataLoader object for embedding output.
            2. tuple: An (X, y) tuple for other scikit models.
    '''
    print(f'[Dataloader Status]: Loading the dataset...')
    
    if datashape == 'text':
        if embedding_method:
            print(f'[Warning]: You provided an embedding method despite datashape=text. Returning textual dataloader.')
        # For text-based loading, directly return a DataLoader using the raw dataset
        print(f'[Dataloader Status]: Done.')
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    elif datashape == 'embedding':
        # Ensure you have an embedder instance
        if not embedder:
            raise ValueError("Embedder instance must be provided when datashape='embedding'.")
        
        # Wrap the original TextDataset with the EmbeddingDataset class
        embedding_dataset = EmbeddingDataset(text_dataset=dataset, embedder=embedder, embedding_method=embedding_method)
        
        # Prep the output
        dataloader = DataLoader(embedding_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        
        # Use precomputed embeddings and labels directly from the EmbeddingDataset
        X = embedding_dataset.embeddings.numpy()  # Already precomputed in EmbeddingDataset
        y = embedding_dataset.labels.numpy()      # Already precomputed in EmbeddingDataset

        print(f'[Dataloader Status]: Done.')
        return dataloader, (X, y)
    else:
        raise ValueError(f"Unsupported datashape: {datashape}. Choose 'text' or 'embedding'.")


if __name__ == "__main__":
    # Initialize Dataset
    dataset = TextDataset(
        data_path=DATA_PATH,
        subset=SUBSET,
        id_column_idx=ID_COLUMN_IDX,
        comment_column_idx=COMMENT_COLUMN_IDX,
        label_column_idx=LABEL_COLUMN_IDX,
        subset_column_idx=SUBSET_COLUMN_IDX,
        augmented_classes=AUGMENTED_CLASSES,
        augmentation_ratio=AUGMENTATION_RATIO,
        augmentation_methods=AUGMENTATION_METHODS,
        adversation_ratio = ADVERSATION_RATIO
    )

    # Save dataset to CSV for inspection
    dataset.save_to_csv('augmented_dataset_tmp.csv')
    
    # Create Dataloader
    embedder = Embedder()
    
    # Get the DataLoader with embeddings
    # Note the multiple objects outputted here
    dataloader = get_dataloader(dataset, 
                                embedder=embedder, 
                                datashape=DATALOADER_SHAPE, 
                                embedding_method=EMBEDDING_METHOD, 
                                batch_size=BATCH_SIZE,
                                shuffle=True, 
                                num_workers=2)
    
    