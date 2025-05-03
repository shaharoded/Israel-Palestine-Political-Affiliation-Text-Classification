import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import random
import re
import contractions
import nltk
from nltk.corpus import wordnet, stopwords
import torch
from torch.utils.data import Dataset, DataLoader

STOPWORDS = set(stopwords.words('english'))

# Local Code
from Config.dataset_config import *
from embedder import *


# ----------------------------------------------------------------------
# 0.  Tiny CSV cache (load once, reuse for every split)
# ----------------------------------------------------------------------

class _CSVCache:
    '''
    Use to avoid re-loading a CSV file multiple times
    '''
    df = None                 # class attribute

def _load_csv(path, encoding='utf-8'):
    if _CSVCache.df is None:             # load once, reuse for all splits
        try:
            _CSVCache.df = pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            _CSVCache.df = pd.read_csv(path, encoding='ISO‑8859‑1')
    return _CSVCache.df



# ----------------------------------------------------------------------
#  Text Augmentation methods, applied in TextDataset
# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------

class TextDataset(Dataset):
    """
    Hold ALL rows (TRAIN / VAL / TEST) internally.
    Augmentation / undersampling are applied **only** to the TRAIN rows.
    Call get_subset('TRAIN'|'VAL'|'TEST') to obtain a view that behaves
    like a normal torch Dataset.
    """
    def __init__(self, csv_path, id_column_idx, comment_column_idx, label_column_idx, split_column_idx,
                 augmented_classes=[], augmentation_ratio=3, augmentation_methods = ['wordnet'], adversation_ratio=0.1, undersampling_targets={}):
        """
        Initiates the dataset, which is the base Dataset for the embeddings.
        
        Args:
            csv_path (str): The path to the .csv data file.
            id_column_idx (int): Idx for the ID column in the dataframe.
            comment_column_idx (int): Idx for the text column in the dataframe.
            label_column_idx (int): Idx for the label column in the dataframe.
            split_column_idx (int): Idx for the subset column in the dataframe, indicating how to split it.
            augmented_classes (list): A list of the classes to augment. Chose from ['Pro-Israel', 'Pro-Palestine', 'Undefined']
            augmentation_ratio (int): Increase in the comments number. Meaning -> 1 comments turns to 1 + AUGMENTATION_RATIO comments.
            augmentation_methods (list): Choose from ['deletion', 'swap', 'wordnet'].
            adversation_ratio (float): Replacement ratio within the comment.
            undersampling_targets (dict): A mapping object of how much to undersample each class.
        """
        self.csv_path = csv_path
        self.id_column_idx = id_column_idx
        self.comment_column_idx = comment_column_idx
        self.label_column_idx = label_column_idx
        self.split_column_idx = split_column_idx
        self.action = 'regular'

        # ------------ load --------------------------------------------------
        df = _load_csv(csv_path)
        
        # ------------ basic text cleaning ------------------------------------------
        df = self.__preprocess(df)

        # ------------ TRAIN‑only operations ----------------------------------------
        mask_train = df.iloc[:, split_column_idx] == "TRAIN"

        if undersampling_targets:
            df.loc[mask_train] = self._undersample(
                df.loc[mask_train], undersampling_targets)
            self.action = 'undersampled'

        if (augmented_classes and augmentation_ratio > 0
                and adversation_ratio > 0):
            df = self._augment(df, mask_train,
                             augmented_classes, augmentation_ratio,
                             augmentation_methods, adversation_ratio)
            self.action = 'augmented'

        df.reset_index(drop=True, inplace=True)  # keep indices clean
        self.data = df

        # pre‑compute row indices per split for fast lookup
        self.idx_split = {
            s: np.flatnonzero(df.iloc[:, split_column_idx] == s)
            for s in ("TRAIN", "VAL", "TEST")
        }

        print(f"[TextDataset] rows: "
              f"train={len(self.idx_split['TRAIN'])}, "
              f"val={len(self.idx_split['VAL'])}, "
              f"test={len(self.idx_split['TEST'])}")

    # ------------------------------------------------------------------ helpers ---

    @staticmethod
    def _normalize(text: str) -> str:
        """
        Perform basic text normalization:
        - Replace quotes with a placeholder (to mark them)
        - Replace URLs with <URL>.
        - Replace user mentions with <USER>.
        - Clean hashtags, retaining the word only.
        - Remove irrelevant characters (e.g., special symbols, emojis).
        - Normalize whitespace.
        """
        replacements = {
            '“': '"', '”': '"', '‘': "'", '’': "'",
            'â\x80\x9c': '"', 'â\x80\x9d': '"', 'â\x80\x99': "'", '&#x200B;': ' ',
            '&amp;': '&', '&lt;': '<', '&gt;': '>'
        }

        for bad_char, good_char in replacements.items():
            text = text.replace(bad_char, good_char)

        text = re.sub(r'(^|\n)\s*(?:>|\&gt;).*?(?=\n|$)', '<QUOTE>', text) # Any line that starts with > or &gt; is a quoted parent text
        text = contractions.fix(text)  # Expand contractions
        text = " ".join(text.split())  # Normalize whitespace
        text = re.sub(r'http\S+|www\S+|https\S+', '<URL>', text, flags=re.MULTILINE)  # Replace URLs
        text = re.sub(r'@\w+', '<USER>', text)  # Replace user mentions
        text = re.sub(r'#(\w+)', r'\1', text)  # Clean hashtags, retain the word
        text = re.sub(r'[^\w\s.,!?\'"\{\(\[\-\\/:;]', '', text)  # Remove irrelevant characters
        return text
    
    def __preprocess(self, df):
        """
        Apply text preprocessing to the comment column with a progress bar.
        In addition, drop irrelevant comments and nans.
        """
        # Wrap the progress bar around the column iteration
        tqdm.pandas(desc="Cleaning Comments")
        df.iloc[:, self.comment_column_idx] = df.iloc[:, self.comment_column_idx].progress_apply(self._normalize)
        df = df.dropna(subset=[df.columns[self.comment_column_idx]])
        df = df[
        df[df.columns[self.comment_column_idx]].apply(lambda x: len(x.split()) >= 2)
        ]
        return df

    def _undersample(self, df, targets: dict):
        """
        Performs undersampling so that all the labels will have a predefined number of rows in df (later to be self.data).
        Args:
            df (pd.DataFrame): The df for processing.
            targets (Dict(str:int)): A dictionary that defined the max number of rows for each label in the output like:
                {
                    "Pro-Palestine": 5500,
                    "Pro-Israel": 5500,
                    "Undefined": 5500
                }
        """
        dfs = []
        for lab, n in targets.items():
            lab_df = df[df.iloc[:, self.label_column_idx] == lab]
            dfs.append(lab_df.sample(min(n, len(lab_df)), random_state=42))
        return pd.concat(dfs, ignore_index=True)
    
    def _augment(self, df, mask_train, classes, ratio, methods, adv_ratio):
        """
        Performs augmentation using the TextAugmenter class to the classes that needs augmentation.
        Augmentation will create |ratio| adversed copies of comments from the augmented class, giving them unique UIDs.
        Args:
            df (pd.DataFrame): The df for processing.
            mask_train (pd.DataFrame): boolean mask selecting TRAIN rows inside *df*.
            classes (List(str)): The classes to augment (class labels like "Pro-Israel").
            ratio (int): Number of new adversed copies to add.
            methods (List(str)): A list with the augmentation methods.
            adv_ratio (float): Ratio of each comment (words from total comment) to adverse. 
        
        NOTE: All of the params are handled in the llm_config.py file.
        """
        aug = TextAugmenter(adv_ratio, methods)
        train_df = df.loc[mask_train]
        extra = []

        for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Augment", unit="row"):
            lab = row.iloc[self.label_column_idx]
            if lab not in classes:
                continue
            for i in range(int(ratio)):
                new = row.copy()
                new.iloc[self.comment_column_idx] = aug.augment_comment(row.iloc[self.comment_column_idx])
                new.iloc[self.id_column_idx]  = f"{row.iloc[self.id_column_idx]}_aug{i+1}"
                extra.append(new)
        if extra:
            df = pd.concat([df, pd.DataFrame(extra, columns=df.columns)], ignore_index=True)
        return df

    def save_to_csv(self, output_repo: str = "Data"):
        """
        Save the dataset to a CSV file for inspection.
        """
        action_tag = getattr(self, "action", "regular")     # augmented / undersampled / regular
        filename  = f"{action_tag}_research_data_for_inspection.csv"
        out_path = Path(output_repo) / filename
        self.data.to_csv(out_path, index=False)
        print(f"[TextDataset] saved CSV → {out_path}")

    # -------------------------- Dataset API -------------------
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        comment_id = row.iloc[self.id_column_idx]
        comment = row.iloc[self.comment_column_idx]
        encoded_label = LABELS_ENCODER.get(row.iloc[self.label_column_idx])
        return comment_id, comment, encoded_label
   


class EmbeddingDataset(Dataset):
    '''
    Dataset class to handle embedding generation.
    For modularity, as this object is the needed object for classification task,
    it is responsible for the creation of the TextDatasets and their modification to Embedding
    based one.
    Caching of dataset memory is defined to avoid re-calculations of embeddings.
    The embeddings are precomputed on init to support non NN models that cannot handle
    a dataloader, so beware of memory usage.
    If a strict NN based model is selected as best model, there is no need for the pre-computation.

    NOTE: One dataset will be created from a .csv file and will have TRAIN, VAL and TEST callable subsets (using _View).
    '''
    class _View(Dataset):
        def __init__(self, emb, labels, idx):
            self.embeddings, self.labels = emb[idx], labels[idx]
        def __len__(self):        return len(self.labels)
        def __getitem__(self, i): return self.embeddings[i], self.labels[i]

    def __init__(self, text_dataset, embedder, embedding_method, cache_dir=r"Data\cache"):
        """
        Creates the Dataset instance which fits the classification task.
        Most recurrent parameters are used to initiate the TextDataset in the init.
        Args:
            text_dataset (TextDataset): A pre-computed TextDataset instance
            embedder (Embedder): Embedder instance for generating embeddings.
            embedding_method (str): Method for embedding generation (e.g., 'distilbert', 'tf-idf').
            cache_dir (str): Directory to save the pre-computed embeddings instead of re-calculate.
        """
        # ---- load CSV once -------------------------------------------------
        self.text_dataset = text_dataset
        self.embedder, self.embedding_method = embedder, embedding_method
        cache_dir = Path(cache_dir); cache_dir.mkdir(exist_ok=True)
        self.action = self.text_dataset.action
        self.cache_file = cache_dir / f"{embedding_method}_embeddings_{self.action}.pkl"

        if self.cache_file.exists():
            print(f"[EmbeddingDataset]: Loading precomputed embeddings from {self.cache_file}...")
            with open(self.cache_file, "rb") as f:
                blob = pickle.load(f)
            self.embeddings = blob["embeddings"]          # (N,D) torch tensor
            self.labels = blob["labels"]           # (N,)  torch tensor
        else:
            print(f"[EmbeddingDataset]: Precomputing embeddings and saving to {self.cache_file}...")
            self.embeddings, self.labels = self._build_and_cache()
        print("[EmbeddingDataset Status]: Embedding generation complete.")

    
    def _build_and_cache(self):
        emb, lab = [], []
        for txt in tqdm(self.text_dataset.data.iloc[:, self.text_dataset.comment_column_idx],
                        total=len(self.text_dataset.data), desc="Embedding Comments"):
            v = self.embedder.embed(txt, method=self.embedding_method)
            emb.append(torch.as_tensor(v, dtype=torch.float32))
        emb = torch.stack(emb)
        lab = torch.tensor(
            self.text_dataset.data.iloc[:, self.text_dataset.label_column_idx]
                .map(LABELS_ENCODER).to_numpy(),
            dtype=torch.long)
        with open(self.cache_file, "wb") as f:
            pickle.dump({"embeddings": emb, "labels": lab}, f)
        return emb, lab
    # --------------------- Public Methods ---------------------------------
    def get_subset(self, split: str) -> Dataset:
        """
        Get a desired split fromt the Dataset -> TRAIN, VAL or TEST.
        """
        idx = self.text_dataset.idx_split[split]
        return self._View(self.embeddings, self.labels, idx)
    
    # --------------------- Dataset API ---------------------------------
    def __len__(self):
        return len(self.text_dataset)  # Length is based on the original TextDataset

    def __getitem__(self, idx):
        """
        Return the precomputed embedding and label for the given index.
        """
        return self.embeddings[idx], self.labels[idx]
        
def _to_numpy(t):
    """
    Helper:  tensor  ➜  numpy (always on CPU, detached).
    Accepts torch.Tensor or anything that is already a numpy array.
    """
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)
   
def get_dataloader(dataset, batch_size=32, shuffle=True, num_workers=2):
    '''
    Will create the DataLoader object.
    Assumption is that a Dataset object is passed. Function will response if the dataset is TextDataset or EmbeddingDataset.
    If EmbeddingDataset, this function will return a Dataloader and a (X, y) tuple for other scikit models.
    Else, if dataset is TextDataset it will return a text dataset for it, which is designed for analysis of Transformer model's feed.
    The function is GPU‑safe:   `.cpu()` before `.numpy()`.

    Args:
        dataset (TextDataset or EmbeddingDataset): The original dataset object.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the dataset.
        num_workers (int): Number of subprocesses for data loading.
    
    Returns:
        if type(dataset) == 'TextDataset':
            DataLoader: A PyTorch DataLoader object for text output.
        elif type(dataset) == 'EmbeddingDataset':
            1. DataLoader: A PyTorch DataLoader object for embedding output.
            2. tuple: An (X, y) tuple for other scikit models.
    '''
    print(f'[Dataloader Status]: Preparing the dataloader...')
    pin = torch.cuda.is_available()          # use pinned memory when GPU present
    dl = DataLoader(dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                    pin_memory=pin)
    
    # --- TextDataset: nothing else to do ---------------------------------
    if 'TextDataset' in dataset.__class__.__name__:
        return dl

    # -------- EmbeddingDataset or its _View -------------------------------------
    if isinstance(dataset, EmbeddingDataset) or isinstance(dataset, EmbeddingDataset._View):
        # quick peek at first batch
        for b, (_, y_b) in enumerate(dl):
            print(f"[DL] peek batch {b}: y[:5] =", _to_numpy(y_b)[:5])
            break

        X = _to_numpy(dataset.embeddings)
        y = _to_numpy(dataset.labels)
        print(f"[DL] EmbeddingDataset ready. X shape {X.shape}, y len {len(y)}")
        return dl, (X, y)

    # -------- unknown dataset ---------------------------------------------------
    raise ValueError(f"[DL] Unrecognized dataset type: {type(dataset)}")
    
    
if __name__ == "__main__":
    # Initialize Dataset
    text_dataset = TextDataset(
        csv_path=DATA_PATH,
        id_column_idx=ID_COLUMN_IDX,
        comment_column_idx=COMMENT_COLUMN_IDX,
        label_column_idx=LABEL_COLUMN_IDX,
        subset_column_idx=SUBSET_COLUMN_IDX,
        augmented_classes=AUGMENTED_CLASSES,
        augmentation_ratio=AUGMENTATION_RATIO,
        augmentation_methods=AUGMENTATION_METHODS,
        adversation_ratio = ADVERSATION_RATIO,
        undersampling_targets=UNDERSAMPLING_TARGETS
    )

    
    # Save dataset to CSV for inspection
    text_dataset.save_to_csv()
    
    # Create Embedding Dataset
    embedding_dataset = EmbeddingDataset(text_dataset=text_dataset,
                                        embedder=Embedder(), 
                                        embedding_method=EMBEDDING_METHOD)
    
    train_ds = embedding_dataset.get_subset('TRAIN')   # returns EmbeddingDataset._View
    
    # Get the DataLoader with embeddings
    # Note the multiple objects outputted here
    dataloader = get_dataloader(embedding_dataset,  
                                batch_size=BATCH_SIZE,
                                shuffle=False, 
                                num_workers=2)
    
    