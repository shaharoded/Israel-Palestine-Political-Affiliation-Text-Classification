import pandas as pd
import random
import re
import contractions
from nltk.corpus import wordnet
from torch.utils.data import Dataset, DataLoader

# Local Code
from Config.dataset_config import *


class TextDataset(Dataset):
    '''
    Creates a dataset object to interact with different ML and DL models.
    Will load the text comments and allow a dataloader that will get them as vectors or embeddings.
    '''
    def __init__(self, data_path, subset, id_column_idx, comment_column_idx, label_column_idx, subset_column_idx,
                 augmented_classes=None, augmentation_ratio=1.0, adversation_ratio=0.1):
        self.data_path = data_path
        self.subset = subset
        self.id_column_idx = id_column_idx
        self.comment_column_idx = comment_column_idx
        self.label_column_idx = label_column_idx
        self.subset_column_idx = subset_column_idx
        self.augmented_classes = augmented_classes or []
        self.augmentation_ratio = augmentation_ratio
        self.adversation_ratio = adversation_ratio

        # Load and preprocess data
        self.data = self.__load_and_filter_data()
        self.data = self.__preprocess_data()
        self.data = self.__augment_data()

    def __load_and_filter_data(self):
        df = pd.read_csv(self.data_path)
        subset_data = df[df.iloc[:, self.subset_column_idx] == self.subset]
        return subset_data

    def __preprocess_data(self):
        # Apply text preprocessing to the comment column
        self.data.iloc[:, self.comment_column_idx] = self.data.iloc[:, self.comment_column_idx].apply(self.preprocess_text)
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

    def __augment_sentence(self, sentence):
        words = sentence.split()
        augmented_words = []
        for word in words:
            if random.random() < self.adversation_ratio:
                synonyms = self.__get_synonyms(word)
                if synonyms:
                    augmented_words.append(random.choice(synonyms))
                else:
                    augmented_words.append(word)
            else:
                augmented_words.append(word)
        return " ".join(augmented_words)

    def __get_synonyms(self, word):
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().replace('_', ' '))
        # Remove the original word from synonyms
        synonyms.discard(word)
        return list(synonyms)

    def __augment_data(self):
        augmented_data = []
        for _, row in self.data.iterrows():
            comment = row.iloc[self.comment_column_idx]
            label = row.iloc[self.label_column_idx]
            if label in self.augmented_classes:
                for _ in range(int(self.augmentation_ratio)):
                    augmented_comment = self.__augment_sentence(comment)
                    augmented_row = row.copy()
                    augmented_row.iloc[self.comment_column_idx] = augmented_comment
                    augmented_data.append(augmented_row)
        return pd.concat([self.data, pd.DataFrame(augmented_data, columns=self.data.columns)])

    def save_to_csv(self, output_path):
        """
        Save the dataset to a CSV file for inspection.
        """
        self.data.to_csv(output_path, index=False)
    
    def get_dataloader():
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        comment_id = row.iloc[self.id_column_idx]
        comment = row.iloc[self.comment_column_idx]
        label = row.iloc[self.label_column_idx]
        return comment_id, comment, label
    
def get_dataloader(dataset, datashape='text', embedding_file_path=None, batch_size=32, shuffle=True, num_workers=2):
    '''
    Will create the DataLoader object as needed for next steps.

    Args:
        dataset (TextDataset): Gets the original dataset object.
        datashape (str): Choose between 'text' or 'embedding'. 'text' is meant for
                        loading into an embedding model for fine-tune, while embedding is meant for 
                        loading to classifier models.
        embedding_file_path (str): Path to the embedding weights or TF-IDF file which will
                                    be loaded to the Embedding class for text process.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the dataset.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        DataLoader: A PyTorch DataLoader object for the specified dataset.
    '''
    if datashape == 'text':
        if embedding_file_path:
            print(f'[Warning]: A vectorization file path was inserted despite datashape=text. Returning textual dataloader.')
        # For text-based loading, directly return a DataLoader using the raw dataset
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    elif datashape == 'embedding':
        # Placeholder for embedding logic
        # If embedding_file_path is None, raise an error
        if embedding_file_path is None:
            raise ValueError("embedding_file_path must be provided when datashape='embedding'.")
        
        # Load embeddings from the file and preprocess them (to be implemented later)
        # embeddings = load_embeddings(embedding_file_path)
        # vectorized_dataset = create_vectorized_dataset(dataset, embeddings)
        # return DataLoader(vectorized_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        raise NotImplementedError("Embedding-based DataLoader is not implemented yet.")
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
        adversation_ratio=ADVERSATION_RATIO
    )

    # Save dataset to CSV for inspection
    dataset.save_to_csv('augmented_dataset_tmp.csv')
    
    # Create Dataloader
    dataloader = get_dataloader(dataset, 
                                datashape=DATALOADER_SHAPE, 
                                embedding_file_path=EMBEDDING_PATH, 
                                batch_size=BATCH_SIZE, 
                                shuffle=True, 
                                num_workers=2)
    
    