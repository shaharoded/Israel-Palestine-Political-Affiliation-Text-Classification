import re
import contractions
from tqdm import tqdm
tqdm.pandas()

def preprocess_text(text):
    """
    Perform basic text normalization:
    - Replace URLs with <URL>.
    - Replace user mentions with <USER>.
    - Clean hashtags, retaining the word only.
    - Remove irrelevant characters (e.g., special symbols, emojis).
    - Normalize whitespace.

    Args:
        text (str): Input text to be normalized.

    Returns:
        str: Normalized text.
    """
    # Replacement mappings for common misencoded characters
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


df["cleaned text"] = df["tweet text"].progress_apply(preprocess_text)