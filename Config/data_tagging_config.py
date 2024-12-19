# Locations
FULL_DATA_PATH = 'Data/research_data.csv'
TAGGED_DATA_PATH = 'Data/manually_tagged_data.csv'
OUTPUT_FILE_PATH = 'Data/research_data_tagged.csv'
COMMENT_ID_COLUMN_IDX = 0
TEXT_COMMENT_COLUMN_IDX = 1   # The column where the raw text is in the file to be tagged

# Labeling Configurations
OPENAI_ENGINE = 'gpt-4o-mini'
TEMPERATURE = 0.0
BATCH_SIZE = 10
TEST_MODE = True    # Will shrink the batch size, use the manually_tagged_data and calculate accuracy


# Prompt / for the Prompt
LABELING_INSTRUCTIONS = """
    You are a labeling assistant. The data are social-media comments regarding the Israel Palestine conflict.
    The comments are from indipendent users which might be politically affiliated / connected to one of the sides.
    Pro Palestinian comments will usually...
    Pro Israeli comments will usually...
    Your task is to label the comment as one of the following:
    1. "0" - The comment displays clear affiliation to the Pro Palestinian side.     
    2. "1" - The comment displays clear affiliation to the Pro Israeli side.
    3. "2" - You can't define the political affiliation with confidence, or not sure that the comment is relevant to this conflict.
    Only respond with one of the labels (0, 1, or 2).
    """