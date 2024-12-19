# Locations
FULL_DATA_PATH = 'Data/research_data.csv'
TAGGED_DATA_PATH = 'Data/manually_tagged_data.csv'
BATCH_FILE_PATH = 'Data/batch_file_for_job.jsonl'   # Path to save jsonl file for batch job
OUTPUT_FILE_PATH = 'Data/research_data_tagged.csv'  # Path to save tagged comments
ID_COLUMN_IDX = 0
COMMENT_COLUMN_IDX = 1   # The column where the raw text is in the file to be tagged
LABEL_COLUMN_IDX = 6    # The column where the label for the comment is (on tagged data only)

# Labels
LABELS_DECODER = {
    "0": "Pro-Palestine",
    "1": "Pro-Israel",
    "2": "Undefined"
}

# Prompt / for the Prompt
LABELING_INSTRUCTIONS = """
    You are a labeling assistant tasked with analyzing social-media comments regarding the Israel-Palestine conflict.
    The comments are written by independent users and may reflect political affiliations or perspectives related to one of the sides.

    Your task is to assign one of the following labels to each comment based on the content:

    "0" - The comment exhibits a clear affiliation to the Pro-Palestinian perspective.
    "1" - The comment exhibits a clear affiliation to the Pro-Israeli perspective.
    "2" - The comment's affiliation cannot be confidently determined, or it is not relevant to this conflict.
    To help you distinguish between perspectives:

    Pro-Palestinian comments often focus on the perceived mistreatment of Palestinians, using terms such as "apartheid," "genocide," or criticizing settlements and the IDF. These comments may describe Palestinian military actions as resistance against an "occupying force" and often criticize the role of the U.S. and other Western governments.
    Pro-Israeli comments often emphasize efforts by the IDF to avoid civilian harm, describe Palestinian and Hamas actions on October 7th as terrorism, and highlight perceived disregard for Western cultural values and liberal principles. These comments may criticize liberal figures or movements supporting the Palestinian side.
    Additional considerations:

    Pay attention to sarcasm or indirect expressions that may convey affiliation subtly.
    Avoid making assumptions based on the tone alone; focus on the content of the comment.
    Respond with only one of the labels (0, 1, or 2) based on the criteria above.
    """
    

# LLM Configurations
OPENAI_ENGINE = 'gpt-4o-mini'
TEMPERATURE = 0.0   # Level of randomness / creativity of the comment. Set to 0 to return the same response every time.

TEST_BATCH_SIZE = 10    # Number of comments for a single test of the model
TEST_MODE = True    # Will shrink the batch size, use the manually_tagged_data and calculate accuracy
