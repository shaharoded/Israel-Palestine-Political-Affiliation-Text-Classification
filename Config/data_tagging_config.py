# Locations
FULL_DATA_PATH = 'Data/research_data_untagged.csv'
TAGGED_DATA_PATH = 'Data/manually_tagged_data.csv'
BATCH_FILE_PATH = 'Data/Batches/mini_batch_{}.jsonl'   # Path to save jsonl file for batch job
OUTPUT_FILE_PATH = 'Data/research_data_autotagged.csv'  # Path to save tagged comments
ID_COLUMN_IDX = 0
COMMENT_COLUMN_IDX = 1   # The column where the raw text is in the file to be tagged
LABEL_COLUMN_IDX = 7    # The column where the label for the comment is (on tagged data only)

# Labels
LABELS_DECODER = {
    "0": "Pro-Palestine",
    "1": "Pro-Israel",
    "2": "Undefined"
}

# Prompt / for the Prompt
PRO_ISRAEL_KEYWORDS = ['Hamas-ISIS (comparison between the two)', 'hostages', 'human shields', 'terrorist', 'antisemitism', 'Pallywood (insult for fake media comming from Gaza)']
PRO_PALESTINE_KEYWORDS = ['IOF or diaper force (insult to the IDF)', 'Free Palestine', 'Genocide Joe (for Joe Biden)', 'zionist', 'isnotreal (instead of Israel)', 'war crime']
LABELING_INSTRUCTIONS = f"""
    You are a labeling assistant tasked with analyzing social-media comments regarding the Israel-Palestine conflict.
    The comments are written by independent users and may reflect political affiliations or perspectives related to one of the sides.

    Your task is to assign one of the following labels to each comment based on the content:

    "0" - The comment exhibits a clear affiliation to the Pro-Palestinian perspective.
    "1" - The comment exhibits a clear affiliation to the Pro-Israeli perspective.
    "2" - The comment's affiliation cannot be confidently determined, or it is not relevant to this conflict.
    To help you distinguish between perspectives:

    Pro-Palestinian comments often focus on the perceived mistreatment of Palestinians, using terms such as "apartheid", "opression", "genocide," and criticizing settlements or the IDF, refering to them as if they commit those, or tying to compare them to terrorists. These comments may frame Palestinian military actions by Hamas as resistance against an "occupying force", mitigating the atrocities of October 7th claming it's propaganda, draw parallels to Hezbollah’s actions in the Lebanese front, and frequently critique the role of the U.S. and other Western governments in these conflicts, blaming jews for influencing them. Commonly used terms include: {', '.join(PRO_PALESTINE_KEYWORDS)}.
    Pro-Israeli comments often emphasize the IDF's efforts to avoid civilian harm, refer to it as the most moral army, highlight Hamas's and similar groups disregard for Palestinian and Arab lives and prosperity, and describe actions by Hamas on October 7th, Hezbollah's and the Houthis since, as acts of terrorism or call for the release of the hostages. These comments may focus on the perceived Arab disregard for Western cultural values and liberal principles, LGBT as example case, positioning Israel as a defender of liberalism and democracy in the Middle East. Criticisms often extend to liberal figures or movements supporting the Palestinian side, citing conflicts of values, as well as the global influence of states like Iran and Russia. Commonly used terms include: {', '.join(PRO_ISRAEL_KEYWORDS)}. 
    Both sides might try and claim they have ancestral ownership over the land. Jews / Israelis for thousands of years, while Palestinians will claim Jews came from europe.   
    
    Additional considerations:
     - Pay close attention to sarcasm or indirect expressions, which may subtly convey affiliation by mocking or discrediting the opposing perspective. Evaluate the tone and intent behind the comment to understand the commenter’s stance.
     - Be mindful of quoted statements or references to prior comments. The context of the response (e.g., agreement, disagreement, or critique) should determine the affiliation, not the content of the quoted material.
     - Identify criticism directed at narratives, claims, or terminology commonly used by the opposing side, as this often indicates the commenter’s alignment with their own perspective.
     - Respond with only one of the labels (0, 1, or 2) based on the criteria above.
    """
    

# LLM Configurations
OPENAI_ENGINE = 'gpt-4o-mini'
MAX_COMMENT_LENGTH = 500    # To avoid comments which are too long, limit length.
BATCH_TOKENS_LIMIT = 1000000   # Based on OpenAI limitations which are 2,000,000. Account for calculation method and 1 additional comment after batch is too long.
BATCH_REQUESTS_LIMIT = 50000    # Limit to the number of requests in a single batch
TEMPERATURE = 0.0   # Level of randomness / creativity of the comment. Set to 0 to return the same response every time.
BATCH_ROW_FORMAT = {
    "custom_id": "{comment_id}",
    "method": "POST",
    "url": "/v1/chat/completions",
    "body": {
        "model": OPENAI_ENGINE,
        "messages": [
            {"role": "system", "content": LABELING_INSTRUCTIONS},
            {"role": "user", "content": "Comment: {comment}"}
        ],
        "temperature": TEMPERATURE
    }
}

TEST_BATCH_SIZE = 10    # Number of comments for a single test of the model
TEST_MODE = False    # Will shrink the batch size, use the manually_tagged_data and calculate accuracy
