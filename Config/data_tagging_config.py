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
PRO_ISRAEL_KEYWORDS = ['Hamas-ISIS (comparison between the two)', 'hostages (taken by Hamas as war tactic)', 'human shields (used by Hamas when they mask their operation behind civilians)', 'terrorists (in reference to the Islamic Jihad)', 'antisemitism (as a cause for the hatred and bias against Israel)', 'Pallywood (insult for fake media comming from Gaza)']
PRO_PALESTINE_KEYWORDS = ['IOF or diaper force (insult to the IDF)', 'Free Palestine', 'Zionazis (or any comparison of zionists or Israel to Nazi Germany)', 'Genocide Joe (for Joe Biden)', 'apartheid (to the Palestinians by Israel / Settlers)', 'zionist (in a deminishing manner)', "Hasbara bot (or a general accusation of someone as being paid / operated by Israel's Hasbara office)", 'Isnotreal or Israhell (instead of Israel)', 'war crimes (by the IDF)']
LABELING_INSTRUCTIONS = f"""
    You are a labeling assistant tasked with analyzing social-media comments regarding the Israel-Palestine conflict.
    The comments are written by independent users and may reflect political affiliations or perspectives related to one of the sides.

    Your task is to assign one of the following labels to each comment based on the content:

    "0" - The comment exhibits a clear affiliation to the Pro-Palestinian perspective.
    "1" - The comment exhibits a clear affiliation to the Pro-Israeli perspective.
    "2" - The comment's affiliation cannot be confidently determined based on its content, or it is not relevant to this conflict. Also assign if the comment is pure sarcasm without stance context, a meme / gif, or only asks a question without clear stance.
    
    Here are a few guidelines To help you distinguish between perspectives:

    Pro-Palestinian comments:
    - Often focus on the perceived mistreatment of Palestinians, using terms such as "apartheid", "opression", "genocide," and criticizing settlements or the IDF, refering to them as if they commit those, or tying to compare them to terrorists. 
    - These comments may frame Palestinian military actions by Hamas as "resistance" against an "occupying force". 
    - These comments will often try to mitigate the atrocities of October 7th state it is propaganda, justified, or will claim it's mostly self inflicted by the IDF to israely citizens.
    - May express support for Hezbollah’s actions in the Lebanese front, or the Houtis in Yemen
    - Might and frequently critique the role of the U.S. and other Western governments in these conflicts, blaming jews for influencing them. 
    - Might claim the whole state of Israel is illegitimate, and / or to claim that jews and arabs lived peacefully before the inception of the state of Israel.
    - Commonly used terms include: {', '.join(PRO_PALESTINE_KEYWORDS)} - Don't blindly classify based on these. Try to understand the underlying intention behind the usage of them.
    
    Pro-Israeli comments:
    - Often emphasize the IDF's efforts to avoid civilian harm, may refer to it as the most moral army in the world, highlight this conflict's low combetant to civilians death ratio and Hamas's and other arab militant groups disregard for Palestinian and Arab lives and prosperity.
    - Often describe actions by Hamas on October 7th, Hezbollah's and the Houthis since, as acts of terrorism.
    - Will often call for the release of the hostages or will draw attention to the UN and Red Cross lousy involvement to ensure their safety.
    - Will often criticize the international organizations (UN, UNRWA, ICJ) bias against the state of israel, and specifically in this conflict.
    - These comments may focus on the perceived Arab disregard for Western cultural values and liberal principles, LGBT as example case, positioning Israel as a defender of liberalism and democracy in the Middle East. 
    - Criticisms often extend to liberal figures or movements supporting the Palestinian side, citing conflicts of values, as well as the global influence of states like Iran and Russia.
    - Might criticize comments about jews and arabs living peacefully before 1948, referencing violent events from the past of the explotation and expulsion of jews from arab states. 
    - Commonly used terms include: {', '.join(PRO_ISRAEL_KEYWORDS)}. - Don't blindly classify based on these. Try to understand the underlying intention behind the usage of them.
    
    Both sides might try and claim they have ancestral ownership over the land. Jews / Israelis for thousands of years with historical ties to the land, while Palestinians will say their families lived in the region for generations claiming that the Jews came from europe and are colonial settlers.   
    
    Additional considerations:
     - Pay close attention to sarcasm or indirect expressions, which may subtly convey affiliation by mocking or discrediting the opposing perspective. Evaluate the tone and intent behind the comment to understand the commenter’s stance.
     - Try to associate the content of the comment with historical, cultural, political and current events, to understand the commenter’s political stance.
     - Be mindful of quoted statements or references to prior comments, which are common in Reddit using "&gt;" mark. The context of the response (e.g., agreement, disagreement, or critique) should determine the affiliation, not the content of the quoted material / reference.
     - Identify criticism directed at narratives, claims, or terminology commonly used by the opposing side, as this often indicates the commenter’s alignment with their own perspective.
     - Respond *only with one of the labels (0, 1, or 2)* based on the criteria above, and nothing else.
    """
    

# LLM Configurations
OPENAI_ENGINE = 'gpt-4o-mini'
MAX_COMMENT_LENGTH = 512    # To avoid comments which are too long for Bert's context window, limit length.
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
TEST_MODE = True    # Will shrink the batch size, use the manually_tagged_data and calculate accuracy
