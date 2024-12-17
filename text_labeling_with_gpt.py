import openai
import json
import time
import requests

# Set your OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"

def prepare_batch_file(comments, instructions, batch_file_path):
    """
    Prepare a JSONL file for OpenAI Batch API.

    Args:
        comments (list): List of strings to label.
        instructions (str): Instructions for GPT to label the data.
        batch_file_path (str): Path to save the JSONL file.
    """
    with open(batch_file_path, 'w') as f:
        for idx, comment in enumerate(comments):
            data = {
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4",  # Replace with gpt-3.5-turbo if needed
                    "messages": [
                        {"role": "system", "content": instructions},
                        {"role": "user", "content": f"Comment: {comment}"}
                    ],
                    "temperature": 0.0
                }
            }
            f.write(json.dumps(data) + "\n")
    print(f"Batch file prepared at: {batch_file_path}")

def upload_batch_file(batch_file_path):
    """
    Upload the JSONL file to OpenAI.

    Args:
        batch_file_path (str): Path to the JSONL file.

    Returns:
        str: File ID of the uploaded file.
    """
    with open(batch_file_path, "rb") as f:
        response = openai.File.create(
            file=f,
            purpose="batch"
        )
    print(f"File uploaded. File ID: {response['id']}")
    return response['id']

def create_batch_job(file_id, output_file_name):
    """
    Create a batch job to process the uploaded file.

    Args:
        file_id (str): The File ID of the uploaded JSONL file.
        output_file_name (str): Desired name for the output file.

    Returns:
        str: Batch Job ID.
    """
    response = openai.Batch.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        output_file_name=output_file_name
    )
    print(f"Batch job created. Job ID: {response['id']}")
    return response['id']

def wait_for_batch_completion(batch_job_id):
    """
    Wait for the batch job to complete.

    Args:
        batch_job_id (str): The Batch Job ID.
    """
    while True:
        batch_status = openai.Batch.retrieve(batch_job_id)
        status = batch_status['status']
        print(f"Batch status: {status}")
        if status in ["completed", "failed", "cancelled"]:
            break
        time.sleep(30)  # Check status every 30 seconds
    print(f"Batch job finished with status: {status}")
    return status

def download_output_file(batch_job_id, output_file_path):
    """
    Download the output file for the completed batch job.

    Args:
        batch_job_id (str): The Batch Job ID.
        output_file_path (str): Path to save the output JSONL file.
    """
    batch_info = openai.Batch.retrieve(batch_job_id)
    output_file_url = batch_info.get('output_file_url')

    if not output_file_url:
        print("Output file URL not available.")
        return

    print(f"Downloading output file from: {output_file_url}")
    response = requests.get(output_file_url)
    with open(output_file_path, 'wb') as f:
        f.write(response.content)
    print(f"Output file saved at: {output_file_path}")

def parse_output_file(output_file_path):
    """
    Parse the output JSONL file and print the results.

    Args:
        output_file_path (str): Path to the output JSONL file.
    """
    with open(output_file_path, 'r') as f:
        for line in f:
            result = json.loads(line)
            input_comment = result['input']['messages'][-1]['content']
            output_label = result['response']['choices'][0]['message']['content']
            print(f"Comment: {input_comment}\nLabel: {output_label}\n")

if __name__ == "__main__":
    # Example instructions and comments
    labeling_instructions = """
    You are a labeling assistant. Label the comment as one of the following:
    1. Positive
    2. Negative
    3. Neutral
    Only respond with the label (Positive, Negative, or Neutral).
    """

    comments_to_label = [
        "I love this product! It's fantastic.",
        "This is the worst experience I've ever had.",
        "It was okay, not great but not terrible either.",
        "Absolutely wonderful, highly recommend it!",
        "I wouldn't buy this again, very disappointed."
    ]

    # Define file paths
    batch_file_path = "input_batch.jsonl"
    output_file_path = "output_batch.jsonl"

    # Step 1: Prepare the batch file
    prepare_batch_file(comments_to_label, labeling_instructions, batch_file_path)

    # Step 2: Upload the batch file
    uploaded_file_id = upload_batch_file(batch_file_path)

    # Step 3: Create a batch job
    batch_job_id = create_batch_job(uploaded_file_id, output_file_name="labeled_comments")

    # Step 4: Wait for the batch job to complete
    status = wait_for_batch_completion(batch_job_id)

    # Step 5: Download the output file if completed
    if status == "completed":
        download_output_file(batch_job_id, output_file_path)

    # Step 6: Parse and display the results
    parse_output_file(output_file_path)
