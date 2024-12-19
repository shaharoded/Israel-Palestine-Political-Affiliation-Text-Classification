import openai
import json
import csv
import time
import requests

# Local Code
from secret_keys import OPENAI_API_KEY
from Config.data_tagging_config import *


# Set your OpenAI API key in a SecretKeys.py file
openai.api_key = OPENAI_API_KEY


class AITagger:
    '''
    Automated LLM based comments tagger.
    Will take a csv file, send a batch request and return labels to the dataset.
    Class has an option for batch tag on your entire untagged dataset, or testing module
    which will provide a tag on a small subset of the data and accuracy score.
    Modify parameters in the config file.
    '''
    def __init__(self, engine, temperature, comment_id_idx=0, text_comment_idx=1):
        """
        Initialize the AITagger instance.

        Args:
            engine (str): OpenAI model engine to use (e.g., 'gpt-4').
            temperature (float): Sampling temperature for the model.
            comment_id_idx (int): Column index for comment IDs in the input file.
            text_comment_idx (int): Column index for comments in the input file.
        """
        self.engine = engine
        self.temperature = temperature
        self.comment_id_idx = comment_id_idx
        self.text_comment_idx = text_comment_idx

    def prepare_batch_file(self, input_file_path, instructions, output_batch_file_path):
        """
        Prepare a JSONL file for OpenAI Batch API with comment IDs.
        """
        comments = []
        with open(input_file_path, 'r') as csv_file:
            reader = csv.reader(csv_file)
            next(reader)  # Skip the header
            for row in reader:
                comment_id = row[self.comment_id_idx].strip()
                comment = row[self.text_comment_idx].strip()
                comments.append((comment_id, comment))

        with open(output_batch_file_path, 'w') as f:
            for comment_id, comment in comments:
                data = {
                    "id": comment_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.engine,
                        "messages": [
                            {"role": "system", "content": instructions},
                            {"role": "user", "content": f"Comment: {comment}"}
                        ],
                        "temperature": self.temperature
                    }
                }
                f.write(json.dumps(data) + "\n")
        print(f"Batch file prepared at: {output_batch_file_path}")

    def upload_batch_file(self, batch_file_path):
        """
        Upload the JSONL file to OpenAI.
        """
        with open(batch_file_path, "rb") as f:
            response = openai.File.create(file=f, purpose="batch")
        print(f"Job file uploaded. File ID for tracking: {response['id']}")
        return response['id']

    def create_batch_job(self, file_id, output_file_name):
        """
        Create a batch job to process the uploaded file.
        """
        response = openai.Batch.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            output_file_name=output_file_name
        )
        time.sleep(10)
        print(f"Batch job created. Job ID: {response['id']}")
        return response['id']

    def wait_for_batch_completion(self, batch_job_id):
        """
        Wait for the batch job to complete.
        """
        counter = 0
        while True:
            batch_status = openai.Batch.retrieve(batch_job_id)
            status = batch_status['status']
            print(f"Batch status: {status}")
            if status in ["completed", "failed", "canceled"]:
                print(f'[Runtime Status]: Job completed, time = {counter+1}h')
                break
            time.sleep(3600)
            print(f'[Runtime Status]: Job still in process, time = {counter+1}h')
            counter += 1
        print(f"Batch job finished with status: {status}")
        return status

    def download_batch_output(self, batch_job_id, output_file_path):
        """
        Download the output file for the completed batch job, parse its contents,
        and save the results to a CSV file.
        """
        batch_info = openai.Batch.retrieve(batch_job_id)
        output_file_url = batch_info.get('output_file_url')

        if not output_file_url:
            print("[Error]: Output file URL not available.")
            return

        response = requests.get(output_file_url)
        if response.status_code != 200:
            print(f"[Error]: Failed to download file. Status code: {response.status_code}")
            return

        with open(output_file_path, mode='w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=["comment_id", "comment", "label"])
            csv_writer.writeheader()

            for line in response.content.decode('utf-8').splitlines():
                try:
                    result = json.loads(line)
                    csv_writer.writerow({
                        "comment_id": result['id'],
                        "comment": result['input']['messages'][-1]['content'],
                        "label": result['result']['choices'][0]['message']['content']
                    })
                except KeyError as e:
                    print(f"[Error]: Missing expected key in result. {e}")
                except json.JSONDecodeError as e:
                    print(f"[Error]: Failed to parse JSON line. {e}")

        print(f"Parsed results saved to {output_file_path}")


'''
If TEST the process should run a few randomally chosen tests (not as batch) and save an
output file for examination. This should produce the accuracy score, outputting the taggs compared 
to the real tags in order to allow the user to fine tune the prompt.

If TEST is False run a batch job on all of the data, assuming the prompt was already verified
and create a tagged file for research based on FULL_DATA_PATH.
'''
if __name__ == "__main__":
    # Define file paths
    input_file_path = TAGGED_DATA_PATH if TEST_MODE else FULL_DATA_PATH
    batch_file_path = BATCH_FILE_PATH
    output_file_path = OUTPUT_FILE_PATH
    labeling_instructions = LABELING_INSTRUCTIONS

    # Step 1: Prepare the batch file
    prepare_batch_file(input_file_path, 
                       labeling_instructions, 
                       batch_file_path)

    # Step 2: Upload the batch file
    uploaded_file_id = upload_batch_file(batch_file_path)

    # Step 3: Create a batch job
    batch_job_id = create_batch_job(uploaded_file_id, output_file_path=OUTPUT_FILE_PATH)

    # Step 4: Wait for the batch job to complete
    status = wait_for_batch_completion(batch_job_id)

    # Step 5: Download and parse the output file if completed
    if status == "completed":
        download_batch_output(batch_job_id, output_file_path)
