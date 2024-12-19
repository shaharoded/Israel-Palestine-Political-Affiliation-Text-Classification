from openai import OpenAI
import json
import csv
import time
import random

# Local Code
from secret_keys import OPENAI_API_KEY
from Config.data_tagging_config import *


# Set your OpenAI API key in a SecretKeys.py file
client = OpenAI(api_key=OPENAI_API_KEY)


class AITagger:
    '''
    Automated LLM based comments tagger.
    Will take a csv file, send a batch request and return labels to the dataset.
    Class has an option for batch tag on your entire untagged dataset, or testing module
    which will provide a tag on a small subset of the data and accuracy score.
    Modify parameters in the config file.
    '''
    def __init__(self, engine, temperature, system_prompt, test_batch_size,
                 id_column_idx=0, comment_column_idx=1, label_column_idx = None):
        """
        Initialize the AITagger instance.
        Structure assumes the position of comment idx column and comment content column
        is identical in both full research file and manually tagged train file.

        Args:
            engine (str): OpenAI model engine to use (e.g., 'gpt-4').
            temperature (float): Sampling temperature for the model.
            system_prompt (str): Instructions on how to perform labeling, AKA assistant system prompt.
            test_batch_size (int): Number of random comments for testing
            id_column_idx (int): Column index for comment IDs in the input file.
            comment_column_idx (int): Column index for comments in the input file.
            label_column_idx (int) : Column index for manual labels in the tagged train file.
        """
        self.engine = engine
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.test_batch_size = test_batch_size
        self.id_column_idx = id_column_idx
        self.comment_column_idx = comment_column_idx
        self.label_column_idx = label_column_idx
    
    
    def __generate_response(self, comment):
        """
        Modified function to work with the openai API to generate a response for a single prompt.
        """
        output = client.chat.completions.create(
            messages=[
                {"role": "system", "content": [{'type': 'text', 'text': self.system_prompt}]},
                {"role": "user", "content": [{'type': 'text', 'text': f'Comment: {comment}'}]},
            ],
            model=self.engine,
            #max_tokens=self.engine_max_tokens,
            temperature=self.temperature,  # How "creative" with variaced responses
            response_format={
                "type": "text"
            }
        )

        output = output.choices[0].message.content
        
        # Decode output to label
        label = LABELS_DECODER.get(output, None)
        if not label:
            print(f"[Error]: Model returned an unidentified label: '{output}'")
        return label
    
    
    def __test_model(self, tagged_file_path, output_csv_file_path):
        """
        Test the model on a subset of tagged data and evaluate accuracy.

        Args:
            tagged_file_path (str): Path to the CSV file containing manually tagged data.
            output_csv_file_path (str): Path to save the mismatched predictions for inspection.

        Returns:
            float: Accuracy score for the test.
        """
        print("[Test Mode]: Testing model on a random subset of tagged data.")

        # Step 1: Read the tagged data
        tagged_data = []
        with open(tagged_file_path, 'r') as csv_file:
            reader = csv.reader(csv_file)
            header = next(reader)  # Skip the header
            for row in reader:
                comment_id = row[self.id_column_idx].strip()
                comment = row[self.comment_column_idx].strip()
                label = row[self.label_column_idx].strip() if self.label_column_idx is not None else None
                tagged_data.append((comment_id, comment, label))

        # Step 2: Sample a batch for testing
        if len(tagged_data) < self.test_batch_size:
            print(f"[Warning]: Test batch size ({self.test_batch_size}) exceeds the number of samples in the data ({len(tagged_data)}). Using all available data.")
            test_batch_size = len(tagged_data)
        else:
            test_batch_size = self.test_batch_size

        test_batch = random.sample(tagged_data, test_batch_size)

        # Step 3: Evaluate predictions
        mismatched_predictions = []
        correct_predictions = 0

        for comment_id, comment, true_label in test_batch:
            predicted_label = self.__generate_response(comment)
            if predicted_label == true_label:
                correct_predictions += 1
            else:
                mismatched_predictions.append({
                    "comment_id": comment_id,
                    "comment": comment,
                    "true_label": true_label,
                    "predicted_label": predicted_label
                })

        # Step 4: Calculate accuracy
        accuracy = correct_predictions / test_batch_size
        print(f"[Test Mode]: Accuracy on test batch: {accuracy * 100:.2f}%")

        # Step 5: Save mismatched predictions to a CSV file
        if mismatched_predictions:
            with open(output_csv_file_path, mode='w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.DictWriter(csvfile, fieldnames=["comment_id", "comment", "true_label", "predicted_label"])
                csv_writer.writeheader()
                csv_writer.writerows(mismatched_predictions)
            print(f"[Test Mode]: Mismatched predictions saved to {output_csv_file_path}")

        return accuracy


    def __prepare_batch_file(self, input_file_path, instructions, output_batch_file_path):
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

    def __upload_batch_file(self, batch_file_path):
        """
        Upload the JSONL file to OpenAI.
        """
        with open(batch_file_path, "rb") as f:
            response = client.files.create(file=f, purpose="batch")
        print(f"Job file uploaded. File ID for tracking: {response['id']}")
        return response['id']

    def __create_batch_job(self, file_id, output_file_name):
        """
        Create a batch job to process the uploaded file.
        """
        response = client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            output_file_name=output_file_name
        )
        time.sleep(10)
        print(f"Batch job created. Job ID: {response['id']}")
        return response['id']

    def __wait_for_batch_completion(self, batch_job_id):
        """
        Wait for the batch job to complete.
        """
        counter = 0
        while True:
            batch_info = client.batches.retrieve(batch_job_id)
            status = batch_info.get('status')
            print(f"Batch status: {status}")
            if status in ["completed", "failed", "cancelled"]:
                print(f'[Runtime Status]: Job completed, time = {counter+1}h')
                break
            time.sleep(3600)
            print(f'[Runtime Status]: Job still in process, time = {counter+1}h')
            counter += 1
        print(f"Batch job finished with status: {status}")
        return status

    def __download_batch_output(self, batch_job_id, output_file_path):
        """
        Download the output file for the completed batch job, parse its contents,
        and save the results to a CSV file.
        """
        batch_info = client.batches.retrieve(batch_job_id)
        output_file_url = batch_info.get('output_file_url')

        if not output_file_url:
            print("[Error]: Output file URL not available.")
            return

        response = client.files.content(output_file_url)
        if response.status_code != 200:
            print(f"[Error]: Failed to download file. Status code: {response.status_code}")
            return

        with open(output_file_path, mode='w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=["comment_id", "comment", "label"])
            csv_writer.writeheader()

            for line in response.text.splitlines():
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
    
    
    def run_pipeline(self, input_file_path, tagged_file_path, instructions, 
                     output_batch_file_path, output_csv_file_path, test_mode=True):
        """
        Run the complete tagging pipeline: prepare batch file, upload, create job, wait, and download output.

        Args:
            input_file_path (str): Path to the CSV file of the untagged comments.
            tagged_file_path (str): Path to the manually tagged csv sample file.
            instructions (str): Instructions for GPT to label the data.
            output_batch_file_path (str): Path to save the JSONL file for the batch job.
            output_csv_file_path (str): Path to save the processed output as a CSV file.
            test_mode (bool): If True, runs up to batch file preparation only for testing.
                            If False, completes the entire process.

        Returns:
            None
        """
        if test_mode:
            print("[Test Mode]: Attempting model on small sample batch")
            self.__test_model(tagged_file_path, output_csv_file_path)
        
        else:
            # Proceed with full process in non-test mode
            print("[Pipeline Start]: Initializing tagging process.")
            self.__prepare_batch_file(input_file_path, instructions, output_batch_file_path)
            file_id = self.__upload_batch_file(output_batch_file_path)
            job_id = self.__create_batch_job(file_id, "output.jsonl")
            status = self.__wait_for_batch_completion(job_id)

            if status == "completed":
                self.__download_batch_output(job_id, output_csv_file_path)
            else:
                print(f"[Pipeline Error]: Batch job did not complete successfully. Status: {status}")


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
