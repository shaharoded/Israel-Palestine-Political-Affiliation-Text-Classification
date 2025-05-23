import openai
from openai import OpenAI
from transformers import BertTokenizer
import json
import csv
import time
import os
import random
from copy import deepcopy
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import pandas as pd

# Local Code
from secret_keys import OPENAI_API_KEYS
from Config.data_tagging_config import *


# Set your OpenAI API key in a SecretKeys.py file
client = OpenAI(api_key=OPENAI_API_KEYS.get('shahar_personal_key'))
TOKENIZER = BertTokenizer.from_pretrained('bert-base-cased')


class AITagger:
    '''
    Automated LLM based comments tagger.
    Will take a csv file, send a batch request and return labels to the dataset.
    Class has an option for batch tag on your entire untagged dataset, or testing module
    which will provide a tag on a small subset of the data and accuracy score.
    Modify parameters in the config file.
    '''
    def __init__(self, batch_row_format, test_batch_size,
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
        self.batch_row_format = batch_row_format
        self.system_prompt_length = self.__count_tokens(json.dumps(batch_row_format)) # Calculate length of batch row structure
        self.test_batch_size = test_batch_size
        self.id_column_idx = id_column_idx
        self.comment_column_idx = comment_column_idx
        self.label_column_idx = label_column_idx

    
    def __count_tokens(self, text):
        """
        Assess the number of tokens in a text.
        """
        encoding = TOKENIZER.tokenize(text)
        return len(encoding)


    def __generate_response(self, comment):
        """
        Modified function to work with the openai API to generate a response for a single prompt.
        """
        if self.__count_tokens(comment) >= MAX_COMMENT_LENGTH:
            return 'Comment is too long for classification'
        output = client.chat.completions.create(
            messages=[
                {"role": "system", "content": [{'type': 'text', 'text': self.batch_row_format.get("body", {}).get("messages", [])[0].get("content", None)}]},
                {"role": "user", "content": [{'type': 'text', 'text': f'Comment: {comment}'}]},
            ],
            model=self.batch_row_format.get("body",{}).get("model", None),
            #max_tokens=self.engine_max_tokens,
            temperature=self.batch_row_format.get("body",{}).get("temperature", 0.0),  # How "creative" with variaced responses
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
            tup (float): accuracy, f1 scores for the test.
        """
        print("[Test Mode]: Testing model on a random subset of tagged data.")

        # Step 1: Read the tagged data
        tagged_data = []
        with open(tagged_file_path, 'r', encoding="utf-8", errors="ignore") as csv_file:
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
        true_labels = []
        predicted_labels = []

        for comment_id, comment, true_label in test_batch:
            predicted_label = self.__generate_response(comment)
            true_labels.append(true_label)
            predicted_labels.append(predicted_label)
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
        
        # Step 5: Calculate precision, recall, and F1 score
        _, _, f1, _ = precision_recall_fscore_support(
            true_labels, predicted_labels, average="weighted", zero_division=0
        )
        print(f"[Test Mode]: F1 Score on test batch: {f1:.2f}")

        # Step 6: Generate and print confusion matrix
        print("[Test Mode]: Confusion Matrix:")
        labels = sorted(set(true_labels))  # Ensure consistent label order
        cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        print(cm_df)

        # Step 7: Save mismatched predictions to a CSV file
        if mismatched_predictions:
            with open(output_csv_file_path, mode='w', newline='', encoding='utf-8', errors="ignore") as csvfile:
                csv_writer = csv.DictWriter(csvfile, fieldnames=["comment_id", "comment", "true_label", "predicted_label"])
                csv_writer.writeheader()
                csv_writer.writerows(mismatched_predictions)
            print(f"[Test Mode]: Mismatched predictions saved to {output_csv_file_path}")

        return accuracy, f1
    
    def __process_dataset(self, untagged_file_path, output_csv_file_path):
        """
        Tag every comment in `untagged_file_path` and stream the results
        to `output_csv_file_path` in 500‑row chunks.
        """
        print("[Status]: Tagging the comments using the regular (non‑batch) API.")

        # --- 1️⃣  Read the un‑tagged data -------------------------------------------------
        with open(untagged_file_path, newline="", encoding="utf‑8", errors="ignore") as fh:
            reader  = csv.reader(fh)
            header  = next(reader)                                    # skip CSV header
            raw_rows = [(row[self.id_column_idx].strip(),
                        row[self.comment_column_idx].strip())
                        for row in reader]

        # --- 2️⃣  Helper that APPENDS a chunk to the CSV ---------------------------------
        def _flush_chunk(chunk):
            """Append <chunk> (list[dict]) to <output_csv_file_path>."""
            df = pd.DataFrame(chunk)
            file_exists = os.path.isfile(output_csv_file_path)
            df.to_csv(
                output_csv_file_path,
                mode    = "a" if file_exists else "w",   # append if the file is already there
                header  = not file_exists,               # write header only once
                index   = False
            )

        # --- 3️⃣  Generate predictions and stream to disk --------------------------------
        chunk, CHUNK_SIZE = [], 500
        for i, (comment_id, comment) in enumerate(raw_rows, start=1):
            chunk.append(
                {
                    "comment_id":      comment_id,
                    "comment":         comment,
                    "predicted_label": self.__generate_response(comment)
                }
            )

            if i % CHUNK_SIZE == 0:
                print(f"[Status]:   {i} rows processed – saving checkpoint…")
                _flush_chunk(chunk)
                chunk.clear()              # free memory

        # --- 4️⃣  Flush whatever is left --------------------------------------------------
        if chunk:
            _flush_chunk(chunk)

        print("[Status]: Job finished!")

    def __upload_batch_file(self, batch_file_path):
        """
        Upload the JSONL file to OpenAI.
        """
        with open(batch_file_path, "rb") as f:
            response = client.files.create(file=f, purpose="batch")
        print(f"[Job Status]: Job file uploaded. File ID for tracking: {response.id}")
        return response.id


    def __create_batch_job(self, file_id):
        """
        Create a batch job to process the uploaded file.
        """
        response = client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        print(f"[Job Status]: Batch job created. Job ID: {response.id}")
        return response.id


    def __wait_for_batch_completion(self, batch_job_id):
        """
        Wait for the batch job to complete. Delete the job if it fails.
        """
        counter = 0
        while True:
            try:
                batch_info = client.batches.retrieve(batch_job_id)
                
                # Safely access the status attribute
                status = getattr(batch_info, "status", None)
                
                if status is None:
                    print("[Error]: Unable to retrieve the status from the batch information.")
                    break
                
                if status in ["completed", "failed", "cancelled"]:
                    print(f'[Job Status]: Job completed, time = {counter+1}m')
                    break
                
                print(f'[Job Status]: Job still in process, status = {status}, time = {counter + 1}m')
                counter += 1
                time.sleep(60)  # Wait a little before checking again
            except openai.APIConnectionError:
                print(f"[Warning]: Connection error on mini-batch, check your internet connection. Will retry in 60 seconds")
                counter += 1
                time.sleep(60)
                

        print(f"[Job Status]: Batch job finished with status: {status}")

        # Delete the job if it failed
        if status == "failed":
            self.debug_batch_failure(batch_job_id)
        
        return status


    def __download_batch_output(self, batch_job_id, output_file_path):
        """
        Download the output file for the completed batch job, parse its contents,
        and save the results to a CSV file.
        Parsed content will have commentID and label (the tagger response). 
        You'll need to merge this output with the original file.
        """
        batch_info = client.batches.retrieve(batch_job_id)
        
        # Retrieve the output_file_id
        output_file_id = getattr(batch_info, 'output_file_id', None)
        if not output_file_id:
            print("[Error]: Output file ID not available.")
            return

        # Fetch the file content using the output_file_id
        try:
            response = client.files.content(output_file_id)
        except Exception as e:
            print(f"[Error]: Failed to download file content. Error: {e}")
            return

        with open(output_file_path, mode='w', newline='', encoding='utf-8', errors="ignore") as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=["comment_id", "label"])
            csv_writer.writeheader()

            for line in response.text.splitlines():
                try:
                    result = json.loads(line)
                    csv_writer.writerow({
                        "comment_id": result['custom_id'],
                        "label": LABELS_DECODER.get(result['response']['body']['choices'][0]['message']['content'], None)
                    })
                except KeyError as e:
                    print(f"[Error]: Missing expected key in result. {e}")
                except json.JSONDecodeError as e:
                    print(f"[Error]: Failed to parse JSON line. {e}")

        print(f"[Job Status]: Parsed results saved to {output_file_path}")
    
    
    def __process_batch(self, batch_comments, batch_file_path, output_csv_file_path):
        """
        Process a single batch: upload, execute, and append the output.

        Args:
            batch_comments (list): List of tuples containing comment IDs and comments.
            batch_file_path (str): Path for saving the batch JSONL file.
            output_csv_file_path (str): Path to save the combined output.
        """
        # Write the batch to a JSONL file
        with open(batch_file_path, 'w', encoding='utf-8') as f:
            for comment_id, comment in batch_comments:
                batch_row = deepcopy(self.batch_row_format)
                batch_row["custom_id"] = comment_id
                batch_row["body"]["model"] = self.batch_row_format.get("body",{}).get("model", None)
                batch_row["body"]["temperature"] = self.batch_row_format.get("body",{}).get("temperature", 0.0)
                batch_row["body"]["messages"][0]["content"] = self.batch_row_format.get("body", {}).get("messages", [])[0].get("content", None)
                batch_row["body"]["messages"][1]["content"] = f"Comment: {comment}"                
                f.write(json.dumps(batch_row) + "\n")
        
        # Submit the batch job
        file_id = self.__upload_batch_file(batch_file_path)
        job_id = self.__create_batch_job(file_id)

        print(f"[Job Status]: Submitted batch job {job_id}. Waiting for completion.")
        
        # Wait for job completion
        status = self.__wait_for_batch_completion(job_id)

        if status == "completed":
            # Download and process the output
            temp_output_file = os.path.join("Data", os.path.splitext(os.path.basename(batch_file_path))[0] + ".csv")
            self.__download_batch_output(job_id, temp_output_file)

            # Append results to the combined output file
            with open(temp_output_file, 'r', encoding='utf-8', errors='ignore') as temp_csvfile:
                temp_reader = csv.DictReader(temp_csvfile)
                with open(output_csv_file_path, mode='a', newline='', encoding='utf-8') as combined_csvfile:
                    combined_writer = csv.DictWriter(combined_csvfile, fieldnames=["comment_id", "label"])
                    for row in temp_reader:
                        combined_writer.writerow(row)

            # Optionally delete temporary files
            os.remove(temp_output_file)
            os.remove(batch_file_path)
            print(f"[Job Status]: Batch job {job_id} completed and output saved.")
        else:
            print(f"[Pipeline Error]: Batch job {job_id} failed with status: {status}")
        
        
    def __batch_process_dataset(self, input_file_path, batch_file_path_template, output_csv_file_path):
        """
        Split the input file into smaller mini-batches, upload each as a JSONL file,
        and concatenate their outputs into a single CSV file.

        Args:
            input_file_path (str): Path to the CSV file with comments.
            batch_file_path_template (str): Template for saving batch JSONL files (e.g., 'Data/Batches/mini_batch_{}.jsonl').            
            instructions (str): Labeling instructions for the model.
            output_csv_file_path (str): Path to save the combined output.
            batch_size (int): Maximum number of entries per mini-batch.
        """
        # Ensure the directory for batch files exists
        batch_dir = os.path.dirname(batch_file_path_template)
        os.makedirs(batch_dir, exist_ok=True)
        
        # Read and tokenize input comments
        comments = []
        with open(input_file_path, 'r', encoding='utf-8', errors='ignore') as csv_file:
            reader = csv.reader(csv_file)
            next(reader)  # Skip the header
            for row in reader:
                comment_id = row[self.id_column_idx].strip()
                comment = row[self.comment_column_idx].strip()
                comments.append((comment_id, comment))
        
        # Open the combined output file in write mode initially
        with open(output_csv_file_path, mode='w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=["comment_id", "label"])
            csv_writer.writeheader()  # Write header only once
        
        # Cleanup existing batches (prior failures / undocumented batches)
        # Iterate through batches and cancel active or in-progress ones
        batches = client.batches.list()
        for batch in batches.data:        
            batch_id = getattr(batch, 'id', None)
            status = getattr(batch, 'status', None)
            if status in ["in_progress", "queued"]:
                print(f"[Cleanup Status]: Existing batch {batch_id} with status: {status}. Canceling...")
                client.batches.cancel(batch_id)

        # Initialize batch processing
        batch_index = 0
        total_tokens = 0
        batch_comments = []
        print("[Job Status]: Starting batch processing...")
        for idx, (comment_id, comment) in enumerate(comments):
            comment_tokens = self.__count_tokens(comment)
            if comment_tokens < MAX_COMMENT_LENGTH:
                total_tokens += comment_tokens + self.system_prompt_length
            
                # Add the comment to the current batch
                batch_comments.append((comment_id, comment))
            
                # Check if token limit is reached or if this is the last comment
                # Pass batches of size +- BATCH_TOKENS_LIMIT
                if total_tokens >= BATCH_TOKENS_LIMIT or len(batch_comments) >= 0.95 * BATCH_REQUESTS_LIMIT or idx == len(comments) - 1:
                    # Process the current batch
                    self.__process_batch(batch_comments, batch_file_path_template.format(batch_index), output_csv_file_path)
                    
                    # Reset for the next batch
                    batch_index += 1
                    total_tokens = 0
                    batch_comments = []

        print("[Job Status]: Completed processing all batches.")


    def run_pipeline(self, input_file_path, batch_file_path, output_csv_file_path, test_mode=True, method='batch'):
        """
        Run the complete tagging pipeline with mini-batch processing.

        Args:
            input_file_path (str): Path to the CSV file (tagged/untagged comments).
            output_csv_file_path (str): Path to save the final combined output.
            test_mode (bool): If True, runs only the test mode.
            method (str): How to get the responses from OpenAI - Choose from ['batch', 'regular']. 
                            Batch is cheaper but might take a lot of time. Only applicable when test_mode=False.
        """
        if test_mode:
            print("[Test Mode]: Attempting model on small sample batch")
            self.__test_model(input_file_path, output_csv_file_path)
        else:
            if method == 'batch':
                print("[Batch Pipeline Start]: Processing mini-batches")
                self.__batch_process_dataset(input_file_path, batch_file_path, output_csv_file_path)
            else: # method == 'regular'
                print("[Regular Pipeline Start]: Processing comments")
                self.__process_dataset(input_file_path, output_csv_file_path)


    def debug_batch_failure(self, batch_job_id):
        """
        Retrieve details about a failed batch job.
        """
        batch_info = client.batches.retrieve(batch_job_id)

        # Print detailed information about the batch job
        print("[Batch Debug]: Batch job details:")
        print(batch_info)

        # Check if there's an 'error' attribute or log in the response
        error_details = getattr(batch_info, "error", None)
        if error_details:
            print("[Batch Debug]: Error details:")
            print(error_details)
        else:
            print("[Batch Debug]: No specific error details found in batch info.")
            


if __name__ == "__main__":
    # Define file paths
    input_file_path = TAGGED_DATA_PATH if TEST_MODE else FULL_DATA_PATH
    batch_file_path = BATCH_FILE_PATH
    output_file_path = OUTPUT_FILE_PATH

    # Initiate tagger object
    tagger = AITagger(
    batch_row_format=BATCH_ROW_FORMAT,
    test_batch_size=TEST_BATCH_SIZE,
    id_column_idx=ID_COLUMN_IDX,
    comment_column_idx=COMMENT_COLUMN_IDX,
    label_column_idx=LABEL_COLUMN_IDX
    )

    # Activate
    tagger.run_pipeline(input_file_path=input_file_path, 
                        batch_file_path=batch_file_path,
                        output_csv_file_path=output_file_path, 
                        test_mode=TEST_MODE,
                        method='regular')  
        
        