# Israel-Palestine-Political-Affiliation-Text-Classification
Work in progress...

## Overview
This project focuses on creating a machine learning (ML) and deep learning (DL) pipeline for classifying social media comments related to the Israel-Palestine conflict into distinct political affiliations. To enable this, the project includes an **Automated Tagger** module that generates labeled data for training ML and DL models, otherwise untagged as in the original dataset.

## Features
- **Automated Tagger**: A module to generate labeled datasets from untagged social media comments (`llm_tagger`) using a Large Language Model (LLM) by OpenAI.
- **Dataset Class**: Responsible for text preprocessing for both classification process, augmentation, adversation and for handeling the dataloading for textual data (for finetuning of embedder) and for classifier training.
- **Customizable Configurations**: Easily adjustable parameters for LLM settings, , dataset settings, best model's configurations, and testing modes. Each major component has it's own configurations file under `Config` folder.

## Usage
1. Configure settings in the `Config` file:
   - Define file paths.
   - Set parameters (eg. OpenAI engine), augmentation ratio, and test batch size.
2. Ensure `secret_keys.py` is in place with your OpenAI API key.

### Running the Automated Tagger
The `llm_tagger.py` module provides functionality to generate labels for an untagged dataset using the OpenAI API. Run the pipeline through `main.py`:

- **Test Mode** (`TEST_MODE = True`):
  - Tests the tagging process on a small, random subset of manually tagged data (`TAGGED_DATA_PATH`).
  - Outputs an Accuracy and F1 score and saves mismatched predictions for inspection in `OUTPUT_FILE_PATH`.
  - Ideal for fine-tuning the prompt and verifying the labeling quality.
- **Batch Mode** (`TEST_MODE = False`):
  - Processes the entire untagged dataset (`FULL_DATA_PATH`) using the OpenAI batch API.
  - Outputs a fully labeled dataset in `OUTPUT_FILE_PATH`.
  - Used to prepare training data for ML/DL models.

### Running the Dataset
The `dataset.py` module is pretty straight forward and can be run through `main.py`. Adjust the path and choose the subset you wish to work with (pre-divided), set up the file paths for the research data and vectorization weights, set augmentation ratio, traget groups and the data shape of the dataloader, meaning - this module will create a textual dataloader on defualt, assuming you have yet trained a model, but on demand will create a vectorized dataset (with embeddings for comments), under the assumption you have already fine-tuned a model, and have it's weights available.
This module uses the `vectorize.py` (not yet developed) module for that.

### Installation
Set up the environment and install dependencies:

```bash
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Git Updates
```
git add .
git commit -m "commit message"
git push -f origin main
````
