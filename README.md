# Israel-Palestine-Political-Affiliation-Text-Classification
Work in progress...

## Overview
This project focuses on creating a machine learning (ML) and deep learning (DL) pipeline for classifying social media comments related to the Israel-Palestine conflict into distinct political affiliations. To enable this, the project includes an **Automated Tagger** module that generates labeled data for training ML and DL models, otherwise untagged as in the original dataset.

## Features
- **Automated Tagger**: A module to generate labeled datasets from untagged social media comments using a Large Language Model (LLM) by OpenAI.
- **Customizable Configurations**: Easily adjustable parameters for LLM settings, input/output paths, and testing modes.
- **Support for ML/DL Model Training**: Generates high-quality tagged data to train task-specific models for political affiliation classification.

## Usage
1. Configure settings in the `Config` file:
   - Define file paths for input and output datasets.
   - Set parameters such as the OpenAI engine, sampling temperature, and test batch size.
2. Ensure `secret_keys.py` is in place with your OpenAI API key.

### Running the Automated Tagger
The `llm_tagger.py` module provides functionality to generate labels for an untagged dataset using the OpenAI API. Run the pipeline through `main.py`:

#### **Modes of Operation**
- **Test Mode** (`TEST_MODE = True`):
  - Tests the tagging process on a small, random subset of manually tagged data (`TAGGED_DATA_PATH`).
  - Outputs an Accuracy and F1 score and saves mismatched predictions for inspection in `OUTPUT_FILE_PATH`.
  - Ideal for fine-tuning the prompt and verifying the labeling quality.
- **Batch Mode** (`TEST_MODE = False`):
  - Processes the entire untagged dataset (`FULL_DATA_PATH`) using the OpenAI batch API.
  - Outputs a fully labeled dataset in `OUTPUT_FILE_PATH`.
  - Used to prepare training data for ML/DL models.

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
