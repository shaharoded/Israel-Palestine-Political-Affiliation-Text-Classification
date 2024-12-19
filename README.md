# Israel-Palestine-Political-Affiliation-Text-Classification
Work in progress...

## Usage
To get started, adjust the ocnfigurations in `Config` file, ensure that `secret_keys.py` is in place, and run `main.py` which will offer you different options.

### Human Supervised LLM Automated Tagging
In the module `llm_tagger.py` there is a pipeline for automating the labeling process of the full research dataset based on a sample, manually tagged dataset (created by us).
Use the `TEST` parameter from 

You'll need to handle installations:

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Git Initiation

```
git status
git remote remove origin
git remote add origin https://github.com/shaharoded/Israel-Palestine-Political-Affiliation-Text-Classification.git
git remote -v
```

## Git Updates
```
git add .
git commit -m "commit message"
git push -f origin main
````
