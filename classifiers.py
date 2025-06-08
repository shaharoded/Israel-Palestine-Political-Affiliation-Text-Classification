'''
Class to contain the different model, each should be easiliy called for initialization,
fit and predict.
'''
import os
import ast
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import pickle
from tqdm import tqdm

# Local Code
from Config.classifiers_config import *
from Config.dataset_config import *
from dataset import TextDataset, EmbeddingDataset, get_dataloader
from embedder import Embedder

# Setting random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)  # For CUDA (GPU) if you're using it

class DNN(nn.Module):
    def __init__(self, config):
        """
        Initializes a Deep Neural Network (DNN) with the given configuration.

        Args:
            config (dict): Configuration for the DNN.
                {
                    "learning_rate": float,
                    "batch_norm": bool,
                    "drop_out": float,
                    "layers": str(list[int])
                }
        """
        super(DNN, self).__init__()
        layers = []
        if isinstance(config["layers"], str):
            config["layers"] = list(ast.literal_eval(config["layers"]))
        input_size = config["layers"][0]

        # Iterate through the hidden layers
        for output_size in config["layers"][1:-1]:  # Skip the last layer (number of classes)
            layers.append(nn.Linear(input_size, output_size))
            if config.get("batch_norm", False):
                layers.append(nn.BatchNorm1d(output_size))
            layers.append(nn.ReLU())
            if config.get("drop_out", 0.0) > 0:
                layers.append(nn.Dropout(config["drop_out"]))
            input_size = output_size

        # Final classification layer
        num_classes = config["layers"][-1]
        layers.append(nn.Linear(input_size, num_classes))  # Last layer matches num_classes
        # layers.append(nn.Softmax(dim=1))  # Softmax activation for multi-class output

        self.model = nn.Sequential(*layers)
        self.learning_rate = config["learning_rate"]

    def forward(self, x):
        return self.model(x)
    
class Classifier:   
    def __init__(self, config, model_type, log=True, init_model=True):
        """
        Initializes the classifier based on the model type and configuration.

        Args:
            config (dict): Configuration for the model. Format varies based on the model type.
            model_type (str): One of "logistic_regression", "svm", "xgboost", or "dnn".
            log (bool): Print the loss progress? Redundent if epochs are optimized externally.
            init_model (bool): Allows model initialization for loaded model. Called like clf.load() after init.
        """
        self.model_type = model_type
        self.model_params = config          # save for cloning
        self.log = log

        if not init_model:
            self.model = None
            return

        if model_type in ["logistic_regression", "dnn"]:
            # A one layered logistic regression implementation using the DNN class
            self.model = DNN(config)
            self.optimizer = optim.Adam(self.model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
            self.criterion = None  # Will be created in fit() based on label distribution
            self.num_epochs = config["num_epochs"]
        elif model_type == "svm":
            config.setdefault("class_weight", "balanced")  # Balance class weights
            self.model = SVC(random_state=42, **config)
        elif model_type == "xgboost":
            self.model = XGBClassifier(random_state=42, **config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")


    def fit(self, train_data_package):
        """
        Fits the model to the training data.
        Allows to load an existing model, if training already reached a best saved model.
        Works with the output of the function get_dataloader which gets the desired datashape
        per model.

        Args:
            train_data_package (tuple): A tuple containing (DataLoader, (X, y))
        """
        # ----------------- Determine checkpoint path -----------------
        checkpoint_path = os.path.join(CHECKPOINTS, f'best_{self.model_type}.pt' if self.model_type in ["logistic_regression", "dnn"] else f'best_{self.model_type}.pkl')
        if os.path.exists(checkpoint_path):
            self.log and print(f"[Model Fit Status]: Loading pre-trained {self.model_type} from checkpoint.")
            
            if self.model_type in ["logistic_regression", "dnn"]:
                self.model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
                self.model.to(DEVICE)
                self.model.eval()
            else:  # sklearn/xgboost
                with open(checkpoint_path, "rb") as f:
                    self.model = pickle.load(f)
            return

        # ----------------- Otherwise, train the model -----------------
        train_dataloader, (X_train, y_train) = train_data_package
        if self.model_type == "svm":
            self.log and print(f'[Model Fit Status]: Fitting the model...')
            self.model.fit(X_train, y_train)
        elif self.model_type == "xgboost":
            self.log and print(f'[Model Fit Status]: Fitting the model...')
            y_train = y_train.detach().cpu().numpy() if isinstance(y_train, torch.Tensor) else y_train
            class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train).astype(int), y=y_train.astype(int))
            sample_weight = np.array([class_weights[label] for label in y_train])
            self.model.fit(X_train, y_train, sample_weight=sample_weight)
        elif self.model_type in ["logistic_regression", "dnn"]:
            self.log and print(f'[Model Fit Status]: Fitting the model...')
            y_train = y_train.detach().cpu().numpy() if isinstance(y_train, torch.Tensor) else y_train
            class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train).astype(int), y=y_train.astype(int))
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
            self.model.train()
            for epoch in range(self.num_epochs):
                for _, (features, labels) in enumerate(train_dataloader):
                    self.optimizer.zero_grad()
                    outputs = self.model(features.float())
                    loss = self.criterion(outputs.squeeze(), labels.long())
                    loss.backward()
                    self.optimizer.step()
                self.log and print(f"Epoch {epoch + 1}: Training Loss = {loss.item()}")


    def _batch_predict_sklearn(self, X, batch_size=1024, proba=False):
        preds, probs = [], []
        for i in tqdm(range(0, len(X), batch_size), desc="Predicting", ncols=100):
            X_batch = X[i:i+batch_size]
            if proba and hasattr(self.model, "predict_proba"):
                batch_probs = self.model.predict_proba(X_batch)
                probs.append(batch_probs)
                preds.append(batch_probs.argmax(axis=1))
            else:
                preds.append(self.model.predict(X_batch))
        predictions = np.concatenate(preds)
        if proba and probs:
            probas = np.concatenate(probs)
            return predictions, probas
        return predictions, None

    def predict(self, test_data_package, proba=False):
        """
        Predicts labels for the given test data. Optionally returns class probabilities.

        Args:
            test_data_package (tuple): (DataLoader, (X, y))
            proba (bool): If True, also return class probabilities.

        Returns:
            list: Predicted labels
            (optional) list: Predicted class probabilities
        """
        predictions = []
        probas = []

        test_dataloader, (X_test, _) = test_data_package

        if self.model_type in ["svm", "xgboost"]:
            self.log and print(f'[Model Pred Status]: Generating predictions...')
            predictions, probas = self._batch_predict_sklearn(X_test, batch_size=1024, proba=proba)
            return (predictions.tolist(), probas.tolist()) if proba else predictions.tolist()

        elif self.model_type in ["logistic_regression", "dnn"]:
            self.log and print(f'[Model Pred Status]: Generating predictions...')
            predictions = []
            probas = []
            self.model.eval()
            with torch.no_grad():
                for _, (features, _) in enumerate(tqdm(test_dataloader, desc="Predicting", ncols=100)):
                    features = features.to(DEVICE)
                    outputs = self.model(features.float())  # shape: (batch_size, num_classes)
                    if proba:
                        probs = torch.softmax(outputs, dim=1)
                        probas.extend(probs.cpu().tolist())
                        preds = torch.argmax(probs, dim=1)
                    else:
                        preds = torch.argmax(outputs, dim=1)
                    predictions.extend(preds.cpu().tolist())
            return (predictions, probas) if proba else predictions


    def load(self, checkpoint_path=None):
        """
        Loads a pre-trained model from disk.
        Args:
            checkpoint_path (str, optional): Path to checkpoint file. If None, uses default from config.
        """
        if checkpoint_path is None:
            checkpoint_path = os.path.join(CHECKPOINTS, f'best_{self.model_type}.pt' if self.model_type in ["logistic_regression", "dnn"] else f'best_{self.model_type}.pkl')

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"[Load Error] No checkpoint found at: {checkpoint_path}")

        if self.model_type in ["logistic_regression", "dnn"]:
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
            self.model.to(DEVICE)
            self.model.eval()
        else:
            with open(checkpoint_path, "rb") as f:
                self.model = pickle.load(f)

        self.log and print(f"[Model Load Status]: Loaded pretrained {self.model_type} from {checkpoint_path}")

# Function to calculate model's accuracy
def assess_model(predictions, test_data_package, valid_labels=[0, 1, 2]):
    '''
    Uses the y_test from test_data_package to evaluate the model while ignoring bad labels.
    
    Args:
        predictions (list): Predicted labels from the model.
        test_data_package (tuple): A tuple containing (DataLoader, (X_test, y_test)).
        valid_labels (list): A list of valid labels to consider for the report.
        
    Returns:
        tuple: (F1 (float), classification_report)
    '''
    _, (_, true_labels) = test_data_package
    true_labels = true_labels.cpu().numpy().astype(int) if isinstance(true_labels, torch.Tensor) else np.array(true_labels).astype(int)
    print('[Debug] Glimpse of true labels:', true_labels[:10], 'Length:', len(true_labels))
    
    # Print confusion matrix (truth table)
    conf_matrix = confusion_matrix(true_labels, predictions, labels=valid_labels)
    print("\nConfusion Matrix (Truth Table):")
    conf_matrix_df = pd.DataFrame(conf_matrix, 
                                  index=[f"True {label}" for label in valid_labels], 
                                  columns=[f"Pred {label}" for label in valid_labels])
    print(conf_matrix_df)
    
    # Print F1-score (micro)
    f1_micro = f1_score(true_labels, predictions, average='micro')
    print(f"\nF1-Score (Micro): {f1_micro:.4f}")
    
    # Return accuracy and classification report
    class_report = classification_report(true_labels, predictions, zero_division=0)
    print("\nClassification Report:")
    print(class_report)
    return f1_micro, class_report

    
if __name__ == "__main__":
    print(f'[Pipeline Status]: Building datasets and dataloaders...')

    # 1. Load full dataset
    text_dataset = TextDataset(
        csv_path=DATA_PATH,
        id_column_idx=ID_COLUMN_IDX,
        comment_column_idx=COMMENT_COLUMN_IDX,
        label_column_idx=LABEL_COLUMN_IDX,
        split_column_idx=SUBSET_COLUMN_IDX,
        augmented_classes=[],
        augmentation_ratio=0,
        undersampling_targets={},
    )

    embedding_dataset = EmbeddingDataset(
        text_dataset=text_dataset,
        embedder=Embedder(),
        embedding_method=EMBEDDING_METHOD
    )

    # 2. Use TRAIN+VAL for final training
    train_ds = embedding_dataset.get_subset("TRAIN+VAL")
    test_ds = embedding_dataset.get_subset("TEST")

    train_data_package = get_dataloader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_data_package = get_dataloader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # 3. Train & Evaluate Each Model
    for model_type, model_config in MODEL_CONFIG.items():
        print(f'\n[Pipeline Status]: Processing model: {model_type}')
        checkpoint_path = os.path.join(
            CHECKPOINTS,
            f'best_{model_type}.pt' if model_type in ["logistic_regression", "dnn"]
            else f'best_{model_type}.pkl'
        )

        # Initialize
        classifier = Classifier(model_config, model_type=model_type, log=True)

        # Train only if checkpoint missing
        if not os.path.exists(checkpoint_path):
            print(f"[Checkpoint Missing]: Training {model_type}...")
            classifier.fit(train_data_package)
        else:
            print(f"[Checkpoint Found]: Loading {model_type}...")
            classifier.load(checkpoint_path)

        # Predict on test
        print(f"[Prediction Status]: Predicting with {model_type} on test set...")
        predictions = classifier.predict(test_data_package)

        # Evaluate
        print(f"[Evaluation]: {model_type}")
        f1, report = assess_model(predictions, test_data_package, valid_labels=[0, 1, 2])

        # Save predictions to CSV
        test_comment_ids = embedding_dataset.text_dataset.data.iloc[
            embedding_dataset.text_dataset.idx_split["TEST"],
            text_dataset.id_column_idx
        ].tolist()
        real_labels = embedding_dataset.text_dataset.data.iloc[
            embedding_dataset.text_dataset.idx_split["TEST"],
            text_dataset.label_column_idx
        ].to_numpy()

        results_df = pd.DataFrame({
            "Comment ID": test_comment_ids,
            "Real Label": real_labels,
            "Predicted Label": predictions
        })

        results_csv_path = f"classification_results_{model_type}.csv"
        results_df.to_csv(results_csv_path, index=False)
        print(f"[CSV Saved]: {results_csv_path}")
