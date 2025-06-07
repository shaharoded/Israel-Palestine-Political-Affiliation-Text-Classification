'''
Class to contain the different model, each should be easiliy called for initialization,
fit and predict.
'''
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import pickle

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
                    "layers": list[int]
                }
        """
        super(DNN, self).__init__()
        layers = []
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
            self.criterion = nn.CrossEntropyLoss()  # Cross-Entropy Loss for multiclass problem
            self.num_epochs = config["num_epochs"]
        elif model_type == "svm":
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
        if self.model_type in ["svm", "xgboost"]:
            self.log and print(f'[Model Fit Status]: Fitting the model...')
            self.model.fit(X_train, y_train)
        elif self.model_type in ["logistic_regression", "dnn"]:
            self.log and print(f'[Model Fit Status]: Fitting the model...')
            self.model.train()
            for epoch in range(self.num_epochs):
                for _, (features, labels) in enumerate(train_dataloader):
                    self.optimizer.zero_grad()
                    outputs = self.model(features.float())
                    loss = self.criterion(outputs.squeeze(), labels.long())
                    loss.backward()
                    self.optimizer.step()
                self.log and print(f"Epoch {epoch + 1}: Training Loss = {loss.item()}")


    def predict(self, test_data_package):
        """
        Predicts labels for the given test data.
        Works with the output of the function get_dataloader which gets the desired datashape
        per model.

        Args:
            test_data_package (tuple): A tuple containing (DataLoader, (X, y))
        
        Returns:
            list: Predicted labels.
        """
        predictions = []
        test_dataloader, (X_test, _) = test_data_package
        if self.model_type in ["svm", "xgboost"]:
            self.log and print(f'[Model Pred Status]: Generating predictions...')
            predictions = self.model.predict(X_test)
        elif self.model_type in ["logistic_regression", "dnn"]:
            self.log and print(f'[Model Pred Status]: Generating predictions...')
            self.model.eval()
            with torch.no_grad():
                for _, (features, _) in enumerate(test_dataloader):
                    features = features.to(DEVICE)
                    outputs = self.model(features.float())  # outputs.shape should be (batch_size, 3)
                    _, predictions_batch = torch.max(outputs, 1)  # Get the index of the max probability for each sample
                    predictions.extend(predictions_batch.int().tolist())  # Append predictions

        return predictions


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
    print(f'[Testing Status]: Building datasets and dataloaders...')
    print(f'[Testing Status]: Building train dataloader...')
    
    # Create Embedding Dataset
    text_dataset = TextDataset(
    csv_path          = DATA_PATH,
    id_column_idx     = ID_COLUMN_IDX,
    comment_column_idx= COMMENT_COLUMN_IDX,
    label_column_idx  = LABEL_COLUMN_IDX,
    split_column_idx  = SUBSET_COLUMN_IDX,  # TRAIN / VAL / TEST column
    augmented_classes = [],                 # ‑‑ no aug
    augmentation_ratio= 0,
    undersampling_targets = {},             # ‑‑ no undersampling
    )
    embedding_dataset = EmbeddingDataset(text_dataset=text_dataset,
                                        embedder=Embedder(), 
                                        embedding_method=EMBEDDING_METHOD)
    
    train_ds = embedding_dataset.get_subset('TRAIN')   # returns EmbeddingDataset._View
    test_ds = embedding_dataset.get_subset('TEST')   # returns EmbeddingDataset._View
    
    # Get the DataLoader with embeddings
    # Note the multiple objects outputted here
    train_data_package = get_dataloader(train_ds,  
                                batch_size=BATCH_SIZE,
                                shuffle=False, 
                                num_workers=2)
    
    test_data_package = get_dataloader(test_ds,  
                            batch_size=BATCH_SIZE,
                            shuffle=False, 
                            num_workers=2)
    
    # Choose a model
    print(f'[Testing Status]: Fitting a classifier...')
    model_config = MODEL_CONFIG.get(MODEL_TYPE)

    # Initialize and train the model
    classifier = Classifier(model_config, 
                            model_type=MODEL_TYPE,
                            log=True)
    classifier.fit(train_data_package)

    # Test the model
    print(f'[Testing Status]: Testing on test subset...')
    predictions = classifier.predict(test_data_package)
    
    # # Extract comment IDs and real labels from the test dataset
    # test_comment_ids = embedding_dataset.text_dataset.data.iloc[embedding_dataset.text_dataset.idx_split["TEST"], text_dataset.id_column_idx].tolist()
    # real_labels = embedding_dataset.text_dataset.data.iloc[embedding_dataset.text_dataset.idx_split["TEST"], text_dataset.label_column_idx].to_numpy()
    
    # # Create a DataFrame for predictions
    # results_df = pd.DataFrame({
    #     "Comment ID": test_comment_ids,
    #     "Real Label": real_labels,
    #     "Predicted Label": predictions
    # })

    # # Save the DataFrame to a CSV file
    # results_csv_path = "classification_results.csv"
    # results_df.to_csv(results_csv_path, index=False)
    # print(f"Results saved to {results_csv_path}")

    # Show accuracy score per class + macro (classification report)
    # Calculate accuracy and show classification report
    _, _ = assess_model(predictions, test_data_package, valid_labels=[0, 1, 2])
