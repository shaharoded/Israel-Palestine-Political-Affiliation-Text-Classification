'''
Class to contain the different model, each should be easiliy called for initialization,
fit and predict.
'''
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Local Code
from Config.classifiers_config import *
from Config.dataset_config import *
from dataset import EmbeddingDataset, get_dataloader
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
        layers.append(nn.Softmax(dim=1))  # Softmax activation for multi-class output

        self.model = nn.Sequential(*layers)
        self.learning_rate = config["learning_rate"]

    def forward(self, x):
        return self.model(x)
    
class Classifier:
    def __init__(self, config, model_type, log=True):
        """
        Initializes the classifier based on the model type and configuration.

        Args:
            config (dict): Configuration for the model. Format varies based on the model type.
            model_type (str): One of "logistic_regression", "svm", "xgboost", or "dnn".
            log (bool): Print the loss progress? Redundent if epochs are optimized externally.
        """
        self.model_type = model_type
        self.log = log

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
        Works with the output of the function get_dataloader which gets the desired datashape
        per model.

        Args:
            train_data_package (tuple): A tuple containing (DataLoader, (X, y))
        """
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
                    outputs = self.model(features.float())  # outputs.shape should be (batch_size, 3)
                    _, predictions_batch = torch.max(outputs, 1)  # Get the index of the max probability for each sample
                    predictions.extend(predictions_batch.int().tolist())  # Append predictions
        return predictions

    
# Function to calculate model's accuracy
def calculate_accuracy(predictions, test_data_package, valid_labels=[0,1,2]):
    '''
    Uses the y_test from test_data_package to evaluate the model while ignoring bad labels.
    
    Args:
        test_data_package (tuple): A tuple containing (DataLoader, (X_test, y_test)).
        valid_labels (list): A list of valid labels to consider for the report.
        
    Return:
        tuple: (accuracy (float), classification_report)
    '''
    test_dataloader, (X_test, y_test) = test_data_package
    true_labels = y_test
    print('Glimpse of true labels: ', true_labels[:10], 'Length: ', len(true_labels))
    
    # If valid_labels are provided, filter out the bad labels
    if valid_labels:
        # Only consider the valid labels for true and predicted values
        valid_mask = [label in valid_labels for label in true_labels]
        true_labels = [label for label, mask in zip(true_labels, valid_mask) if mask]
        predictions = [pred for pred, mask in zip(predictions, valid_mask) if mask]
    
    print('Glimpse of true labels after cleanup: ', true_labels[:10], 'Length: ', len(true_labels))
    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predictions)
    
    # Return accuracy and classification report
    return accuracy, classification_report(true_labels, predictions, zero_division=0)

    
if __name__ == "__main__":
    print(f'[Testing Status]: Building datasets and dataloaders...')
    print(f'[Testing Status]: Building train dataloader...')
    # Create Embedding Dataset
    train_dataset = EmbeddingDataset(data_path=DATA_PATH,
                                        subset='TRAIN',
                                        id_column_idx=ID_COLUMN_IDX,
                                        comment_column_idx=COMMENT_COLUMN_IDX,
                                        label_column_idx=LABEL_COLUMN_IDX,
                                        subset_column_idx=SUBSET_COLUMN_IDX,
                                        augmented_classes=AUGMENTED_CLASSES,
                                        augmentation_ratio=0,
                                        augmentation_methods=AUGMENTATION_METHODS,
                                        adversation_ratio = ADVERSATION_RATIO,
                                        undersampling_targets=UNDERSAMPLING_TARGETS,
                                        embedder=Embedder(), 
                                        embedding_method=EMBEDDING_METHOD)
    
    # Get the DataLoader with embeddings
    # Note the multiple objects outputted here
    train_data_package = get_dataloader(train_dataset,  
                                batch_size=BATCH_SIZE,
                                shuffle=False, 
                                num_workers=2)
    
    print(f'[Testing Status]: Building test dataloader...')
    test_dataset = EmbeddingDataset(data_path=DATA_PATH,
                                        subset='TEST',
                                        id_column_idx=ID_COLUMN_IDX,
                                        comment_column_idx=COMMENT_COLUMN_IDX,
                                        label_column_idx=LABEL_COLUMN_IDX,
                                        subset_column_idx=SUBSET_COLUMN_IDX,
                                        augmented_classes=[],
                                        augmentation_ratio=0,
                                        augmentation_methods=[],
                                        adversation_ratio = 0,
                                        undersampling_targets={},
                                        embedder=Embedder(), 
                                        embedding_method=EMBEDDING_METHOD)
    
    # Get the DataLoader with embeddings
    # Note the multiple objects outputted here
    test_data_package = get_dataloader(test_dataset,  
                                batch_size=BATCH_SIZE,
                                shuffle=False, 
                                num_workers=2)
    
    # Choose a model
    print(f'[Testing Status]: Fitting a classifier...')
    model_config = MODEL_CONFIG.get(MODEL_TYPE)

    # Initialize and train the model
    classifier = Classifier(model_config, 
                            model_type=MODEL_TYPE,
                            log=False)
    classifier.fit(train_data_package)

    # Test the model
    print(f'[Testing Status]: Testing on test subset...')
    predictions = classifier.predict(test_data_package)

    # Show accuracy score per class + macro (classification report)
    # Calculate accuracy and show classification report
    accuracy, report = calculate_accuracy(predictions, test_data_package)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:")
    print(report)
