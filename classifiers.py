'''
Class to contain the different model, each should be easiliy called for initialization,
fit and predict.
'''

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Local Code
from Config.classifiers_config import *
from Config.dataset_config import *
from dataset import TextDataset, get_dataloader


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
    def __init__(self, config, model_type):
        """
        Initializes the classifier based on the model type and configuration.

        Args:
            config (dict): Configuration for the model. Format varies based on the model type.
            model_type (str): One of "logistic_regression", "svm", "xgboost", or "dnn".
        """
        self.model_type = model_type

        if model_type == "logistic_regression":
            self.model = LogisticRegression(**config)
        elif model_type == "svm":
            self.model = SVC(**config)
        elif model_type == "xgboost":
            self.model = XGBClassifier(**config)
        elif model_type == "dnn":
            self.model = DNN(config)
            self.optimizer = optim.Adam(self.model.parameters(), lr=config["learning_rate"])
            self.criterion = nn.CrossEntropyLoss()  # Cross-Entropy Loss for multiclass problem
            self.num_epochs = config["num_epochs"]
        else:
            raise ValueError(f"Unsupported model type: {model_type}")


    def fit(self, train_loader):
        """
        Fits the model to the training data.

        Args:
            train_loader (DataLoader): DataLoader providing training data.
        """
        if self.model_type in ["logistic_regression", "svm", "xgboost"]:
            X, y = [], []
            for batch in train_loader:
                features, labels = batch
                X.append(features)
                y.append(labels)
            X = torch.cat(X).numpy()
            y = torch.cat(y).numpy()
            self.model.fit(X, y)
        elif self.model_type == "dnn":
            self.model.train()
            for epoch in range(self.num_epochs):
                for _, (features, labels) in enumerate(train_loader):
                    self.optimizer.zero_grad()
                    outputs = self.model(features.float())
                    loss = self.criterion(outputs.squeeze(), labels.float())
                    loss.backward()
                    self.optimizer.step()
                print(f"Epoch {epoch + 1}: Loss = {loss.item()}")


    def predict(self, test_loader):
        """
        Predicts labels for the given test data.

        Args:
            test_loader (DataLoader): DataLoader providing test data.

        Returns:
            list: Predicted labels.
        """
        predictions = []
        if self.model_type in ["logistic_regression", "svm", "xgboost"]:
            X = []
            for batch in test_loader:
                comment_ids, features, labels = batch
                X.append(features)
            X = torch.cat(X).numpy()
            predictions = self.model.predict(X)
        elif self.model_type == "dnn":
            self.model.eval()
            with torch.no_grad():
                for _, (features, _) in enumerate(test_loader):
                    outputs = self.model(features.float())  # outputs.shape should be (batch_size, 3)
                    _, predictions_batch = torch.max(outputs, 1)  # Get the index of the max probability for each sample
                    predictions.extend(predictions_batch.int().tolist())  # Append predictions
        return predictions
    
    
if __name__ == "__main__":
    # Initialize Dataset
    dataset = TextDataset(
        data_path=DATA_PATH,
        subset=SUBSET,
        id_column_idx=ID_COLUMN_IDX,
        comment_column_idx=COMMENT_COLUMN_IDX,
        label_column_idx=LABEL_COLUMN_IDX,
        subset_column_idx=SUBSET_COLUMN_IDX,
        augmented_classes=AUGMENTED_CLASSES,
        augmentation_ratio=AUGMENTATION_RATIO,
        adversation_ratio=ADVERSATION_RATIO
    )
    
    # Create Dataloader
    dataloader = get_dataloader(dataset, 
                                datashape=DATALOADER_SHAPE, 
                                embedding_file_path=EMBEDDING_PATH, 
                                batch_size=BATCH_SIZE, 
                                shuffle=True, 
                                num_workers=2)
    
    # Choose a model
    model_config = MODEL_CONFIG.get(MODEL_TYPE)

    # Initialize and train the model
    classifier = Classifier(model_config, MODEL_TYPE)
    classifier.fit(dataloader)

    # # Test the model
    # test_dataset = TextClassificationDataset(...)  # Replace with test dataset parameters
    # test_loader = get_dataloader(test_dataset, datashape="embedding")
    # predictions = classifier.predict(test_loader)

    # print(predictions)