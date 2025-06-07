# Classifier __init__ configurations

# Get the project root directory (move up one level from the 'Config' folder)
import os
import torch
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINTS = os.path.join(project_root, 'Checkpoint')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_TYPE = 'xgboost'  # Choose from [dnn, logistic_regression, svm, xgboost]
MODEL_CONFIG = {
    "logistic_regression": {
        "num_epochs": 20,
        "learning_rate": 0.0006763931659185851, 
        "weight_decay": 9.239168299828083e-05,
        "batch_norm": False,    # Do not modify in optimization
        "drop_out": 0.0,    # Do not modify in optimization
        "layers": "[768, 3]"  # Do not modify in optimization
    },
    "svm": {
        'C': 14.113323810506161, 
        'kernel': 'rbf', 
        'degree': 4, 
        'gamma': 'scale'
    },
    "xgboost": {
        'n_estimators': 97, 
        'learning_rate': 0.19893565401854227, 
        'booster': 'gbtree', 
        'max_depth': 7, 
        'min_child_weight': 3, 
        'colsample_bytree': 0.698884472870104, 
        'subsample': 0.5802363751849828, 
        'reg_alpha': 1.1264120790978405e-07, 
        'reg_lambda': 0.7242980309966568, 
        'gamma': 1.0153525598898357e-05, 
        'grow_policy': 'depthwise'
    },
    "dnn": {
        'num_epochs': 30, 
        'learning_rate': 3e-4, 
        'weight_decay': 1e-4, 
        'batch_norm': True, 
        'drop_out': 0.2, 
        'layers': "[768, 256, 3]"
    }
}