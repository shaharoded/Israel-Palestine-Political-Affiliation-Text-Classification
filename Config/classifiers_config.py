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
        "num_epochs": 18,
        "learning_rate": 0.001, 
        "weight_decay": 3.925e-05,
        "batch_norm": False,    # Do not modify in optimization
        "drop_out": 0.0,    # Do not modify in optimization
        "layers": "[768, 3]"  # Do not modify in optimization
    },
    "svm": {
        'C': 0.01, 
        'kernel': 'rbf', 
        'degree': 2, 
        'gamma': 'scale'
    },
    "xgboost": {
        'n_estimators': 132, 
        'learning_rate': 0.11101741103143277, 
        'booster': 'gbtree', 
        'max_depth': 9, 
        'min_child_weight': 6, 
        'colsample_bytree': 0.8129944223404375, 
        'subsample': 0.736423234099351, 
        'reg_alpha': 0.0006658328205950035, 
        'reg_lambda': 1.6465814969612817e-05, 
        'gamma': 0.0010927884898282638, 
        'grow_policy': 'depthwise'
    },
    "dnn": {
        'num_epochs': 18, 
        'learning_rate': 3e-4, 
        'weight_decay': 1e-4, 
        'batch_norm': True, 
        'drop_out': 0.2, 
        'layers': "[768, 256, 3]"
    }
}