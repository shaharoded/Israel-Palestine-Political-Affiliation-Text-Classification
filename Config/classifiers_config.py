# Classifier __init__ configurations

MODEL_TYPE = 'dnn'  # Choose from [dnn, logistic_regression, svm, xgboost]
MODEL_CONFIG = {
    "logistic_regression": {
        "num_epochs": 20,
        "learning_rate": 0.0006763931659185851, 
        "weight_decay": 9.239168299828083e-05,
        "batch_norm": False,    # Do not modify in optimization
        "drop_out": 0.0,    # Do not modify in optimization
        "layers": [768, 3]  # Do not modify in optimization
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
        'num_epochs': 16, 
        'learning_rate': 0.00014610022118293262, 
        'weight_decay': 0.0001396624345530416, 
        'batch_norm': True, 
        'drop_out': 0.30471334715503484, 
        'layers': [768, 512, 128, 64, 3]
    }
}