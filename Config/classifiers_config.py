# Classifier __init__ configurations

MODEL_TYPE = 'xgboost'  # Choose from [dnn, logistic_regression, svm, xgboost]
MODEL_CONFIG = {
    "logistic_regression": {
        "num_epochs": 17,
        "learning_rate": 2.1e-4 ,
        "weight_decay": 8.3e-5,  
        "batch_norm": False,    # Do not modify in optimization
        "drop_out": 0.0,    # Do not modify in optimization
        "layers": [768, 3]  # Do not modify in optimization
    },
    "svm": {
        'C': 0.038324376490610844, 
        'kernel': 'rbf', 
        'degree': 4, 
        'gamma': 'scale'
    },
    "xgboost": {
        'n_estimators': 53, 
        'learning_rate': 0.0053071977040925495, 
        'booster': 'gblinear', 
        'reg_alpha': 6.544161671352116e-08, 
        'reg_lambda': 6.030421588983791e-06
    },
    "dnn": {
        "num_epochs": 20,  # Adjust after trial and error
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_norm": True,
        "drop_out": 0.2,
        "layers": [768, 512, 256, 128, 64, 3]  # Layer dimentions, including an input and an output layer.
    }
}