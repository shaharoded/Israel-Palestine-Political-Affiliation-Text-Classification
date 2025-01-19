# Classifier __init__ configurations

MODEL_TYPE = 'logistic_regression'  # Choose from [dnn, logistic_regression, svm, xgboost]
MODEL_CONFIG = {
    "logistic_regression": {
        "num_epochs": 20,  # Adjust after trial and error
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_norm": False,    # Do not modify in optimization
        "drop_out": 0.0,    # Do not modify in optimization
        "layers": [768, 3]  # Do not modify in optimization
    },
    "svm": {
        "kernel": "linear",
        "C": 1.0
    },
    "xgboost": {
        "n_estimators": 100,
        "learning_rate": 0.1
    },
    "dnn": {
        "num_epochs": 20,  # Adjust after trial and error
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_norm": True,
        "drop_out": 0.5,
        "layers": [768, 128, 64, 3]  # Layer dimentions, including an input and an output layer.
    }
}