# Classifier __init__ configurations

MODEL_TYPE = 'dnn'  # Choose from [dnn, logistic_regression, svm, xgboost]
MODEL_CONFIG = {
    "logistic_regression": {
        "solver": "liblinear",
        "max_iter": 100
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
        "num_epochs": 10,  # Adjust after trial and error
        "learning_rate": 1e-3,
        "batch_norm": True,
        "drop_out": 0.5,
        "layers": [768, 128, 64, 3]  # Layer dimentions, including an input and an output layer.
    }
}