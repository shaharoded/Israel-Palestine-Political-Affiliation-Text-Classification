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
        "learning_rate": 0.001,
        "batch_norm": True,
        "drop_out": 0.5,
        "layers": [100, 50, 25]
    }
}