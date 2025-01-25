'''
The purpose of this module is to allow you to get the predictions 
for a new unseen data set (as test set), based on a complete trained model.

Results will be saved in classification_results.csv
'''
import pandas as pd

# Local Code
from Config.classifiers_config import *
from Config.dataset_config import *
from dataset import EmbeddingDataset, get_dataloader
from embedder import Embedder
from classifiers import Classifier


if __name__ == "__main__":
    print(f'[Testing Status]: Building datasets and dataloaders...')
    
    print(f'[Testing Status]: Building test dataloader...')
    test_dataset = EmbeddingDataset(data_path='./Preprocessed_Dataset.csv',
                                        subset=None,
                                        id_column_idx=0,
                                        comment_column_idx=3,
                                        label_column_idx=13,
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
    
    print(f'[Testing Status]: Building train dataloader...')
    # Create Embedding Dataset
    train_dataset = EmbeddingDataset(data_path=DATA_PATH,
                                        subset='TRAIN',
                                        id_column_idx=ID_COLUMN_IDX,
                                        comment_column_idx=COMMENT_COLUMN_IDX,
                                        label_column_idx=LABEL_COLUMN_IDX,
                                        subset_column_idx=SUBSET_COLUMN_IDX,
                                        augmented_classes=AUGMENTED_CLASSES,
                                        augmentation_ratio=AUGMENTATION_RATIO,
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
    
    # Choose a model
    print(f'[Testing Status]: Fitting a classifier...')
    model_config = MODEL_CONFIG.get(MODEL_TYPE)

    # Initialize and train the model
    classifier = Classifier(model_config, 
                            model_type=MODEL_TYPE,
                            log=False)
    classifier.fit(train_data_package)

    # Test the model
    print(f'[Generating Labels]:')
    predictions = classifier.predict(test_data_package)
    
        # Extract comment IDs and real labels from the test dataset
    test_comment_ids = test_dataset.comment_ids  # Assuming it's stored as an attribute
    real_labels = test_dataset.labels.numpy()    # Convert to numpy if stored as a tensor
    
    # Create a DataFrame for predictions
    results_df = pd.DataFrame({
        "Comment ID": test_comment_ids,
        "Real Label": real_labels,
        "Predicted Label": predictions
    })

    # Save the DataFrame to a CSV file
    results_csv_path = "classification_results.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"Results saved to {results_csv_path}")
