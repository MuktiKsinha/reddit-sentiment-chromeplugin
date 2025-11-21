import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
import pandas as pd
import pickle
import logging
import yaml
import mlflow
from mlflow.sklearn import log_model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import os
import matplotlib.pyplot as plt
try:
    import seaborn as sns  # type: ignore
except Exception:
    sns = None
import json
from mlflow.models import infer_signature
from scipy.sparse import issparse, csr_matrix, hstack
import sys

# When running the script directly (python src/model/model_evaluation.py),
# the `src` package isn't on sys.path. Add the repository `src/` to path
# so we can import the feature extractor used during training.
_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
_SRC_DIR = os.path.join(_ROOT_DIR, 'src')
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

try:
    from features.feature_engineering import extract_custom_features_batch
except Exception as _e:
    extract_custom_features_batch = None
    # logger may not be configured yet; print a warning to stderr
    print(f"Warning: could not import extract_custom_features_batch: {_e}", file=sys.stderr)
from src.features.feature_engineering import extract_custom_features_batch


logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('model_building_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


###############################
# Helper Functions
###############################

def get_root_directory() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../../'))


def path_processed(filename: str) -> str:
    return os.path.join(get_root_directory(), 'data/processed', filename)


def model_path(filename: str) -> str:
    return os.path.join(get_root_directory(), 'models/', filename)


def file_path(filename: str) -> str:
    return os.path.join(get_root_directory(), 'data/interim', filename)


def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        df.fillna('', inplace=True)
        logger.debug('Data loaded and NaNs filled from %s', path)
        return df
    except Exception as e:
        logger.error('Error loading test dataset: %s', e)
        raise


def load_model(path: str):
    try:
        with open(path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', path)
        return model
    except Exception as e:
        logger.error('Error loading model from %s: %s', path, e)
        raise


def load_vectorizer(path: str) -> CountVectorizer:
    try:
        with open(path, 'rb') as file:
            vectorizer = pickle.load(file)
        logger.debug('Vectorizer loaded from %s', path)
        return vectorizer
    except Exception as e:
        logger.error('Error loading vectorizer from %s: %s', path, e)
        raise


def load_params(root: str) -> dict:
    try:
        with open(os.path.join(root, "params.yaml"), "r") as f:
            params = yaml.safe_load(f)

        if not isinstance(params, dict):
            raise ValueError("params.yaml loaded as string — invalid YAML formatting")

        model_params = params.get("model_building", {})
        
        # handle nested case
        if "lgbm_params" in model_params:
            return model_params["lgbm_params"]
        
        return model_params

    except Exception as e:
        logger.error('Failed to load params.yaml: %s', e)
        raise


def evaluate_model(model, X_test, y_test):
    try:
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        return report, cm
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise


def log_confusion_matrix(cm, dataset_name):
    plt.figure(figsize=(8, 6))
    if sns is not None:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    else:
        plt.imshow(cm, cmap='Blues')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='white')

    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Ensure reports directory exists and save there
    reports_dir = os.path.join(get_root_directory(), 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    cm_filename = f'confusion_matrix_{dataset_name}.png'
    cm_file_path = os.path.join(reports_dir, cm_filename)
    plt.savefig(cm_file_path)
    mlflow.log_artifact(cm_file_path)
    plt.close()


def save_model_info(run_id: str, model_path: str, file_path: str):
    try:
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug('Model info saved to %s', file_path)
    except Exception as e:
        logger.error('Error saving model info: %s', e)
        raise


######################################
#        MAIN EVALUATION PIPELINE
######################################

def main():
    try:
        root = get_root_directory()
        models_dir = os.path.join(root, "models")
        vectorizer_dir = os.path.join(root, "data", "processed")
        interim_dir = os.path.join(root, "data", "interim")

        # Load yaml params
        params = load_params(root)

        mlflow.set_experiment("reddit_sentiment_evaluation")

        with mlflow.start_run() as run:

            # Log params
            for key, value in params.items():
                mlflow.log_param(key, value)

            model_file = os.path.join(models_dir, 'lgbm_reddit_sentiment.pkl')
            # bow_vectorizer.pkl is the fitted CountVectorizer saved in data/processed/
            vectorizer_file = os.path.join(vectorizer_dir, 'bow_vectorizer.pkl')
            # X_train_features.pkl will be loaded later as reference for feature dimensions
            features_file = os.path.join(vectorizer_dir, 'X_train_BOW_custom.pkl')
            test_file = os.path.join(interim_dir, 'test_processed.csv')

            # Debug: log paths to verify they exist
            logger.info(f"Model file path: {model_file}")
            logger.info(f"Vectorizer file path: {vectorizer_file}")
            logger.info(f"Features file path: {features_file}")
            logger.info(f"Test file path: {test_file}")

            # Load model + vectorizer
            model = load_model(model_file)
            vectorizer = load_vectorizer(vectorizer_file)

            # Load test data
            test_df = load_data(test_file)

            # Validate columns
            if 'clean_comment' not in test_df or 'category' not in test_df:
                raise KeyError("test_processed.csv must contain 'clean_comment' and 'category' columns")

            # Transform test comments using the fitted vectorizer (BOW features)
            X_bow = vectorizer.transform(test_df["clean_comment"].values)
            
            # Load pre-computed features (which include both BOW and custom features from training)
            # For evaluation, we need to extract custom features for test set and combine
            with open(features_file, 'rb') as f:
                X_train_features_ref = pickle.load(f)
            
            # Get the number of custom feature columns from training
            n_bow_features = X_bow.shape[1]
            n_total_features = X_train_features_ref.shape[1]
            n_custom_features = n_total_features - n_bow_features
            
            logger.info(f"BOW features: {n_bow_features}, Custom features: {n_custom_features}")
            
            # Extract custom features for the test set using the same function
            # used during training so feature names/ordering match.
            # Use the same extractor used during training when available
            if callable(extract_custom_features_batch):
                try:
                    custom_df_test = extract_custom_features_batch(test_df["clean_comment"].tolist(), batch_size=64)
                except Exception as e:
                    logger.error("Failed to extract custom features for test set: %s", e)
                    custom_df_test = pd.DataFrame(
                        np.zeros((X_bow.shape[0], n_custom_features), dtype=np.float32),
                        columns=[f"custom_feature_{i}" for i in range(n_custom_features)]
                    )
            else:
                logger.warning("Custom feature extractor unavailable; using zero-filled placeholders.")
                custom_df_test = pd.DataFrame(
                    np.zeros((X_bow.shape[0], n_custom_features), dtype=np.float32),
                    columns=[f"custom_feature_{i}" for i in range(n_custom_features)]
                )

            # Ensure column count matches training custom feature count
            custom_np = custom_df_test.fillna(0).astype(np.float32).values
            if custom_np.shape[1] != n_custom_features:
                logger.warning(
                    "Custom feature dimension mismatch: training had %d, test produced %d. "
                    "Truncating or padding with zeros to match.",
                    n_custom_features,
                    custom_np.shape[1],
                )
                if custom_np.shape[1] > n_custom_features:
                    custom_np = custom_np[:, :n_custom_features]
                else:
                    # pad with zeros
                    pad_width = n_custom_features - custom_np.shape[1]
                    custom_np = np.hstack([custom_np, np.zeros((custom_np.shape[0], pad_width), dtype=np.float32)])

            X_custom_test = csr_matrix(custom_np)
            
            # Combine BOW + custom features
            X_test = hstack([X_bow, X_custom_test]).tocsr()
            y_test = test_df["category"].values

            # Create feature names: BOW features + custom feature placeholders
            bow_feature_names = vectorizer.get_feature_names_out()
            custom_feature_names = [f"custom_feature_{i}" for i in range(n_custom_features)]
            all_feature_names = list(bow_feature_names) + custom_feature_names
            
            # Convert to DataFrame with proper feature names for model predictions
            X_test_df = pd.DataFrame(X_test.toarray(), columns=all_feature_names)

            # Create example for MLflow
            # Safely obtain number of samples; some analyzers warn if shape could be None
            _shape = getattr(X_test_df, "shape", None)
            if _shape is not None:
                n_rows = _shape[0]
            else:
                n_rows = 0
            sample_n = min(5, n_rows)

            input_example = X_test_df.iloc[:sample_n]
                    
            preds = model.predict(X_test_df.iloc[:sample_n])
            signature = infer_signature(input_example, preds)

            # Log model
            log_model(
                model,
                artifact_path="model",
                signature=signature,
                input_example=input_example
            )

            # Save model info
            model_info_path = os.path.join(models_dir, "model_info.json")
            save_model_info(run.info.run_id, model_file, model_info_path)
            mlflow.log_artifact(model_info_path)
            mlflow.log_artifact(vectorizer_file)


            # Evaluate
            report, cm = evaluate_model(model, X_test_df, y_test)

            # Ensure reports directory exists
            reports_dir = os.path.join(root, 'reports')
            os.makedirs(reports_dir, exist_ok=True)

            # Save classification report to JSON and log as artifact
            classification_report_path = os.path.join(reports_dir, 'classification_report.json')
            try:
                with open(classification_report_path, 'w') as f:
                    json.dump(report, f, indent=4)
                mlflow.log_artifact(classification_report_path)
            except Exception as e:
                logger.error('Failed to write classification report: %s', e)

            # Log metrics (flattens all labels)
            if isinstance(report, dict):
                for label, metrics_dict in report.items():
                    if isinstance(metrics_dict, dict):
                        for m_name, m_value in metrics_dict.items():
                            if m_name != "support":
                                mlflow.log_metric(f"test_{label}_{m_name}", float(m_value))

            # Log confusion matrix (saved inside reports/ by the function)
            log_confusion_matrix(cm, "test_set")

            mlflow.set_tag("model_type", "LightGBM")
            mlflow.set_tag("task", "Sentiment Analysis")

            logger.info("Model evaluation completed successfully. Run ID: %s", run.info.run_id)

    except Exception as e:
        logger.exception("Evaluation failed: %s", e)
        raise


if __name__ == '__main__':
    main()

