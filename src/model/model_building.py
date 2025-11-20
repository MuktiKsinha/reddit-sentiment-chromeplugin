# src/model/model_building.py

import os
import pickle
import logging
import yaml
import lightgbm as lgb
import numpy as np


logger = logging.getLogger('model_building')
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


def get_root_directory() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../../'))

def path_processed(filename: str) -> str:
    return os.path.join(get_root_directory(), 'data/processed', filename)


def load_params():
    root = get_root_directory()
    try:
        with open(os.path.join(root, "params.yaml"), "r") as f:
            params = yaml.safe_load(f)
        return params["model_building"]
    except Exception as e:
        logger.error(f'Failed to load params.yaml:{e}')



def load_features():
    """Load saved Bow transformed X_train and y_train."""
    try:
        with open(os.path.join(path_processed( "X_train_features.pkl")), "rb") as f:
            X = pickle.load(f)

        with open(os.path.join(path_processed("y_train.pkl")), "rb") as f:
            y = pickle.load(f)
    except Exception as e:
        logger.error(f"Feature files missing: {e}")


    logger.debug(f"Loaded engineered features: X={X.shape},y={y.shape}")
    return X, y


def train_lgbm(X, y, params):
    lgbm_params = params["lgbm_params"]
    
    model = lgb.LGBMClassifier(**lgbm_params)
    model.fit(X, y)
    
    logger.debug("LGBM training complete.")
    return model


def save_model(model):
    model_path = os.path.join(get_root_directory(), "models", "lgbm_reddit_sentiment.pkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    with open(model_path,"wb") as f:
        pickle.dump(model, f)
    
    logger.debug(f"Model saved: {model_path}")


def main():
    try:
        params = load_params()
        X, y = load_features()
        model = train_lgbm(X, y, params)
        save_model(model)

    except Exception as e:
        logger.error(f"Model building failed: {e}")
        raise


if __name__ == "__main__":
    main()
