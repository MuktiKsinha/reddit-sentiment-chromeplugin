import os
import pickle
import logging
import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
import yaml
import spacy

# Load spaCy model ONCE globally
nlp = spacy.load("en_core_web_sm")

# logging configuration
logger = logging.getLogger('Feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('Feature_engineering_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def get_root_directory() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../../'))

def path_interim(filename: str) -> str:
    return os.path.join(get_root_directory(), 'data/interim', filename)

def path_processed(filename: str) -> str:
    return os.path.join(get_root_directory(), 'data/processed', filename)

def load_params(param_path : str) ->dict:
    """Load parameter from yaml file"""
    try:
        with open (param_path,'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s',param_path)
        return params
    except FileNotFoundError:
        logger.error('File not found %s',param_path)
        raise
    except yaml.YAMLError as e:
        logger.error('Yaml error %s',e)
        raise
    except Exception as e:
        logger.error('Unexpected error %s',e)
        raise 

def load_data (file_path: str)->pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('',inplace=True) # Fill any NaN values
        logger.debug('Data loaded and  %s',file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to Parse the csv file: %s',e)
        raise
    except Exception as e:
        logger.error('Unexpected error occured while loading the data : %s',e)
        raise
# ---------------------------------------------------------
# CUSTOM FEATURES
# ---------------------------------------------------------
def extract_custom_features_batch(text_list,batch_size):
    results = []

    try:
        # nlp.pipe is MUCH faster than individual calls
        for doc in nlp.pipe(text_list, batch_size=batch_size):
            text = doc.text
            word_list = [token.text for token in doc]
            word_count = len(word_list)
            unique_words = len(set(word_list))
            pos_tags = [token.pos_ for token in doc]

            # Base features
            features = {
                "comment_length": len(text),
                "word_count": word_count,
                "avg_word_length": (
                    sum(len(w) for w in word_list) / word_count
                    if word_count else 0
                ),
                "unique_word_count": unique_words,
                "lexical_diversity": (
                    unique_words / word_count if word_count else 0
                ),
                "pos_count": len(pos_tags),
            }

            # POS proportions
            if word_count > 0:
                for tag in set(pos_tags):
                    features[f"pos_ratio_{tag}"] = pos_tags.count(tag) / word_count

            results.append(features)
        return pd.DataFrame(results)
    except Exception as e:
        logger.critical("Critical failure in extract_custom_features_batch: %s",e)
        raise

# ---------------------------------------------------------
# BOW
# ---------------------------------------------------------

def apply_bow(train_data:pd.DataFrame,max_features:int,ngram_range:tuple)->tuple:
    """Apply BOW with ngrams to the data."""
    try:
        vectorizer =CountVectorizer(ngram_range=ngram_range,max_features=max_features)

        X_train = train_data['clean_comment'].values
        y_train = train_data['category'].values

        # Perform BOW transformation
        X_train_bow = vectorizer.fit_transform(X_train)
        logger.debug(f'Bow transformation complete.Train shape:{X_train.shape}')

        #save the vectorizer in root directory
        with open(os.path.join(path_processed('bow_vectorizer.pkl')),'wb') as f:
            pickle.dump(vectorizer,f)

        logger.debug('Bow applied with trigrams and data transformed')
        return X_train_bow,y_train
    except Exception as e:
        logger.error('Error during BOW transformation: %s', e)
        raise

# ---------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------

def main():
    try:
        logger.info("Starting Feature Engineering...")

        # Get root directory and resolve the path for params.yaml
        root_dir = get_root_directory()

        #load parameter from the root directory 
        params = load_params(os.path.join(root_dir, 'params.yaml'))
        max_features = params['feature_engineering']['max_features']
        ngram_range = tuple(params['feature_engineering']['ngram_range'])
        batch_size = params['feature_engineering']['custom_features_batch_size']



         # Load the preprocessed training data from the interim directory
        train_data = load_data(os.path.join(root_dir, 'data/interim/train_processed.csv'))

        # Apply BOW feature engineering on training data
        X_train_bow, y_train = apply_bow(train_data, max_features, ngram_range)

        #apply custom features 
        logger.info("Extracting custom features (spaCy)...")
        custom_df = extract_custom_features_batch(train_data["clean_comment"].tolist(),batch_size=batch_size)
        
        custom_df.to_csv(path_processed("custom_features_train.csv"), index=False)

        custom_np = custom_df.fillna(0).astype(np.float32).values
        X_custom_sparse = csr_matrix(custom_np)

        #combine bow+custom
        X_final = hstack([X_train_bow, X_custom_sparse])

        # ---------------------------
        # 4. Save final training data
        # ---------------------------
        y = y_train

        with open(path_processed("X_train_BOW_custom.pkl"), "wb") as f:
            pickle.dump(X_final, f)

        with open(path_processed("y_train.pkl"), "wb") as f:
            pickle.dump(y, f)

        logger.info("Feature Engineering Completed Successfully")

    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        raise


if __name__ == "__main__":
    main()

    






