# necessary libraries
import numpy as np
import pandas as pd
import mlflow
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# read the csv 
df = pd.read_csv('https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv')
df.head()

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df = df[~(df['clean_comment'].str.strip() == '')]

# ensure necessary nltk data is downloaded
nltk.download('stopwords')
nltk.download('wordnet')

# define the preprocessing function
def preprocess_comment(comment):
    # convert to lower case
    comment = comment.lower()

    # remove spaces
    comment = comment.strip()

    # remove newline chars
    comment = re.sub('\n', ' ', comment)

    # remove non-alphanumeric (keep ?,!,.)
    comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

    # remove stopwords but keep sentiment words
    stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
    comment = ' '.join([word for word in comment.split() if word.lower() not in stop_words])

    # lemmatize
    lemmatizer = WordNetLemmatizer()
    comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

    return comment


# Apply preprocessing
df['clean_comment'] = df['clean_comment'].apply(preprocess_comment)

# Apply Bag of Words
vectorizer = CountVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_comment']).toarray()
y = df['category']

# connect MLflow
mlflow.set_tracking_uri("http://ec2-51-21-245-249.eu-north-1.compute.amazonaws.com:5000/")
mlflow.set_experiment("RF baseline")

# Train–test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Start MLflow run
with mlflow.start_run() as run:

    # Run tags
    mlflow.set_tag("mlflow.runName", "RandomForest_Baseline_TrainTestSplit")
    mlflow.set_tag("experiment_type", "Baseline")
    mlflow.set_tag("model_type", "RandomForestClassifier")
    mlflow.set_tag("description", "Baseline RF using BoW")

    # Log vectorizer params
    mlflow.log_param("vectorizer_type", "CountVectorizer")
    mlflow.log_param("vectorizer_max_features", vectorizer.max_features)

    # RF params
    n_estimators = 150
    max_depth = 15
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # Train model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)

    # Classification report
    classification_rep = classification_report(y_test, y_pred, output_dict=True)

    for label, metrics in classification_rep.items():
        if isinstance(metrics, dict):
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"{label}_{metric_name}", value)

    # Confusion matrix plot
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

    # Log model
    mlflow.sklearn.log_model(model, "random_forest_model")

    # Log dataset
    df.to_csv("dataset.csv", index=False)
    mlflow.log_artifact("dataset.csv")

    print("Accuracy:", accuracy)
