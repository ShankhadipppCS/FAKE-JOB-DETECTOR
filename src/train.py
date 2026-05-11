import os
import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from src.utils import clean_text

DATA_PATH = "data/fake_job_postings.csv"
MODEL_DIR = "models"


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df[['description', 'fraudulent']].dropna()
    df['description'] = df['description'].apply(clean_text)
    return df


def train():
    df = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        df['description'],
        df['fraudulent'],
        test_size=0.2,
        random_state=42
    )

    vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    print("\nModel Evaluation:\n")
    print(classification_report(y_test, y_pred))

    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(f"{MODEL_DIR}/model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(f"{MODEL_DIR}/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print("\nModel and vectorizer saved!")


if __name__ == "__main__":
    train()