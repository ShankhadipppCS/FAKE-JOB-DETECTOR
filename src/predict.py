import pickle
from src.utils import clean_text, get_reasons

MODEL_PATH = "models/model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"


class FakeJobDetector:
    def __init__(self):
        self.model = pickle.load(open(MODEL_PATH, "rb"))
        self.vectorizer = pickle.load(open(VECTORIZER_PATH, "rb"))

    def predict(self, text: str):
        cleaned = clean_text(text)
        X = self.vectorizer.transform([cleaned])

        pred = self.model.predict(X)[0]
        prob = self.model.predict_proba(X)[0].max()

        label = "Fake" if pred == 1 else "Real"
        reasons = get_reasons(text)

        return {
            "label": label,
            "confidence": round(prob, 2),
            "reasons": reasons
        }