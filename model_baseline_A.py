import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report


# Custom tokenizer (pickle-safe)
def custom_tokenizer(text):
    return text.split()


class TFIDFLogRegBaseline:
    def __init__(self,
                 max_features=200000,
                 ngram_range=(1, 2),
                 C=1.0):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.C = C
        self.pipeline = None

    #TF-IDF + Logistic Regression model
    def _build_pipeline(self):
        vectorizer = TfidfVectorizer(
            tokenizer=custom_tokenizer,  # tokenizer
            token_pattern=None,
            lowercase=True,
            max_features=self.max_features,
            ngram_range=self.ngram_range
        )

        clf = LogisticRegression(
            C=self.C,
            max_iter=300,
            n_jobs=-1
        )

        self.pipeline = Pipeline([
            ("tfidf", vectorizer),
            ("clf", clf)
        ])

    # Fit model
    def fit(self, X, y):
        self._build_pipeline()
        self.pipeline.fit(X, y)

    # -----------------------------
    # Predict labels
    def predict(self, X):
        return self.pipeline.predict(X)

    # Evaluate model
    def evaluate(self, X, y):
        pred = self.predict(X)
        return {
            "accuracy": accuracy_score(y, pred),
            "f1_macro": f1_score(y, pred, average="macro"),
            "report": classification_report(y, pred)
        }

    #Saving trained model
    def save(self, path):
        joblib.dump(self.pipeline, path)

    # Load a saved model
    @staticmethod
    def load(path):
        model = TFIDFLogRegBaseline()
        model.pipeline = joblib.load(path)
        return model
