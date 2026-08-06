import joblib
from pathlib import Path


class FraudAgent:

    def __init__(self):

        BASE_DIR = Path(__file__).parent.parent

        MODEL_PATH = BASE_DIR / "models" / "random_forest_model.pkl"

        self.model = joblib.load(MODEL_PATH)

    def predict(self, transaction):

        prediction = self.model.predict(transaction)

        return int(prediction[0])