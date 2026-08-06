from pathlib import Path
import pandas as pd


class LearningAgent:

    def __init__(self):

        BASE_DIR = Path(__file__).parent.parent

        self.log_file = BASE_DIR / "data" / "learning_log.csv"

    def save_transaction(self, amount, prediction, risk):

        from datetime import datetime

        record = pd.DataFrame([{
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Amount": amount,
            "Prediction": "Fraud" if prediction == 1 else "Legitimate",
            "Risk": risk
        }])

        if self.log_file.exists():
            record.to_csv(
                self.log_file,
                mode="a",
                header=False,
                index=False
            )
        else:
            record.to_csv(
                self.log_file,
                index=False
            )

        return "Transaction saved for future learning."