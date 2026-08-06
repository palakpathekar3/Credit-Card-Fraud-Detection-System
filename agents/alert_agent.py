from datetime import datetime


class AlertAgent:

    def generate_alert(self, prediction, risk):

        if prediction == 1:

            return {
                "status": "ALERT",
                "risk": risk,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "message": "🚨 Fraudulent transaction detected! Immediate review required."
            }

        return {
            "status": "SAFE",
            "risk": risk,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "message": "✅ Transaction appears legitimate."
        }