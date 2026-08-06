class RiskAgent:

    def assess_risk(self, prediction, amount):

        if prediction == 1:

            if amount >= 10000:
                return "🔴 HIGH RISK"

            elif amount >= 5000:
                return "🟠 MEDIUM RISK"

            else:
                return "🟡 LOW RISK"

        return "🟢 SAFE"