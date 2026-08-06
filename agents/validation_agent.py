class ValidationAgent:

    def validate(self, df):

        # Check missing values
        if df.isnull().sum().sum() > 0:
            return False, "Missing values detected."

        # Check Amount column
        if "Amount" in df.columns:
            if (df["Amount"] < 0).any():
                return False, "Negative amount detected."

        return True, "Validation Successful."