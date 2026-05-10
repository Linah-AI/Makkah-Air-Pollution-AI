import pandas as pd

def load_and_clean_data(file_path):
    """
    Loads dataset, handles missing values, and categorizes risk levels.
    """
    # Load Excel file
    df = pd.read_excel(file_path)
    
    # Print dataset columns for verification
    print("Dataset columns:")
    print(df.columns.tolist())

    # Drop rows with missing values
    df = df.dropna()

    # Filter out records where the respondent did not suffer from pollution
    # Note: Arabic strings are maintained here to match the dataset schema.
    df = df[df["عانت من التلوث"] != "لا يعاني"]

    def classify_risk(value):
        """
        Classifies numerical distribution into risk categories.
        """
        if value >= 10:
            return "High"
        elif value >= 5:
            return "Medium"
        else:
            return "Low"

    # Create target variable 'Risk' based on relative distribution
    df["Risk"] = df["التوزيع النسبي"].apply(classify_risk)

    return df