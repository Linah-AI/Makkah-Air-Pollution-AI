import pandas as pd
import sys

def load_and_clean_data(file_path):
    """
    Loads dataset, handles missing values, and categorizes risk levels.
    """
    # 1. إضافة الـ try-except هنا لحماية النظام من التعطل
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found! Please check the path.")
        sys.exit() # يوقف البرنامج بأمان بدل الانهيار المفاجئ
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit()

    # Print dataset columns for verification
    print("Dataset columns:")
    print(df.columns.tolist())

    # Drop rows with missing values
    df = df.dropna()

    # Filter out records
    df = df[df["عانت من التلوث"] != "لا يعاني"]

    def classify_risk(value):
        if value >= 10:
            return "High"
        elif value >= 5:
            return "Medium"
        else:
            return "Low"

    df["Risk"] = df["التوزيع النسبي"].apply(classify_risk)

    return df