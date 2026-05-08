import pandas as pd

def load_and_clean_data(file_path):
    # سلام، هنا تضعين كود قراءة الـ CSV وتنظيفه
    print("🧹 Cleaning data...")
    df = pd.read_csv(file_path)
    # أضيفي خطوات التنظيف هنا
    return df
