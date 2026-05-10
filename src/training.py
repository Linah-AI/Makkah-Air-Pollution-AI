from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

def train_model(df):
    """
    Encodes categorical features and trains the Decision Tree model 
    on the complete dataset for maximum pattern extraction.
    """
    # 1. Feature Encoding
    le_pollution = LabelEncoder()
    df["أنواع التلوث الهوائي"] = le_pollution.fit_transform(df["أنواع التلوث الهوائي"])

    # 2. Define Features (X) and Target (y)
    X = df[["السنة", "أنواع التلوث الهوائي"]]
    y = df["Risk"]

    # 3. Model Initialization and Training
    # Training on the entire dataset to build a comprehensive tree
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)

    print("Decision Tree model trained successfully on the full dataset.")

    # 4. Return model and full data arrays for Cross-Validation
    return model, X, y