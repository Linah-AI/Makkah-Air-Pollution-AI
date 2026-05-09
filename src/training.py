from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def train_model(df):
    # 1. تحويل النصوص إلى أرقام (التشفير)
    le_pollution = LabelEncoder()
    
    # نحول أسماء التلوث لأرقام عشان تفهمها الشجرة
    df["أنواع التلوث الهوائي"] = le_pollution.fit_transform(df["أنواع التلوث الهوائي"])

    # 2. تحديد المدخلات (Features) والمخرجات (Target)
    # المدخلات: السنة ونوع التلوث
    X = df[["السنة", "أنواع التلوث الهوائي"]]
    
    # المخرجات: عمود الخطر اللي صنعته سلام
    y = df["Risk"]

    # 3. تقسيم البيانات (80% تدريب و 20% اختبار)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. إنشاء الموديل وتدريبه
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    print("✅ Decision Tree Model Trained Successfully!")

    # 5. إرجاع الموديل وبيانات الاختبار عشان تستخدمها خديجة
    return model, X_test, y_test
