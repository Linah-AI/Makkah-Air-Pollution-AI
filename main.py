import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from src import preprocessing, training, evaluation

def main():
    print("🚀 بدء تشغيل نظام التنبؤ بتلوث الهواء في مكة...")
    print("-" * 50)

    # 1. مرحلة تنظيف البيانات
    # ---------------------------------
    data_path = "data/makkah_pollution.xlsx"
    clean_df = preprocessing.load_and_clean_data(data_path)
    
    print("✅ تم تنظيف البيانات بنجاح!")
    print("\n📊 عينة من الجداول اللي نظفتها سلام (أول 5 صفوف):")
    print(clean_df.head())
    print("-" * 50)

    # 2. مرحلة تدريب النموذج
    # ---------------------------------
    model, X_test, y_test = training.train_model(clean_df)
    print("✅ تم تدريب المودل (الذكاء الاصطناعي) بنجاح!")
    print(f"🧠 المودل الآن خبير وعنده {model.get_n_leaves()} قاعدة استنتاجية.")
    print("-" * 50)

    # 3. مرحلة الاستعراض البصري
    # ---------------------------------
    print("🌳 جاري رسم شجرة القرار (Decision Tree)...")
    feature_names = X_test.columns.tolist() if hasattr(X_test, "columns") else ["السنة", "أنواع التلوث الهوائي"]
    plt.figure(figsize=(20, 10))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=model.classes_,
        filled=True,
        rounded=True,
        fontsize=10,
    )
    plt.title("Makkah Pollution Decision Tree Model")
    print("💡 ستفتح الآن نافذة جديدة تحتوي على رسمة الشجرة.. أغلقيها للاستمرار.")
    plt.show()

    # 4. مرحلة التقييم
    # ----------------------------
    print("📈 جاري التقييم ورسم النتائج (شغل خديجة)...")
    
    # أولاً: نخلي المودل يحل الاختبار عشان نرسل إجاباته لخديجة
    y_pred = model.predict(X_test)
    
    # ثانياً: نستخدم دوال خديجة الاحترافية
    # 1. طباعة الدقة وباقي المقاييس
    metrics = evaluation.calculate_metrics(y_test, y_pred, labels=model.classes_)
    print("\n🎯 النتائج النهائية:")
    for key, value in metrics.items():
        print(f" - {key.upper()}: {value * 100:.2f}%")
        
    # 2. طباعة التقرير المفصل
    evaluation.print_classification_report(y_test, y_pred, labels=model.classes_)
    
    # 3. رسم مصفوفة الالتباس (Confusion Matrix)
    print("💡 ستفتح الآن نافذة جديدة تحتوي على مصفوفة الالتباس..")
    evaluation.plot_confusion_matrix(y_test, y_pred, labels=model.classes_)

if __name__ == "__main__":
    main()
