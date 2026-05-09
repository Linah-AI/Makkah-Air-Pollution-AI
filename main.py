from src import preprocessing, training, evaluation

def main():
    print("🚀 بدء تشغيل النظام...")

    # 1. تنظيف البيانات (شغل سلام)
    data_path = "data/makkah_pollution.xlsx"
    clean_df = preprocessing.load_and_clean_data(data_path)
    print("✅ تم تنظيف البيانات بنجاح!")
    
    # 2. تدريب النموذج (شغل دانة)
    model, X_test, y_test = training.train_model(clean_df)
    print("✅ تم تدريب المودل ومستعد للتقييم!")
    
    # 3. التقييم (شغل خديجة)
    evaluation.evaluate_performance(model, X_test, y_test)

if __name__ == "__main__":
    main()
