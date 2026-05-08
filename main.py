from src import preprocessing, training, evaluation

def main():
    # 1. تنظيف البيانات (سلام)
    data_path = "data/makkah_pollution.csv"
    clean_df = preprocessing.load_and_clean_data(data_path)
    
    # 2. تدريب النموذج (دانة)
    # model, x_test, y_test = training.train_model(clean_df)
    
    # 3. التقييم (خديجة)
    # evaluation.evaluate_performance(model, x_test, y_test)

if __name__ == "__main__":
    main()
