"""
evaluation.py
=============
Evaluation Pipeline for Air Pollution Risk Classification
 computes: Accuracy, Precision, Recall, F1-score

 الملف: src/evaluation.py
"""

try:
    import sklearn.metrics
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        classification_report,
        confusion_matrix
    )
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as e:
    raise ImportError(f"Required packages not found. Please install: scikit-learn, matplotlib, seaborn. Error: {e}")


def calculate_metrics(y_true, y_pred, labels=None):
    """
    تحسب Accuracy, Precision, Recall, F1-score
    
    Parameters:
        y_true: القيم الحقيقية (من test set)
        y_pred: القيم المتوقعة من النموذج
        labels: اسماء الفئات ['Low', 'Medium', 'High']
    
    Returns:
        dict: قاموس يحتوي الأربعة مقاييس
    """
    
    if labels is None:
        labels = ['Low', 'Medium', 'High']
    
    # 1. ACCURACY - الدقة العامة
    accuracy = accuracy_score(y_true, y_pred)
    
    # 2. PRECISION - الدقة الموزونة
    precision = precision_score(y_true, y_pred, average='weighted', labels=labels)
    
    # 3. RECALL - الاسترجاع الموزون
    recall = recall_score(y_true, y_pred, average='weighted', labels=labels)
    
    # 4. F1-SCORE - المتوسط التوافقي
    f1 = f1_score(y_true, y_pred, average='weighted', labels=labels)
    
    metrics = {
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4)
    }
    
    return metrics


def print_classification_report(y_true, y_pred, labels=None):
    """
    تطبع تقرير مفصل لكل فئة على حدة
    """
    
    # الحصول على الفئات الفعلية من البيانات
    unique_labels = sorted(list(set(list(y_true) + list(y_pred))))
    
    print("\n" + "=" * 60)
    print("📋 CLASSIFICATION REPORT")
    print("=" * 60)
    
    report = classification_report(
        y_true, 
        y_pred, 
        labels=unique_labels,
        digits=4
    )
    print(report)
    
    return report


def plot_confusion_matrix(y_true, y_pred, labels=None, save_path=None):
    """
    ترسم مصفوفة الالتباس (Confusion Matrix)
    """
    
    # الحصول على الفئات الفعلية من البيانات
    unique_labels = sorted(list(set(list(y_true) + list(y_pred))))
    
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=unique_labels,
        yticklabels=unique_labels,
        cbar_kws={'label': 'Count'}
    )
    
    plt.title('Confusion Matrix - Air Pollution Risk', fontsize=14, pad=20)
    plt.xlabel('Predicted (التوقع)', fontsize=12)
    plt.ylabel('Actual (الحقيقة)', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n💾 Saved to: {save_path}")
    
    plt.show()
    
    return cm


def evaluate_performance(model, X_test, y_test, labels=None):
    """
    الدالة الرئيسية للتقييم - تقيم أداء النموذج بشكل كامل
    
    Parameters:
        model: النموذج المدرب
        X_test: بيانات الاختبار (Features)
        y_test: القيم الحقيقية (Target)
        labels: اسماء الفئات (اختياري)
    """
    
    if labels is None:
        labels = ['Low', 'Medium', 'High']
    
    # التنبؤ على بيانات الاختبار
    y_pred = model.predict(X_test)
    
    # حساب المقاييس
    metrics = calculate_metrics(y_test, y_pred, labels)
    
    print("\n" + "=" * 60)
    print("📊 MODEL PERFORMANCE METRICS")
    print("=" * 60)
    print(f"🎯 Accuracy:  {metrics['accuracy']}")
    print(f"📈 Precision: {metrics['precision']}")
    print(f"📍 Recall:    {metrics['recall']}")
    print(f"⚖️  F1-Score:  {metrics['f1_score']}")
    print("=" * 60)
    
    # تقرير مفصل
    print_classification_report(y_test, y_pred, labels)
    
    # مصفوفة الالتباس
    plot_confusion_matrix(y_test, y_pred, labels, save_path='confusion_matrix.png')
    
    return metrics


# ============================================================
# MAIN - للاختبار المحلي فقط
# ============================================================

if __name__ == "__main__":
    
    print("=" * 60)
    print("🧪 TESTING evaluation.py")
    print("=" * 60)
    
    # بيانات وهمية للاختبار
    y_true_test = ['High', 'Medium', 'Low', 'High', 'Medium', 
                   'Low', 'High', 'Medium', 'Low', 'High']
    
    y_pred_test = ['High', 'Medium', 'Low', 'High', 'Medium', 
                   'Low', 'Medium', 'Medium', 'Low', 'High']
    
    # اختبار
    results = calculate_metrics(y_true_test, y_pred_test)
    print("Results:", results)
    
    print_classification_report(y_true_test, y_pred_test)
    
    plot_confusion_matrix(y_true_test, y_pred_test, save_path='test_confusion_matrix.png')
    
    print("\n✅ evaluation.py is ready!")