from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import cross_val_predict, LeaveOneOut
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    sns = None

def calculate_metrics(y_true, y_pred, labels=None):
    """
    Computes weighted Accuracy, Precision, Recall, and F1-score.
    """
    if labels is None:
        labels = sorted(list(set(list(y_true) + list(y_pred))))
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', labels=labels, zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', labels=labels, zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', labels=labels, zero_division=0)
    
    return {
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4)
    }

def print_classification_report(y_true, y_pred, labels=None):
    """
    Prints per-class metrics for detailed evaluation analysis.
    """
    unique_labels = sorted(list(set(list(y_true) + list(y_pred))))
    
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT (CROSS-VALIDATION)")
    print("=" * 60)
    
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=unique_labels,
        labels=unique_labels,
        digits=4,
        zero_division=0
    )
    print(report)
    return report

def plot_confusion_matrix(y_true, y_pred, labels=None, save_path=None):
    """
    Visualizes prediction accuracy via a confusion matrix heatmap.
    """
    unique_labels = sorted(list(set(list(y_true) + list(y_pred))))
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    
    plt.figure(figsize=(8, 6))
    if sns is not None:
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=unique_labels,
            yticklabels=unique_labels,
            cbar_kws={'label': 'Count'},
            linewidths=0.5,
            linecolor='black'
        )
    else:
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar(label='Count')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha='center', va='center', color='black')
        plt.xticks(range(len(unique_labels)), unique_labels)
        plt.yticks(range(len(unique_labels)), unique_labels)
    
    plt.title('Confusion Matrix - Cross Validation', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('Actual Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    plt.show()

def evaluate_with_cv(model, X, y, labels=None):
    """
    Evaluates the model using Leave-One-Out Cross Validation.
    This approach is mathematically robust for extremely small datasets.
    """
    print("\n" + "=" * 60)
    print("PERFORMING LEAVE-ONE-OUT CROSS-VALIDATION")
    print("=" * 60)
    
    # 1. Generate cross-validated predictions
    loo = LeaveOneOut()
    y_pred_cv = cross_val_predict(model, X, y, cv=loo)
    
    # 2. Compute and print metrics
    print("\nComputing CV metrics...")
    metrics = calculate_metrics(y, y_pred_cv, labels)
    for key, value in metrics.items():
        print(f" - {key.upper()}: {value * 100:.2f}%")
    
    # 3. Print detailed report
    print_classification_report(y, y_pred_cv, labels)
    
    # 4. Display Confusion Matrix
    print("\nOpening Confusion Matrix window...")
    plot_confusion_matrix(y, y_pred_cv, labels, save_path='outputs/cv_confusion_matrix.png')