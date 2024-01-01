import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, auc, precision_recall_curve

def calculate_metrics(y, y_pred, y_scores):
    """Calculate metrics"""
    pre, rec, _ = precision_recall_curve(y, y_scores)
    df = pd.DataFrame()
    df[['Accuracy', 'Precision', 'Recall', 'f1-score']] = [[accuracy_score(y, y_pred), precision_score(y, y_pred), recall_score(y, y_pred), f1_score(y, y_pred)]]
    df[['ROC-AUC', 'PR-AUC']] = [roc_auc_score(y, y_scores), auc(rec, pre)]
    df[['TN', 'FP', 'FN', 'TP']] = np.ravel(confusion_matrix(y, y_pred))

    return df