from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score, roc_curve, roc_auc_score
from matplotlib import pyplot
import pandas as pd


def evaluate_metrics(y_true, y_pred):
    results = pd.DataFrame(classification_report(y_true, y_pred, zero_division=0, output_dict=True)).transpose()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn+fp)
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    auc_score = metrics.auc(fpr, tpr)
    return results, specificity, auc_score


def calc_roc_curve(y_true, y_pred, label=''):
    lr_auc = roc_auc_score(y_true, y_pred)
    print('ROC AUC=%.3f' % lr_auc)
    pr_auc = average_precision_score(y_true, y_pred)
    print("PR AUC:", pr_auc)
    lr_fpr, lr_tpr, _ = roc_curve(y_true, y_pred)
    pyplot.plot(lr_fpr, lr_tpr, marker='.', label=label)
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.legend()
    pyplot.show()

