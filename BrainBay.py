# Basic Libraries
import pandas as pd
import numpy as np

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.plotting import plot_decision_regions

# model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold, cross_val_predict

# preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import boxcox
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.base import clone

# pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# model development
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

# hyperparameter tuning
from sklearn.model_selection import GridSearchCV

# performance metrics
from sklearn.metrics import classification_report, make_scorer, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve


def check_tradeoffs(ml_model, X, y, cv=5, threshold_proba=0.5):
    """
    Plots the Precision-Recall tradeoff and the ROC AUC curve for a given machine learning model and data, as well as
    the ROC AUC score. The function uses cross-validation to generate predictions and probabilities for the test data.

    Args:
        ml_model: A machine learning model that implements the `fit` and `predict_proba` methods.
        X (array-like of shape (n_samples, n_features)): The input data.
        y (array-like of shape (n_samples,)): The target values.
        cv (int or cross-validation generator, default=5): Determines the cross-validation splitting strategy.
        threshold_proba (float, default=0.5): The probability threshold to use when making binary predictions.

    Returns:
        None
    """
    y_probas = cross_val_predict(ml_model, X, y, cv=cv, method='predict_proba')

    #selecting probabilities of positive class
    y_probas_donor = y_probas[:,1]
    
    # finding FPR, TPR, and threshold
    false_positive_rate, true_positive_rate, threshold = roc_curve(y, y_probas_donor)

    # Plotting Recall and Precision tradeoff
    precision, recall, threshold= precision_recall_curve(y, y_probas_donor)

    plt.figure(figsize=(15,6))
    plt.subplot(1,2,1)
    plt.plot(threshold, precision[:-1], color='m', label='precision')
    plt.plot(threshold, recall[:-1], color='c', label='recall')
    plt.annotate('Current_Threshold_Line', (threshold_proba, 0.5), (0.2, 0.3),
            arrowprops=dict(arrowstyle="->",connectionstyle="Arc3", color="k"),
            bbox = dict(boxstyle = "round", fc ="none", ec="k"))
    plt.axvline(threshold_proba, alpha=0.5)
    plt.xlabel('Threshold')
    plt.ylabel('Precision & Recall')
    plt.legend()
    plt.title('Precision-Recall Tradeoff'.title(), fontsize=20);

    # Plotting ROC AUC Curve
    plt.subplot(1,2,2)
    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, color="c")
    plt.plot([0,1], [0,1], "r--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title('ROC AUC Curve'.title(), fontsize=20);
    plt.tight_layout()

    # Finding ROC AUC score
    print('ROC AUC Score is: ', roc_auc_score(y, y_probas_donor))
    



def opt_threshold(classifier, train_features, train_labels, test_features, test_labels):
    """
    Optimizes the threshold value for a binary classifier based on the F1 score.
    
    Parameters
    ----------
    classifier : estimator object
        An object of the estimator class implementing 'fit' and 'predict_proba'.
    train_features : array-like of shape (n_samples, n_features)
        The input training data.
    train_labels : array-like of shape (n_samples,)
        The target training labels.
    test_features : array-like of shape (n_samples, n_features)
        The input test data.
    test_labels : array-like of shape (n_samples,)
        The target test labels.
        
    Returns
    -------
    None
        The function plots a graph of F1 scores for different threshold values and prints
        the maximum F1 score and the optimum threshold value.
    """
    # Create a list of threshold values to test
    threshold_list=list(np.arange(0,1, 0.02))
    
    # Initialize an empty list to store F1 scores for each threshold value
    f1_scores=[]

    # Loop through the threshold values and calculate F1 scores for each
    for num, threshold_val in enumerate(threshold_list):

        # Create a new instance of the classifier with the current threshold
        model = classifier

        # Train the model on the training data
        model.fit(train_features, train_labels)

        # Get the predicted probabilities of each class on the test data
        y_proba = model.predict_proba(test_features)

        # Set the thresholds based on the current threshold value
        y_pred_new = (y_proba[:, 1] >= threshold_val).astype(int)

        # Calculate the F1 score for the current threshold value
        f1_score_val = f1_score(test_labels, y_pred_new)

        # Append the F1 score to the list of F1 scores
        f1_scores.append(f1_score_val)

    # Plot the F1 scores for each threshold value
    plt.plot(threshold_list, f1_scores, color='r')

    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 scores for different Thresholds')
    
    # Print the maximum F1 score and the optimum threshold value
    print('Max F1 score is: ', np.max(f1_scores))
    index= np.argmax(f1_scores)
    opt_threshold=threshold_list[index]
    print('Optimum threshold is: ', opt_threshold)

