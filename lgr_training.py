# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 20:49:06 2025
@author: HP
"""

import pandas as pd
import shap
import pickle
import json
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay
)

#Load Dataset
df = pd.read_csv(r"C:\Users\HP\Downloads\archive (6)\creditcard.csv")
X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


#Preprocessing

numeric_features = ['Time', 'Amount']
preprocessor = ColumnTransformer(
    transformers=[('num', StandardScaler(), numeric_features)],
    remainder='passthrough'
)

resampler = SMOTE(random_state=42)

'''Model Training and Evaluation'''

#Logistic Regression Pipeline
lr_pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('resample', resampler),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Train model
lr_pipeline.fit(X_train, y_train)
# Predictions
y_pred = lr_pipeline.predict(X_test)
y_prob = lr_pipeline.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

'''Visual Evaluation '''
#Precision-Recall Curve
precision, recall, pr_thresholds = precision_recall_curve(y_test, y_prob)
plt.figure(figsize=(7, 5))
PrecisionRecallDisplay(precision=precision, recall=recall).plot()
plt.title("Precision–Recall Curve (Logistic Regression)")
plt.show()

#ROC Curve
fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(7, 5))
RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc_score(y_test, y_prob)).plot()
plt.title("ROC Curve (Logistic Regression)")
plt.show()

#Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap="Blues")
plt.title("Confusion Matrix (Logistic Regression)")
plt.show()

#Save as JSON
cm_dict = {
    "model": "Logistic Regression",
    "confusion_matrix": cm.tolist()
}
with open("logistic_regression_confusion_matrix.json", "w") as f:
    json.dump(cm_dict, f, indent=4)

'''Saving of Models and Parameters '''
#save logistic regression pipeline

pickle.dump(lr_pipeline, open("logistic_regression_model.pkl", "wb"))
print("Model saved as logistic_regression_model.pkl")

#Save model parameters (coefficients)

lr_clf = lr_pipeline.named_steps['classifier']
model_params = {
    "model": "LogisticRegression",
    "intercept": lr_clf.intercept_.tolist(),
    "coefficients": lr_clf.coef_.tolist(),
    "features": X_train.columns.tolist()
}

with open("logistic_regression_parameters.json", "w") as f:
    json.dump(model_params, f, indent=4)

print("Model parameters saved to logistic_regression_parameters.json")

#   SAVE ROC, PR metrics for comparison with other models
metrics_to_save = {
    "model": "LogisticRegression",
    "roc_auc": roc_auc_score(y_test, y_prob),
    "roc_curve": {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": roc_thresholds.tolist()
    },
    "precision_recall_curve": {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": pr_thresholds.tolist()
    }
}

with open("logistic_regression_metrics.json", "w") as f:
    json.dump(metrics_to_save, f, indent=4)

print("ROC and PR metrics saved to logistic_regression_metrics.json")

