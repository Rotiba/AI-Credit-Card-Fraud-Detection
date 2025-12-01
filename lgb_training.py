# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 20:57:39 2025
@author: HP
"""

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

import lightgbm as lgb
import pandas as pd
import shap
import pickle
import json
import matplotlib.pyplot as plt

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

#Load Data

df = pd.read_csv(r"C:\Users\HP\Downloads\archive (6)\creditcard.csv")
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#preprocessing

numeric_features = ['Time', 'Amount']
preprocessor = ColumnTransformer(
    transformers=[('num', StandardScaler(), numeric_features)],
    remainder='passthrough'
)

resampler = SMOTE(random_state=42)

'''model training'''

#lightGBM Pipeline
lgb_pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('resample', resampler),
    ('classifier', lgb.LGBMClassifier(random_state=42))
])

# Train
lgb_pipeline.fit(X_train, y_train)

# Predictions
y_pred = lgb_pipeline.predict(X_test)
y_prob = lgb_pipeline.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

'''Model visual evaluation'''
#precision-recall curve

precision, recall, pr_thresholds = precision_recall_curve(y_test, y_prob)

plt.figure(figsize=(7, 5))
PrecisionRecallDisplay(precision=precision, recall=recall).plot()
plt.title("Precision–Recall Curve (LightGBM)")
plt.show()

#ROC Curve

fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(7, 5))
RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc_score(y_test, y_prob)).plot()
plt.title("ROC Curve (LightGBM)")
plt.show()

#Confusion matrix

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap="Oranges")
plt.title("Confusion Matrix (LightGBM)")
plt.show()

#Save as JSON
cm_dict = {
    "model": "LightGBM",
    "confusion_matrix": cm.tolist()
}
with open("lightgbm_confusion_matrix.json", "w") as f:
    json.dump(cm_dict, f, indent=4)

'''model saving'''

pickle.dump(lgb_pipeline, open("lightgbm_model.pkl", "wb"))
print("Model saved as lightgbm_model.pkl")

#save model parameters

lgb_clf = lgb_pipeline.named_steps['classifier']

model_params = {
    "model": "LightGBM",
    "params": lgb_clf.get_params()
}

with open("lightgbm_parameters.json", "w") as f:
    json.dump(model_params, f, indent=4)

print("Model parameters saved to lightgbm_parameters.json")

# Save ROC and PR metrics for comparison
metrics_to_save = {
    "model": "LightGBM",
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

with open("lightgbm_metrics.json", "w") as f:
    json.dump(metrics_to_save, f, indent=4)

print("ROC + PR metrics saved to lightgbm_metrics.json")

#SHAP Explainability
X_test_preprocessed = lgb_pipeline.named_steps['preprocessor'].transform(X_test)

explainer = shap.TreeExplainer(lgb_pipeline.named_steps['classifier'])
shap_values = explainer.shap_values(X_test_preprocessed)

shap.summary_plot(
    shap_values,
    X_test_preprocessed,
    feature_names=X_test.columns
)
