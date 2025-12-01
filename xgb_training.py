# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 20:58:06 2025
@author: HP
"""

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

import xgboost as xgb
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

#XGBoost Classifier Pipeline

xgb_pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('resample', resampler),
    ('classifier', xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    ))
])

# Train
xgb_pipeline.fit(X_train, y_train)

# Predictions
y_pred = xgb_pipeline.predict(X_test)
y_prob = xgb_pipeline.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

#Precision-Recall Curve

precision, recall, pr_thresholds = precision_recall_curve(y_test, y_prob)

plt.figure(figsize=(7, 5))
PrecisionRecallDisplay(precision=precision, recall=recall).plot()
plt.title("Precision–Recall Curve (XGBoost)")
plt.show()

#ROC Curve

fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(7, 5))
RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc_score(y_test, y_prob)).plot()
plt.title("ROC Curve (XGBoost)")
plt.show()

#Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap="viridis")
plt.title("Confusion Matrix (XGBoost)")
plt.show()

#Save as JSON
cm_dict = {
    "model": "XGBoost",
    "confusion_matrix": cm.tolist()
}
with open("xgboost_confusion_matrix.json", "w") as f:
    json.dump(cm_dict, f, indent=4)

#Save Model

pickle.dump(xgb_pipeline, open("xgboost_model.pkl", "wb"))
print("Model saved as xgboost_model.pkl")

#Save Model Parameters

xgb_clf = xgb_pipeline.named_steps['classifier']

model_params = {
    "model": "XGBoost",
    "params": xgb_clf.get_params()
}

with open("xgboost_parameters.json", "w") as f:
    json.dump(model_params, f, indent=4)

print("Model parameters saved to xgboost_parameters.json")

#Save ROC PR for comparison

metrics_to_save = {
    "model": "XGBoost",
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

with open("xgboost_metrics.json", "w") as f:
    json.dump(metrics_to_save, f, indent=4)

print("ROC + PR metrics saved to xgboost_metrics.json")

#SHAP Explainability

X_test_preprocessed = xgb_pipeline.named_steps['preprocessor'].transform(X_test)

explainer = shap.TreeExplainer(xgb_clf)
shap_values = explainer.shap_values(X_test_preprocessed)

shap.summary_plot(
    shap_values,
    X_test_preprocessed,
    feature_names=X_test.columns
)
