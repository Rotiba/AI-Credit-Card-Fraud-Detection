# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 20:58:16 2025
@author: HP
"""

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier
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

#load data

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

#Random Forest Pipeline

rf_pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('resample', resampler),
    ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
])

# Train
rf_pipeline.fit(X_train, y_train)

# Predictions
y_pred = rf_pipeline.predict(X_test)
y_prob = rf_pipeline.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

#Precision-Recall Curve

precision, recall, pr_thresholds = precision_recall_curve(y_test, y_prob)

plt.figure(figsize=(7, 5))
PrecisionRecallDisplay(precision=precision, recall=recall).plot()
plt.title("Precision–Recall Curve (Random Forest)")
plt.show()

#ROC Curve

fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(7, 5))
RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc_score(y_test, y_prob)).plot()
plt.title("ROC Curve (Random Forest)")
plt.show()


#Confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap="nipy_spectral")
plt.title("Confusion Matrix (Random Forest)")
plt.show()

# Save confusion matrix
cm = confusion_matrix(y_test, y_pred)

#Save as JSON
cm_dict = {
    "model": "RandomForest",
    "confusion_matrix": cm.tolist()  # Convert NumPy array to list for JSON
}
with open("random_forest_confusion_matrix.json", "w") as f:
    json.dump(cm_dict, f, indent=4)




#Save model

pickle.dump(rf_pipeline, open("random_forest_model.pkl", "wb"))
print("Model saved as random_forest_model.pkl")

#Save model parameters

rf_clf = rf_pipeline.named_steps['classifier']

model_params = {
    "model": "RandomForest",
    "params": rf_clf.get_params()
}

with open("random_forest_parameters.json", "w") as f:
    json.dump(model_params, f, indent=4)

print("Model parameters saved to random_forest_parameters.json")

#Save ROC and PR for comparison

metrics_to_save = {
    "model": "RandomForest",
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

with open("random_forest_metrics.json", "w") as f:
    json.dump(metrics_to_save, f, indent=4)

print("ROC + PR metrics saved to random_forest_metrics.json")

#SHAP Explainability

X_test_preprocessed = rf_pipeline.named_steps['preprocessor'].transform(X_test)

explainer = shap.TreeExplainer(rf_clf)
shap_values = explainer.shap_values(X_test_preprocessed)

shap.summary_plot(
    shap_values,
    X_test_preprocessed,
    feature_names=X_test.columns
)
