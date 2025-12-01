# -*- coding: utf-8 -*-
"""
Auto-load all model metrics and confusion matrices from a folder
Plots: ROC curves, Precision–Recall curves, Confusion Matrices
Displays: AUC table
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

#set folder path
folder = r"C:\Users\HP\OneDrive\Documents\SRW\MY PORTFOLIO\CreditFraud\modelparameters" 

#Autodetect metric and confusion files

metrics_files = {}
confusion_files = {}

for file in os.listdir(folder):
    if file.endswith("_metrics.json"):
        model_name = file.replace("_metrics.json", "")
        metrics_files[model_name] = os.path.join(folder, file)
    elif file.endswith("_confusion_matrix.json"):
        model_name = file.replace("_confusion_matrix.json", "")
        confusion_files[model_name] = os.path.join(folder, file)

# Ensure that only compare models that have both files
models = sorted(set(metrics_files.keys()) & set(confusion_files.keys()))

'''load metrics'''

all_metrics = {}
all_confusions = {}

for model in models:
    # Load metrics
    with open(metrics_files[model], "r") as f:
        all_metrics[model] = json.load(f)
    
    # Load confusion matrices
    with open(confusion_files[model], "r") as f:
        cm_dict = json.load(f)
        all_confusions[model] = np.array(cm_dict["confusion_matrix"])

#plot ROC curve

plt.figure(figsize=(8, 6))

for model in models:
    metrics = all_metrics[model]
    fpr = np.array(metrics["roc_curve"]["fpr"])
    tpr = np.array(metrics["roc_curve"]["tpr"])
    auc = metrics["roc_auc"]
    plt.plot(fpr, tpr, label=f"{model} (AUC={auc:.4f})")

plt.plot([0, 1], [0, 1], "k--")  # Random baseline
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves Comparison")
plt.legend()
plt.grid(True)
plt.show()

#plot precision recall curve

plt.figure(figsize=(8, 6))

for model in models:
    metrics = all_metrics[model]
    precision = np.array(metrics["precision_recall_curve"]["precision"])
    recall = np.array(metrics["precision_recall_curve"]["recall"])
    plt.plot(recall, precision, label=f"{model}")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curves Comparison")
plt.legend()
plt.grid(True)
plt.show()

#plot all confusion matrix

n_models = len(models)
cols = 2
rows = int(np.ceil(n_models / cols))

fig, axes = plt.subplots(rows, cols, figsize=(cols*6, rows*5))

for ax, model in zip(axes.flatten(), models):
    cm = all_confusions[model]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(model)

# Hide any empty subplots
for ax in axes.flatten()[n_models:]:
    ax.axis("off")

plt.tight_layout()
plt.show()

#create AUC table

print("===== AUC Comparison Table =====")
print("{:<25} {:<10}".format("Model", "ROC AUC"))

for model in models:
    auc = all_metrics[model]["roc_auc"]
    print("{:<25} {:.4f}".format(model, auc))
