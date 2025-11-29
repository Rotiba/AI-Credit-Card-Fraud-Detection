# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 20:49:06 2025

@author: HP
"""

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd
import shap


df = pd.read_csv(r"C:\Users\HP\Downloads\archive (6)\creditcard.csv")
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#preprocessing
numeric_features = ['Time', 'Amount']
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features)
    ],
    remainder='passthrough'
)



# Define resampler
resampler = SMOTE(random_state=42)

# Logistic Regression training and evaluation
lr_pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('resample', resampler),
    ('classifier', LogisticRegression(max_iter=1000))
])

lr_pipeline.fit(X_train, y_train)

y_pred = lr_pipeline.predict(X_test)
y_prob = lr_pipeline.predict_proba(X_test)[:,1]
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

''' Logistic Regression Model Explainability '''
# Extract trained logistic regression model
lr_clf = lr_pipeline.named_steps['classifier']
# Preprocess test data (must match training transformations)
X_test_preprocessed = lr_pipeline.named_steps['preprocessor'].transform(X_test)
# Create SHAP explainer for linear models
explainer = shap.LinearExplainer(lr_clf, X_test_preprocessed)
# Compute SHAP values
shap_values = explainer.shap_values(X_test_preprocessed)
# Summary plot (global feature importance)
shap.summary_plot(shap_values, X_test_preprocessed, feature_names=X_test.columns)







# Random Forest
rf_pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('resample', resampler),
    ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
])

rf_pipeline.fit(X_train, y_train)
y_pred = rf_pipeline.predict(X_test)
y_prob = rf_pipeline.predict_proba(X_test)[:,1]
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

# XGBoost
xgb_pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('resample', resampler),
    ('classifier', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
])

xgb_pipeline.fit(X_train, y_train)
y_pred = xgb_pipeline.predict(X_test)
y_prob = xgb_pipeline.predict_proba(X_test)[:,1]
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))


# LightGBM
lgb_pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('resample', resampler),
    ('classifier', lgb.LGBMClassifier(random_state=42))
])

lgb_pipeline.fit(X_train, y_train)

y_pred = lgb_pipeline.predict(X_test)
y_prob = lgb_pipeline.predict_proba(X_test)[:,1]
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))





''' Logistic Regression Model Explainability '''

# Extract trained logistic regression model
lr_clf = lr_pipeline.named_steps['classifier']
# Preprocess test data (must match training transformations)
X_test_preprocessed = lr_pipeline.named_steps['preprocessor'].transform(X_test)
# Create SHAP explainer for linear models
explainer = shap.LinearExplainer(lr_clf, X_test_preprocessed)
# Compute SHAP values
shap_values = explainer.shap_values(X_test_preprocessed)
# Summary plot (global feature importance)
shap.summary_plot(shap_values, X_test_preprocessed, feature_names=X_test.columns)




'''model explainability'''
# Example with Random Forest
explainer = shap.TreeExplainer(rf_pipeline.named_steps['classifier'])
# Get preprocessed test data
X_test_preprocessed = rf_pipeline.named_steps['preprocessor'].transform(X_test)
shap_values = explainer.shap_values(X_test_preprocessed)

# Summary plot
shap.summary_plot(shap_values, X_test_preprocessed, feature_names=X_test.columns)

