#!/usr/bin/env python
# coding: utf-8

# In[9]:


import joblib
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, roc_auc_score
from sklearn.metrics import classification_report
import shap
from scipy.stats import sem
import statsmodels.stats.proportion as smprop


# In[2]:


np.random.seed(1)
model_path = r'your_data_path\PETCT_model.pkl'
ct_data_path = r'your_data_path\CT_bin50-norm.csv' 
pet_data_path = r'your_data_path\PETbin0.05-TSpR-norm.csv'  

model_dict = joblib.load(model_path)
PETCT_model = model_dict['PETCT_model']
CT_selected_features = model_dict['CT_selected_features']
PET_selected_features = model_dict['PET_selected_features']
feature_config = model_dict['feature_config']


def load_external_data(ct_path, pet_path):
    ct_df = pd.read_csv(ct_path)
    pet_df = pd.read_csv(pet_path)
    y_external = ct_df['Label']
    return ct_df.drop('Label', axis=1), pet_df.drop('Label', axis=1), y_external

ct_features, pet_features, y_external = load_external_data(ct_data_path, pet_data_path)


def process_features(ct_data, pet_data):
    ct_selected = ct_data[CT_selected_features]
    pet_selected = pet_data[PET_selected_features]
    ct_selected = ct_selected.add_prefix(feature_config['CT_prefix'])
    pet_selected = pet_selected.add_prefix(feature_config['PET_prefix'])
    X_external = pd.concat([ct_selected, pet_selected], axis=1)
    return X_external

X_external = process_features(ct_features, pet_features)
y_pred_proba = PETCT_model.predict_proba(X_external)[:, 1]
y_pred = PETCT_model.predict(X_external)


# In[7]:


def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()
    
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    ppv = TP / (TP + FP) if (TP + FP) != 0 else 0
    npv = TN / (TN + FN) if (TN + FN) != 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    f1 = 2 * (ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) != 0 else 0
    
    return {
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'PPV': ppv,
        'NPV': npv,
        'Accuracy': accuracy,
        'F1': f1
    }


metrics = calculate_metrics(y_external, y_pred)
print("Classification Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")
np.random.seed(1)
def bootstrap_ci(y_true, y_pred_proba, n_bootstraps=1000):
    bootstrapped_auc = []
    bootstrapped_ap = []
    n_samples = len(y_true)
    
    for _ in range(n_bootstraps):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        if len(np.unique(y_true[indices])) < 2:
            continue

        auc_score = roc_auc_score(y_true[indices], y_pred_proba[indices])
        precision, recall, _ = precision_recall_curve(y_true[indices], y_pred_proba[indices])
        ap_score = auc(recall, precision)
        
        bootstrapped_auc.append(auc_score)
        bootstrapped_ap.append(ap_score)
    
    auc_ci = np.percentile(bootstrapped_auc, [2.5, 97.5])
    ap_ci = np.percentile(bootstrapped_ap, [2.5, 97.5])
    
    return auc_ci, ap_ci


roc_auc = roc_auc_score(y_external, y_pred_proba)
precision, recall, _ = precision_recall_curve(y_external, y_pred_proba)
pr_auc = auc(recall, precision)


auc_ci, pr_ci = bootstrap_ci(y_external, y_pred_proba)


print("\nAUC with 95% CI:")
print(f"ROC AUC: {roc_auc:.4f} ({auc_ci[0]:.4f}-{auc_ci[1]:.4f})")
print(f"PR AUC: {pr_auc:.4f} ({pr_ci[0]:.4f}-{pr_ci[1]:.4f})")


# In[12]:


plt.rcParams.update({'font.size': 10, 'font.family': 'Arial', 
                    'axes.labelsize': 12, 'axes.labelweight': 'bold'})

explainer = shap.TreeExplainer(PETCT_model)
shap_values = explainer.shap_values(X_external)
plt.figure(figsize=(20, 8), dpi=300)  # 进一步加宽画布

shap_color_map = plt.cm.RdBu(np.linspace(0,1,256))[64:192]
shap_color_map = ListedColormap(shap_color_map)

shap.summary_plot(
    shap_values[1], 
    X_external,
    plot_type="dot",
    show=False,
    color=shap_color_map,
    color_bar_label="Feature value",
    max_display=15,
    plot_size=(15, 8) 
)

plt.xlabel("SHAP Value (impact on model output)", fontsize=12, labelpad=10, fontweight='normal') 
plt.xticks(fontsize=12) 
plt.xlim(-0.3, 0.3) 

cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=10)  
cbar.set_label("Feature value", fontsize=12, labelpad=10, fontweight='normal') 
cbar.outline.set_visible(False)  

plt.tight_layout(pad=3)
plt.show()


# In[ ]:




