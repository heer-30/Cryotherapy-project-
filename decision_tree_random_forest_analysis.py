import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc, classification_report, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler

# Load the dataset
# Replace 'data.csv' with the path to your CSV file
df = pd.read_excel(r'D:\python\FDS project\Cryotherapy.xlsx')

# Assuming the dataset has features and a target column named 'target'
X = df.drop('Result_of_Treatment', axis=1)  # Features (update column name accordingly)
y = df['Result_of_Treatment']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_scaled, y_train)

# Create and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Predict on the test set using Decision Tree
y_pred_dt = dt_model.predict(X_test_scaled)
y_pred_prob_dt = dt_model.predict_proba(X_test_scaled)[:, 1]

# Predict on the test set using Random Forest
y_pred_rf = rf_model.predict(X_test_scaled)
y_pred_prob_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

# Evaluation: Decision Tree
print("\n=== Decision Tree Evaluation ===")
dt_conf_matrix = confusion_matrix(y_test, y_pred_dt)
print("Confusion Matrix:\n", dt_conf_matrix)
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Balanced Accuracy (BAAC):", (recall_score(y_test, y_pred_dt) + 
      dt_conf_matrix[0, 0] / (dt_conf_matrix[0, 0] + dt_conf_matrix[0, 1])) / 2)
print("Precision:", precision_score(y_test, y_pred_dt))
print("Recall:", recall_score(y_test, y_pred_dt))
print("Specificity:", dt_conf_matrix[0, 0] / (dt_conf_matrix[0, 0] + dt_conf_matrix[0, 1]))
print("F1-Score:", f1_score(y_test, y_pred_dt))
dt_auc_score = roc_auc_score(y_test, y_pred_prob_dt)
print("AUROC and AUC:", dt_auc_score)

fpr_dt, tpr_dt, _ = roc_curve(y_test, y_pred_prob_dt)
plt.figure()
plt.plot(fpr_dt, tpr_dt, label=f'DT ROC curve (area = {dt_auc_score:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Decision Tree ROC Curve')
plt.legend(loc='lower right')
plt.show()

# 14. Confusion Matrix Visualization for Decision Tree
plt.figure(figsize=(6, 5))
sns.heatmap(dt_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], 
            yticklabels=['Actual Negative', 'Actual Positive'])
plt.title('Decision Tree: Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Evaluation: Random Forest
print("\n=== Random Forest Evaluation ===")
rf_conf_matrix = confusion_matrix(y_test, y_pred_rf)
print("Confusion Matrix:\n", rf_conf_matrix)
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Balanced Accuracy (BAAC):", (recall_score(y_test, y_pred_rf) + 
      rf_conf_matrix[0, 0] / (rf_conf_matrix[0, 0] + rf_conf_matrix[0, 1])) / 2)
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall:", recall_score(y_test, y_pred_rf))
print("Specificity:", rf_conf_matrix[0, 0] / (rf_conf_matrix[0, 0] + rf_conf_matrix[0, 1]))
print("F1-Score:", f1_score(y_test, y_pred_rf))
rf_auc_score = roc_auc_score(y_test, y_pred_prob_rf)
print("AUROC and AUC:", rf_auc_score)

fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_prob_rf)
plt.figure()
plt.plot(fpr_rf, tpr_rf, label=f'RF ROC curve (area = {rf_auc_score:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest ROC Curve')
plt.legend(loc='lower right')
plt.show()

# Precision-Recall Curve for Random Forest
precision_vals_rf, recall_vals_rf, _ = precision_recall_curve(y_test, y_pred_prob_rf)
plt.figure()
plt.plot(recall_vals_rf, precision_vals_rf, label='RF P-R curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Random Forest Precision-Recall Curve')
plt.legend(loc='lower left')
plt.show()

# 15. Confusion Matrix Visualization for Random Forest
plt.figure(figsize=(6, 5))
sns.heatmap(rf_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], 
            yticklabels=['Actual Negative', 'Actual Positive'])
plt.title('Random Forest: Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Box Plot
plt.figure()
sns.boxplot(data=df)
plt.title('Box Plot of Features')
plt.show()
