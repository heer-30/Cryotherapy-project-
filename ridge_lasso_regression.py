import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_excel(r'D:\python\FDS project\Cryotherapy.xlsx')  # Replace with your path

# Assuming the dataset has features and a target column named 'Result_of_Treatment'
X = df.drop('Result_of_Treatment', axis=1)  # Features
y = df['Result_of_Treatment']  # Target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_scaled)

# Lasso Regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train)
y_pred_lasso = lasso.predict(X_test_scaled)

# Evaluate Ridge Regression
print("Ridge Regression Performance:")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred_ridge)}")
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred_ridge)}")
print(f"R-Squared: {r2_score(y_test, y_pred_ridge)}")

# Evaluate Lasso Regression
print("\nLasso Regression Performance:")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred_lasso)}")
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred_lasso)}")
print(f"R-Squared: {r2_score(y_test, y_pred_lasso)}")

# AUROC and AUC for Ridge Regression
auc_score_ridge = roc_auc_score(y_test, y_pred_ridge)
print("\nRidge Regression AUROC and AUC:", auc_score_ridge)

# AUROC and AUC for Lasso Regression
auc_score_lasso = roc_auc_score(y_test, y_pred_lasso)
print("\nLasso Regression AUROC and AUC:", auc_score_lasso)

# Plotting ROC curves for Ridge and Lasso Regression
fpr_ridge, tpr_ridge, _ = roc_curve(y_test, y_pred_ridge)
fpr_lasso, tpr_lasso, _ = roc_curve(y_test, y_pred_lasso)

plt.figure(figsize=(12, 6))

# Ridge ROC Curve
plt.subplot(1, 2, 1)
plt.plot(fpr_ridge, tpr_ridge, label=f'Ridge (AUC = {auc_score_ridge:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Ridge Regression: ROC Curve')
plt.legend(loc='lower right')

# Lasso ROC Curve
plt.subplot(1, 2, 2)
plt.plot(fpr_lasso, tpr_lasso, label=f'Lasso (AUC = {auc_score_lasso:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Lasso Regression: ROC Curve')
plt.legend(loc='lower right')

plt.tight_layout()
plt.show()

# Plotting Actual vs Predicted for Ridge and Lasso
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_ridge)
plt.title('Ridge Regression: Actual vs Predicted')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_lasso)
plt.title('Lasso Regression: Actual vs Predicted')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

plt.tight_layout()
plt.show()

# 13. Box Plot
plt.figure()
sns.boxplot(data=df)
plt.title('Box Plot of Features')
plt.show()

# Confusion Matrix Visualization for Ridge Regression
ridge_conf_matrix = confusion_matrix(y_test, np.round(y_pred_ridge))
plt.figure(figsize=(6, 6))
sns.heatmap(ridge_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
plt.title('Confusion Matrix - Ridge Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Confusion Matrix Visualization for Lasso Regression
lasso_conf_matrix = confusion_matrix(y_test, np.round(y_pred_lasso))
plt.figure(figsize=(6, 6))
sns.heatmap(lasso_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
plt.title('Confusion Matrix - Lasso Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
