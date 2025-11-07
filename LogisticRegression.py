import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, classification_report, precision_recall_curve
)

# Load the dataset
df = pd.read_excel(r'D:\python\FDS project\Cryotherapy.xlsx')

# Debug: Print column names to verify structure
print("Columns in the dataset:", df.columns)

# Define the target column and features
target_column = 'Result_of_Treatment'
features = ['sex', 'age', 'Time', 'Number_of_Warts', 'Type', 'Area']

# Extract features and target
X = df[features]
y = df[target_column]

X = pd.get_dummies(X, drop_first=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

# 1. Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Plot Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# 2. Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# 3. Balanced Accuracy (BAAC)
baac = (recall_score(y_test, y_pred) + (conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1]))) / 2
print("Balanced Accuracy (BAAC):", baac)

# 4. Precision
precision = precision_score(y_test, y_pred)
print("Precision:", precision)

# 5. Recall
recall = recall_score(y_test, y_pred)
print("Recall:", recall)

# 6. Specificity
specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
print("Specificity:", specificity)

# 7. Sensitivity (Recall)
sensitivity = recall
print("Sensitivity:", sensitivity)

# 8. AUROC and AUC
auc_score = roc_auc_score(y_test, y_pred_prob)
print("AUROC and AUC:", auc_score)

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# 9. Precision-Recall (P-R) Curve
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_prob)
plt.figure()
plt.plot(recall_vals, precision_vals, label='P-R curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.show()

# 10. Classification Report
classification_report_str = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_report_str)

# 11. F1-Score
f1 = f1_score(y_test, y_pred)
print("F1-Score:", f1)

# 12. Error Plot
errors = y_test - y_pred
plt.figure()
plt.plot(errors, marker='o', linestyle='')
plt.title('Error Plot')
plt.xlabel('Samples')
plt.ylabel('Error (y_true - y_pred)')
plt.show()

# 13. Box Plot
plt.figure()
sns.boxplot(data=df[features])  # Boxplot for the features only
plt.title('Box Plot of Features')
plt.show()

