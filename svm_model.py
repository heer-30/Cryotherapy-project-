import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc, classification_report, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler

# Load the dataset
# Replace 'Cryotherapy.xlsx' with the correct path to your dataset
df = pd.read_excel(r'D:\python\FDS project\Cryotherapy.xlsx')  # Ensure the path is correct

# Assuming the dataset has features and a target column named 'Result_of_Treatment'
X = df.drop('Result_of_Treatment', axis=1)  # Features
y = df['Result_of_Treatment']  # Target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the SVM model
model = SVC(probability=True, kernel='rbf', C=1.0, gamma='scale', random_state=42)
model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = model.predict(X_test_scaled)
y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]  # Get probabilities for the positive class

# 1. Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# 2. Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# 3. Balanced Accuracy (BAAC)
specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
baac = (recall_score(y_test, y_pred) + specificity) / 2
print("Balanced Accuracy (BAAC):", baac)

# 4. Precision
precision = precision_score(y_test, y_pred)
print("Precision:", precision)

# 5. Recall (Sensitivity)
recall = recall_score(y_test, y_pred)
print("Recall (Sensitivity):", recall)

# 6. Specificity
print("Specificity:", specificity)

# 7. AUROC and AUC
auc_score = roc_auc_score(y_test, y_pred_prob)
print("AUROC and AUC:", auc_score)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# 8. Precision-Recall (P-R) Curve
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_prob)
plt.figure()
plt.plot(recall_vals, precision_vals, label='P-R curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.show()

# 9. Classification Report
classification_report_str = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_report_str)

# 10. F1-Score
f1 = f1_score(y_test, y_pred)
print("F1-Score:", f1)

# 11. Error Plot
errors = y_test.values - y_pred
plt.figure()
plt.plot(errors, marker='o', linestyle='', label='Errors')
plt.axhline(0, color='r', linestyle='--')
plt.title('Error Plot')
plt.xlabel('Samples')
plt.ylabel('Error (y_true - y_pred)')
plt.legend()
plt.show()

# 12. Box Plot
plt.figure()
sns.boxplot(data=df)
plt.title('Box Plot of Features')
plt.show()

# 13. Confusion Matrix Visualization
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], 
            yticklabels=['Actual Negative', 'Actual Positive'])
plt.title('SVM: Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
