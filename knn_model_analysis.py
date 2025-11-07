import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc, classification_report, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler

# Load the dataset
# Replace 'data.csv' with the path to your dataset
df = pd.read_excel(r'D:\python\FDS project\Cryotherapy.xlsx')  # Ensure correct path to your dataset

# Assuming the dataset has features and a target column named 'target'
X = df.drop('Result_of_Treatment', axis=1)  # Features (update column name accordingly)
y = df['Result_of_Treatment'] # Target (update column name accordingly)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features (important for KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the KNN model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = model.predict(X_test_scaled)
y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]

# 1. Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

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

fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {auc_score:.2f})')
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

# 10. Classification Score
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
sns.boxplot(data=df)
plt.title('Box Plot of Features')
plt.show()

# 14. Confusion Matrix Visualization
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], 
            yticklabels=['Actual Negative', 'Actual Positive'])
plt.title('KNN: Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
