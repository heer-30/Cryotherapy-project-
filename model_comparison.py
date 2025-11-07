import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, classification_report, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_excel(r'D:\python\FDS project\Cryotherapy.xlsx')

# Features and target
X = df.drop('Result_of_Treatment', axis=1)
y = df['Result_of_Treatment']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models to evaluate
models = {
    'Logistic Regression': LogisticRegression(),
    'Ridge Classifier': RidgeClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Support Vector Machine': SVC(probability=True, kernel='rbf'),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

# Collect metrics
comparison_results = []

# Initialize combined ROC and Precision-Recall plots
plt.figure(figsize=(10, 6))
plt.title('Combined ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.figure(figsize=(10, 6))
plt.title('Combined Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')

# Evaluate each model
for idx, (model_name, model) in enumerate(models.items()):
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_pred_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    specificity = conf_matrix[0, 0] / (conf_matrix[0, :].sum())
    auc_score = roc_auc_score(y_test, y_pred_prob) if y_pred_prob is not None else None

    # Store results
    comparison_results.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Specificity': specificity,
        'AUC': auc_score
    })

    # Plot ROC Curve
    if y_pred_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        plt.figure(1)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.2f})')

    # Plot Precision-Recall Curve
    if y_pred_prob is not None:
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_prob)
        plt.figure(2)
        plt.plot(recall_vals, precision_vals, label=f'{model_name}')

    # Plot Confusion Matrix in a subplot (grid layout)
    plt.figure(3)
    plt.subplot(2, 3, idx + 1)  # 2 rows, 3 columns, model idx
    sns.heatmap(conf_matrix, annot=True, fmt='.0f', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

# Finalize combined ROC and Precision-Recall plots
plt.figure(1)
plt.legend(loc='lower right')
plt.show()

plt.figure(2)
plt.legend(loc='lower left')
plt.show()

# Summary of metrics
results_df = pd.DataFrame(comparison_results)
print(results_df)

# Box Plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df)
plt.title('Box Plot of Features')
plt.show()





