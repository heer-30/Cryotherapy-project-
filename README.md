* Cryotherapy Analysis and Prediction
This project analyzes patient data related to Cryotherapy treatment and predicts treatment outcomes using multiple machine learning algorithms.
It compares models like Logistic Regression, Ridge Classifier, KNN, SVM, Decision Tree, and Random Forest to identify the best-performing model.

* Features
Loads Cryotherapy dataset (Cryotherapy.xlsx)
Splits data into training and testing sets
Standardizes features using StandardScaler
Trains and evaluates multiple ML models
Calculates and displays key performance metrics:
Accuracy, Precision, Recall, F1-Score, Specificity, ROC-AUC

* Plots:-
Combined ROC Curve
Precision-Recall Curve
Confusion Matrices for each model
Box Plot of dataset features
Compares all models in a summarized results table

* Files:-
Cryotherapy_Comparison.ipynb → Main notebook containing data preprocessing, training, evaluation, and visualization
Cryotherapy.xlsx → Input dataset
results/ → Contains generated plots (ROC, PR Curve, Confusion Matrices, Box Plot)
README.md → Project description file

* Models Used
Logistic Regression
Ridge Classifier
K-Nearest Neighbors (KNN)
Support Vector Machine (SVM)
Decision Tree
Random Forest

* Requirements
Python 3.8 or later

* Libraries:
pip install pandas numpy matplotlib seaborn scikit-learn kagglehub

* How to Run
Download dataset from Kaggle or use kagglehub to fetch it automatically.
Open the notebook in Google Colab or Jupyter Notebook.
Run all cells sequentially.
View accuracy, ROC, Precision-Recall, and confusion matrix visualizations for each model.

* Result:-
The Support Vector Machine (SVM) achieved the highest accuracy (92%) with a strong balance between precision, recall, and AUC score — making it the best model for Cryotherapy outcome prediction.

* Authors:-
Heer Khalasi
Nisha Singh
