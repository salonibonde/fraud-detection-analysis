# Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# Load Dataset
fraud = pd.read_csv("data.csv")

# Data Preprocessing: Remove special characters from categorical columns
for col in ['customer', 'age', 'gender', 'zipcodeOri', 'merchant', 'zipMerchant', 'category']:
    fraud[col] = fraud[col].str.replace('[^\w\s]', '')

# Convert necessary columns to categorical or appropriate data types
fraud = fraud.drop(["zipcodeOri", "zipMerchant"], axis=1)
fraud["step"] = fraud["step"].astype("category")
fraud["customer"] = fraud["customer"].astype("category")
fraud["age"] = fraud["age"].astype("category")
fraud["gender"] = fraud["gender"].astype("category")
fraud["merchant"] = fraud["merchant"].astype("category")
fraud["category"] = fraud["category"].astype("category")
fraud["amount"] = fraud["amount"].astype(float)
fraud["fraud"] = fraud["fraud"].astype("category")

# One-hot encoding for categorical variables
fraud_encoded = pd.get_dummies(fraud, columns=["age", "gender", "merchant", "category"], drop_first=True)
fraud_encoded = fraud_encoded.drop(['customer'], axis=1)

# Standardize the data (excluding the 'fraud' column)
scaler = StandardScaler()
X = fraud_encoded.drop(['fraud'], axis=1)
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Add 'fraud' column back
X_scaled['fraud'] = fraud_encoded['fraud']

# Address class imbalance using SMOTE
X_balanced, y_balanced = SMOTE().fit_resample(X_scaled.drop('fraud', axis=1), X_scaled['fraud'])

# Split Data into Train-Test
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.4, random_state=42, stratify=y_balanced)

# K-Means Clustering (example, to explore patterns in the dataset)
kmeans = KMeans(n_clusters=4, random_state=42)
X_train['customerGroup'] = kmeans.fit_predict(X_train)

# EDA Visualizations
# Set color palette
custom_colors = ['#9facee', '#f1cad5']

# 1. Transaction Amount Distribution
plt.figure(figsize=(10, 6))
sns.histplot(fraud['amount'], kde=False, color=custom_colors[0], bins=50)
plt.title('Transaction Amount Distribution', fontsize=14)
plt.xlabel('Amount', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.grid(True)
plt.show()

# 2. Fraud Count by Category
plt.figure(figsize=(12, 6))
fraud_by_category = fraud.groupby(['category', 'fraud']).size().unstack()
fraud_by_category.plot(kind='bar', stacked=True, color=custom_colors)
plt.title('Fraud Count by Category', fontsize=14)
plt.xlabel('Category', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.grid(True)
plt.show()

# 3. Transaction Count by Gender
plt.figure(figsize=(10, 6))
sns.countplot(data=fraud, x='gender', palette=custom_colors)
plt.title('Transaction Count by Gender', fontsize=14)
plt.xlabel('Gender', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.grid(True)
plt.show()

# 4. Fraud Count by Transaction Amount
plt.figure(figsize=(10, 6))
sns.boxplot(data=fraud, x='fraud', y='amount', palette=custom_colors)
plt.title('Fraud Count by Transaction Amount', fontsize=14)
plt.xlabel('Fraud (0 = No, 1 = Yes)', fontsize=12)
plt.ylabel('Transaction Amount', fontsize=12)
plt.grid(True)
plt.show()

# 5. Fraud Count by Age Group
plt.figure(figsize=(10, 6))
fraud['age'] = pd.to_numeric(fraud['age'], errors='coerce')
age_bins = [18, 25, 35, 45, 55, 65, 100]
fraud['age_group'] = pd.cut(fraud['age'], bins=age_bins, labels=['18-24', '25-34', '35-44', '45-54', '55-64', '65+'])

sns.countplot(data=fraud, x='age_group', hue='fraud', palette=custom_colors)
plt.title('Fraud Count by Age Group', fontsize=14)
plt.xlabel('Age Group', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.grid(True)
plt.show()

# 1. Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)
print("Naive Bayes - Classification Report:\n", classification_report(y_test, y_pred_gnb))
print("Naive Bayes - ROC AUC Score: ", roc_auc_score(y_test, y_pred_gnb))

# 2. Logistic Regression with Hyperparameter Tuning
log_reg = LogisticRegression()
param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
grid_log_reg = GridSearchCV(log_reg, param_grid, cv=5)
grid_log_reg.fit(X_train, y_train)
y_pred_lr = grid_log_reg.predict(X_test)
print("Logistic Regression - Best Params: ", grid_log_reg.best_params_)
print("Logistic Regression - Classification Report:\n", classification_report(y_test, y_pred_lr))
print("Logistic Regression - ROC AUC Score: ", roc_auc_score(y_test, y_pred_lr))

# 3. Neural Network Classifier
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
mlp.fit(X_train, y_train)
y_pred_nn = mlp.predict(X_test)
print("Neural Network - Classification Report:\n", classification_report(y_test, y_pred_nn))
print("Neural Network - ROC AUC Score: ", roc_auc_score(y_test, y_pred_nn))

# 4. Random Forest with Hyperparameter Tuning
rf = RandomForestClassifier(random_state=42)
param_grid_rf = {'n_estimators': [100, 200, 300], 'max_depth': [5, 10, 15]}
grid_rf = GridSearchCV(rf, param_grid_rf, cv=5)
grid_rf.fit(X_train, y_train)
y_pred_rf = grid_rf.predict(X_test)
print("Random Forest - Best Params: ", grid_rf.best_params_)
print("Random Forest - Classification Report:\n", classification_report(y_test, y_pred_rf))
print("Random Forest - ROC AUC Score: ", roc_auc_score(y_test, y_pred_rf))

# 5. Gradient Boosting Classifier
gbc = GradientBoostingClassifier(random_state=42)
gbc.fit(X_train, y_train)
y_pred_gbc = gbc.predict(X_test)
print("Gradient Boosting - Classification Report:\n", classification_report(y_test, y_pred_gbc))
print("Gradient Boosting - ROC AUC Score: ", roc_auc_score(y_test, y_pred_gbc))

# Confusion Matrix for Random Forest
conf_matrix = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
