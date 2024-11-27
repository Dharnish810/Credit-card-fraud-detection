# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

# Load the dataset
file_path = r'C:\Users\Dharnish\Downloads\creditcardn.csv'  # Adjust the path as necessary
credit_data = pd.read_csv(file_path)

# Display basic information about the dataset
print(credit_data.info())
print(credit_data.describe())

# Check for missing values
print("Missing values in each column:\n", credit_data.isna().sum())

# Prepare features and target variable
X = credit_data.drop('Class', axis=1)  # Features
y = credit_data['Class']  # Target variable

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_scaled, y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.25, random_state=42)

# Train a classification model (Random Forest)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print('Accuracy for Random Forest Classifier: {:.2f}%'.format(accuracy_score(y_test, y_pred) * 100))
print('Classification Report:\n', classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Feature Importance
feat_imp = model.feature_importances_
feat = pd.DataFrame({'features': X.columns, 'Importance': feat_imp})
feat = feat.sort_values(by='Importance', ascending=False)

# Plot Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='features', data=feat)
plt.title('Feature Importance')
plt.show()

# Predicted class distribution
predicted_counts = pd.Series(y_pred).value_counts()
print("Predicted class distribution:\n", predicted_counts)
