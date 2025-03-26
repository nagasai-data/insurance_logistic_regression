# Import essential libraries
import pandas as pd          # For handling and manipulating the dataset
import numpy as np           # For numerical computations
import matplotlib.pyplot as plt  # For plotting graphs and visualizations
import seaborn as sns        # For beautiful and informative plots
# Load the dataset
data = pd.read_csv("train.csv")  # Make sure 'train.csv' is in the same folder

# Display the first 5 rows of the dataset
print("First 5 rows of the dataset:")
print(data.head())

# Display the last 5 rows of the dataset
print("\nLast 5 rows of the dataset:")
print(data.tail())

# Display the shape of the dataset (rows, columns)
print("\nShape of the dataset:")
print(data.shape)
# Check for missing values in the dataset
print("\nMissing values in each column:")
print(data.isnull().sum())

# Convert categorical features to numerical using label encoding
data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
data['Vehicle_Damage'] = data['Vehicle_Damage'].map({'Yes': 1, 'No': 0})
data['Vehicle_Age'] = data['Vehicle_Age'].map({'> 2 Years': 2, '1-2 Year': 1, '< 1 Year': 0})

# Drop columns not useful for prediction
data = data.drop(columns=['id'])

# Define features (X) and target (y)
X = data.drop(columns='Response')   # All features except target
y = data['Response']                # Target column

print("\n Data cleaning and preparation completed.")
# Block 3: Train a Logistic Regression model

from sklearn.model_selection import train_test_split   # To split the dataset
from sklearn.linear_model import LogisticRegression    # Our classification model
from sklearn.metrics import accuracy_score             # To evaluate model performance

# Step 1: Separate features and target
X = data.drop(columns=[ 'Response'])  # Features (everything except ID and target)
y = data['Response']                       # Target column we want to predict

# Step 2: Split the dataset into training and testing sets
# test_size=0.2 means 80% training, 20% testing
# random_state ensures reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Create and train the logistic regression model
model = LogisticRegression(max_iter=1000)  # max_iter helps in case convergence takes longer
model.fit(X_train, y_train)

# Step 4: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 5: Check the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy on Test Set: {accuracy:.4f}")
# 🧠 Step 4: Evaluate model using confusion matrix and classification report

from sklearn.metrics import confusion_matrix, classification_report

# Predict on test set
y_pred = model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Classification Report (as a DataFrame for better display)
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Print classification report
print("\n Classification Report:")
print(report_df)

# Plot confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

