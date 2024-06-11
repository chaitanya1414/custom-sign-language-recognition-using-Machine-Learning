import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Load the dataset
df = pd.read_csv('sign_language_dataset_manual.csv')

# Display the first few rows of the dataframe to verify
print(df.head())

# Extract features (hand landmarks) and labels (gestures)
X = df['landmarks'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))
y = df['label']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shapes to verify
print(f"Training set: X_train shape = {X_train.shape}, y_train shape = {y_train.shape}")
print(f"Testing set: X_test shape = {X_test.shape}, y_test shape = {y_test.shape}")

# Perform feature scaling if needed (for example, StandardScaler for SVM or neural networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(np.vstack(X_train))
X_test_scaled = scaler.transform(np.vstack(X_test))

# Example: Train a Random Forest Classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(np.vstack(X_train), y_train)

# Evaluate the model
accuracy = clf.score(np.vstack(X_test), y_test)
print(f"Accuracy on test set: {accuracy:.2f}")

# Save the model to disk
joblib.dump(clf, 'sign_language_classifier.pkl')

print("Model training and evaluation completed. Model saved as 'sign_language_classifier.pkl'.")
