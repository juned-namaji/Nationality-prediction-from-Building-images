import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  # Import Support Vector Classifier
from sklearn.metrics import classification_report
import joblib

# Load your dataset
data = pd.read_csv('updated_labels_and_features.csv')  # Replace with your dataset file

# Split the dataset into features (X) and the target variable (y)
X = data.drop('Label', axis=1)
y = data['Label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Support Vector Machine (SVM) classifier
svm_classifier = SVC()  # Create a Support Vector Classifier

# Train the classifier on the training data
svm_classifier.fit(X_train, y_train)

# Save the trained model to a file
model_filename = 'svm_model.pkl'
joblib.dump(svm_classifier, model_filename)

# Make predictions on the test data
y_pred = svm_classifier.predict(X_test)

# Evaluate the model and print the classification report
classification_rep = classification_report(y_test, y_pred, output_dict=True)

# Extract and print the average F1 score, accuracy, precision, and recall
avg_metrics = classification_rep['macro avg']
print(f'Average F1: {avg_metrics["f1-score"]}, Accuracy: {avg_metrics["precision"]}, Average Precision: {avg_metrics["precision"]}, Average Recall: {avg_metrics["recall"]}')