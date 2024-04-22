import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier  # Import DecisionTreeClassifier
from sklearn.metrics import classification_report
import joblib

data = pd.read_csv('updated_labels_and_features.csv')

X = data.drop('Label', axis=1)
y = data['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt_classifier = DecisionTreeClassifier(random_state=42)

# Train the classifier on the training data
dt_classifier.fit(X_train, y_train)

# Save the trained model to a file
model_filename = 'decision_tree_model.pkl'
joblib.dump(dt_classifier, model_filename)

# Make predictions on the test data
y_pred = dt_classifier.predict(X_test)

# Evaluate the model and print the classification report
classification_rep = classification_report(y_test, y_pred, output_dict=True)

# Extract and print the average F1 score, accuracy, precision, and recall
avg_metrics = classification_rep['macro avg']
print(f'Average F1: {avg_metrics["f1-score"]}, Accuracy: {avg_metrics["precision"]}, Average Precision: {avg_metrics["precision"]}, Average Recall: {avg_metrics["recall"]}')