import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import classification_report
import joblib

# Load your dataset
data = pd.read_csv("updated_labels_and_features.csv")

# Drop rows with NaN values in the 'Label' column
data.dropna(subset=["Label"], inplace=True)

# Convert 'Label' column to integers
data["Label"] = data["Label"].astype(int)

X = data.drop("Label", axis=1)
y = data["Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

xgb_classifier = xgb.XGBClassifier()

xgb_classifier.fit(X_train, y_train)

model_filename = "xgboost_model.pkl"
joblib.dump(xgb_classifier, model_filename)

y_pred = xgb_classifier.predict(X_test)

classification_rep = classification_report(y_test, y_pred, output_dict=True)

avg_metrics = classification_rep["macro avg"]
print(
    f'Average F1: {avg_metrics["f1-score"]}, Accuracy: {avg_metrics["precision"]}, Average Precision: {avg_metrics["precision"]}, Average Recall: {avg_metrics["recall"]}'
)
