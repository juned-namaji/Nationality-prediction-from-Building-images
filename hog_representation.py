import matplotlib.pyplot as plt
import pandas as pd

selected_data = pd.read_csv("updated_labels_and_features.csv")
# Assuming each class is represented as 0, 1, 2, 3, you can convert the Label column to categorical for better visualization
selected_data["Label"] = selected_data["Label"].astype("category")

# Define the label mapping
label_mapping = {"Iceland": 0, "Japan": 1, "saudi": 2, "Italy": 3}

# Grouping by class label and selecting the features
grouped_data = selected_data.groupby("Label")
features = selected_data.columns[1:10]  # Assuming the features start from column 1

plt.figure(figsize=(14, 8))
for feature in features:
    data_to_plot = [group[1][feature] for group in grouped_data]
    labels = [
        key
        for key, value in label_mapping.items()
        if value in grouped_data.groups.keys()
    ]
    plt.boxplot(data_to_plot, labels=labels)
    plt.title(f"{feature}th HOG feature Across Countries")
    plt.xlabel("Country Label")  # Change x-axis label
    plt.ylabel("Feature Value range")

    plt.show()
