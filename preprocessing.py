import cv2
import os
import pandas as pd
import numpy as np
from skimage.feature import hog, local_binary_pattern
from skimage import exposure
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import joblib

data_directory = "C:/Users/91930/OneDrive/Documents/CV/CP/images"
output_directory = "C:/Users/91930/OneDrive/Documents/CV/CP/processedImages"
hog_features_directory = "C:/Users/91930/OneDrive/Documents/CV/CP/hogFeatures"

if os.path.exists(output_directory):
    for file in os.listdir(output_directory):
        file_path = os.path.join(output_directory, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

if not os.path.exists(hog_features_directory):
    os.makedirs(hog_features_directory)

target_width = 224
target_height = 224

orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)

features = []
labels = []

max_descriptors = 50  # Maximum total descriptors
max_descriptors_per_method = max_descriptors // 2  # Equal division between HOG and LDP

hog_features_count = 0
ldp_features_count = 0

for filename in os.listdir(data_directory):
    image_path = os.path.join(data_directory, filename)

    image = cv2.imread(image_path)

    if image is None:
        print(f"Error loading image: {image_path}")
        continue

    if image.dtype != np.uint8:
        image = cv2.convertScaleAbs(image)

    resized_image = cv2.resize(image, (target_width, target_height))

    grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Compute HOG features
    features_hog = hog(
        grayscale_image,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm="L2-Hys",
        feature_vector=True,  # Ensure feature vector output
    )

    # Limit HOG descriptors to a maximum of max_descriptors_per_method
    if hog_features_count < max_descriptors_per_method:
        features_hog = features_hog[: max_descriptors_per_method - hog_features_count]
        hog_features_count += len(features_hog)
    else:
        features_hog = np.zeros_like(features_hog)  # No additional HOG features

    # Compute LDP features
    radius = 3
    n_points = 8 * radius
    ldp = local_binary_pattern(grayscale_image, n_points, radius, method="uniform")

    # Flatten the LDP array
    features_ldp = ldp.ravel()

    # Limit LDP descriptors to a maximum of max_descriptors_per_method
    if ldp_features_count < max_descriptors_per_method:
        features_ldp = features_ldp[: max_descriptors_per_method - ldp_features_count]
        ldp_features_count += len(features_ldp)
    else:
        features_ldp = np.zeros_like(features_ldp)  # No additional LDP features

    # Enhance the HOG image for better visualization
    hog_image = exposure.rescale_intensity(features_hog, in_range=(0, 10))
    hog_features, real = hog(grayscale_image, visualize=True)

    # Concatenate HOG and LDP features
    combined_features = np.concatenate((features_hog, features_ldp), axis=None)

    label = filename.split("_")[0]

    features.append(combined_features)
    labels.append(label)

    # Save the HOG image
    hog_image_path = os.path.join(hog_features_directory, f"{label}_{filename}_hog.jpg")
    cv2.imwrite(hog_image_path, (real * 255).astype(np.uint8))
data = pd.DataFrame(features)
data["Label"] = labels

from sklearn.impute import SimpleImputer

# Apply SimpleImputer to handle missing values
imputer = SimpleImputer(strategy="mean")
data_imputed = imputer.fit_transform(data.drop("Label", axis=1))

data_imputed = pd.DataFrame(data_imputed, columns=data.drop("Label", axis=1).columns)
data_imputed["Label"] = labels

# Continue with PCA and the rest of your code

# Apply PCA
n_components = 100
pca = PCA(n_components=n_components)

features_pca = pca.fit_transform(data_imputed.drop("Label", axis=1))

data_pca = pd.DataFrame(features_pca)
data_pca["Label"] = labels

csv_filename = "image_features_with_pca_ldp_hog.csv"
data_pca.to_csv(csv_filename, index=False)

print(f"Features with PCA extracted and saved to {csv_filename}")

selector = SelectKBest(score_func=f_classif, k=10)

X = data_pca.drop("Label", axis=1)
y = data_pca["Label"]

X_new = selector.fit_transform(X, y)

selected_feature_indices = selector.get_support(indices=True)

selected_features = X.columns[selected_feature_indices]

selected_data = data_pca[selected_features]
selected_data["Label"] = y

label_mapping = {"Iceland": "0", "Japan": "1", "saudi": "2", "Italy": "3"}

selected_data["Label"] = selected_data["Label"].map(label_mapping)

selected_data.to_csv("updated_labels_and_features_ldp_hog.csv", index=False)

joblib.dump(pca, "pca_model_ldp_hog.pkl")
joblib.dump(selector, "feature_selector_ldp_hog.pkl")

print("Selected features:")
print(selected_features)
