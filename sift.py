import cv2
import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

data_directory = "C:/Users/91930/OneDrive/Documents/CV/CP/images"
output_directory = "C:/Users/91930/OneDrive/Documents/CV/CP/processedImages"
sift_features_directory = "C:/Users/91930/OneDrive/Documents/CV/CP/siftFeatures"

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

if not os.path.exists(sift_features_directory):
    os.makedirs(sift_features_directory)

sift = cv2.SIFT_create()

features = []
descriptors = []
labels = []

for filename in os.listdir(data_directory):
    image_path = os.path.join(data_directory, filename)

    image = cv2.imread(image_path)

    if image is None:
        print(f"Error loading image: {image_path}")
        continue

    resized_image = cv2.resize(image, (224, 224))
    grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Apply preprocessing steps (e.g., denoising, smoothing, etc.) if needed
    # ...

    # Edge detection
    edges = cv2.Canny(grayscale_image, 100, 200)

    # Compute SIFT features on the edge-detected image
    keypoints, descriptors_sift = sift.detectAndCompute(edges, None)

    label = filename.split("_")[0]

    # Save the SIFT-applied images in the output directory
    sift_image_path = os.path.join(output_directory, f"sift_{filename[:-4]}.png")
    cv2.imwrite(
        sift_image_path,
        cv2.drawKeypoints(
            edges,
            keypoints,
            outImage=np.array([]),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        ),
    )

    if descriptors_sift is not None:
        features.append(descriptors_sift.flatten())
        descriptors.append(descriptors_sift)
        labels.append(label)
    else:
        features.append(np.zeros((128,)))  # If no descriptors are found, add zeros

# Saving features and descriptors to CSV
data_sift = pd.DataFrame(features)
data_sift["Label"] = labels

data_sift.to_csv(
    os.path.join(sift_features_directory, "sift_features.csv"), index=False
)
