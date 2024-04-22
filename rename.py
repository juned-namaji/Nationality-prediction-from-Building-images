import matplotlib.pyplot as plt
from skimage import io, color, feature

# Load an example image
image_path = "C:/Users/91930/OneDrive/Documents/CV/CP/images/Iceland_9.jpg"  # Replace with your image path
original_image = io.imread(image_path)

# Convert the image to grayscale (HOG works on grayscale images)
gray_image = color.rgb2gray(original_image)

# Apply the HOG feature extractor
hog_features, hog_image = feature.hog(gray_image, visualize=True)

# Plot the original and HOG image side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(original_image, cmap=plt.cm.gray)
ax[0].set_title("Original Image")

ax[1].imshow(hog_image, cmap=plt.cm.gray)
ax[1].set_title("HOG Features")

plt.tight_layout()
plt.show()
