import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "C:/Users/91930/OneDrive/Documents/CV/CP/images/saudi_7.jpg"
I = cv2.imread(image_path)

# Display the original image
plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(I, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")
plt.show()

I_gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)  # Convert to grayscale


msk = np.zeros((3, 3, 8))
msk[:, :, 0] = np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]])  # East
msk[:, :, 1] = np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]])  # North-east
msk[:, :, 2] = np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]])  # North
msk[:, :, 3] = np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]])  # North-west
msk[:, :, 4] = np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]])  # West
msk[:, :, 5] = np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]])  # South-west
msk[:, :, 6] = np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]])  # South
msk[:, :, 7] = np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]])  # South-east

# Apply Kirsch masks using filter2D
ldp = np.zeros_like(I_gray, dtype=np.uint16)  # Use uint16 for LDP accumulation
for i in range(8):
    mask = msk[:, :, i]
    convolved = cv2.filter2D(I_gray, -1, mask)
    intermediate_result = (convolved >= 0) * (2**i)
    ldp += intermediate_result.astype(
        np.uint16
    )  # Convert intermediate result to uint16

# Display LDP image
plt.figure(figsize=(6, 6))
plt.imshow(ldp, cmap="gray")
plt.title("LDP Image")
plt.axis("off")
plt.show()

# LDP Histogram
H = cv2.calcHist([ldp.astype(np.uint8)], [0], None, [256], [0, 256])
plt.figure()
plt.bar(range(256), H[:, 0])
plt.title("LDP Histogram")
plt.xlabel("Pixel value")
plt.ylabel("Frequency")
plt.show()
