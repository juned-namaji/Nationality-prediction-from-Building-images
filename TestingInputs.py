import cv2
import joblib
from skimage.feature import hog
from skimage import exposure
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import threading

pca_model = joblib.load("pca_model.pkl")

feature_selector = joblib.load("feature_selector.pkl")

xgboost_classifier = joblib.load("xgboost_model.pkl")

max_probability = 0


def process_input_image(input_image_path):
    input_image = cv2.imread(input_image_path)
    if input_image is None:
        return None, None

    if input_image.dtype != np.uint8:
        input_image = cv2.convertScaleAbs(input_image)

    target_width = 224
    target_height = 224
    resized_image = cv2.resize(input_image, (target_width, target_height))
    grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(grayscale_image, 100, 200)
    features_hog = hog(
        edges,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
    )
    features_hog = exposure.equalize_adapthist(features_hog)

    return resized_image, features_hog


def predict_nationality(input_image_path):
    global max_probability

    resized_image, input_features = process_input_image(input_image_path)

    if input_features is None:
        return (
            "Can't identify nation from the image. Please provide a better image",
            None,
        )

    input_features_pca = pca_model.transform([input_features])

    input_features_selected = feature_selector.transform(input_features_pca)

    predicted_label = xgboost_classifier.predict(input_features_selected)
    probability_estimates = xgboost_classifier.predict_proba(input_features_selected)

    confidence_threshold = 0.60

    max_probability = probability_estimates.max()
    if max_probability > confidence_threshold:
        reverse_label_mapping = {
            "0": "Iceland",
            "1": "Japan",
            "2": "Saudi",
            "3": "Italy",
        }
        predicted_nationality = reverse_label_mapping[str(predicted_label[0])]

        return predicted_nationality, resized_image
    else:
        return (
            "Can't identify nation from the image. Please provide a better image",
            resized_image,
        )


window = tk.Tk()
window.title("NATIONALITY PREDICTION")  # Set the title here
window.geometry("800x700")

window.grid_rowconfigure(0, weight=1)
window.grid_rowconfigure(1, weight=1)
window.grid_rowconfigure(2, weight=1)
window.grid_rowconfigure(3, weight=1)
window.grid_columnconfigure(0, weight=1)

title_label = tk.Label(window, text="NATIONALITY PREDICTION", font=("Helvetica", 20))
title_label.grid(row=0, column=0, pady=10)


def update_loading_message():
    loading_label.config(text="Processing the image... Please wait.")
    button.config(state="disabled")


def reset_loading_message():
    loading_label.config(text="")
    button.config(state="active")


def process_image_thread():
    update_loading_message()
    input_image_path = filedialog.askopenfilename()
    thread = threading.Thread(target=process_image, args=(input_image_path,))
    thread.start()


def process_image(input_image_path):
    predicted_nationality, resized_image = predict_nationality(input_image_path)
    reset_loading_message()

    if predicted_nationality:
        confidence_level = max_probability * 100

        img = ImageTk.PhotoImage(
            Image.fromarray(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
        )

        image_label.config(image=img)
        image_label.image = img

        if confidence_level > 35:
            label.config(text=f"Predicted nationality: {predicted_nationality}")
            confidence_label.config(text=f"Confidence: {confidence_level:.2f}%")
        else:
            label.config(
                text="Can't identify nation from the image. Please provide a better image"
            )
            confidence_label.config(text="")

    else:
        label.config(
            text="Can't identify nation from the image. Please provide a better image"
        )
        confidence_label.config(text="")


loading_label = tk.Label(window, text="", font=("Helvetica", 12))
loading_label.grid(row=1, column=0, pady=10)

button = tk.Button(
    window, text="Select Image", command=process_image_thread, width=15, height=2
)
button.grid(row=2, column=0, pady=10)

image_label = tk.Label(window)
image_label.grid(row=3, column=0, pady=10)

label = tk.Label(window, text="", font=("Helvetica", 16))
label.grid(row=4, column=0, pady=10)

confidence_label = tk.Label(window, text="", font=("Helvetica", 16))
confidence_label.grid(row=5, column=0, pady=10)

window.mainloop()
