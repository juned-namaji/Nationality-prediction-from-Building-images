import cv2
import joblib
from skimage.feature import hog
from skimage import exposure
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import easyocr
import threading
import time

# Load the pre-trained PCA model
pca_model = joblib.load('pca_model.pkl')

# Load the pre-trained feature selector model
feature_selector = joblib.load('feature_selector.pkl')

# Load the pre-trained Random Forest classifier
rf_classifier = joblib.load('random_forest_model.pkl')

max_probability = 0

# Load EasyOCR models
readers = {
    'en': easyocr.Reader(['en']),
    'is': easyocr.Reader(['is']),
    'ja': easyocr.Reader(['ja']),
    'ar': easyocr.Reader(['ar'])
}

# Function to process the input image
def process_input_image(input_image_path):
    input_image = cv2.imread(input_image_path)
    if input_image is None:
        return None, None  # Return None if the image cannot be loaded

    if input_image.dtype != np.uint8:
        input_image = cv2.convertScaleAbs(input_image)

    target_width = 224
    target_height = 224
    resized_image = cv2.resize(input_image, (target_width, target_height))
    grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(grayscale_image, 100, 200)
    features_hog = hog(edges, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
    features_hog = exposure.equalize_adapthist(features_hog)

    return resized_image, features_hog

# Function to extract text from an image using EasyOCR
def extract_text(input_image_path):
    input_image = cv2.imread(input_image_path)
    if input_image is None:
        return None, None

    detected_languages = []
    extracted_texts = []

    for lang, reader in readers.items():
        results = reader.readtext(input_image)
        if results:
            extracted_text = ' '.join([result[1] for result in results])
            detected_languages.append(lang)
            extracted_texts.append(extracted_text)

    return detected_languages, extracted_texts

# Function to calculate the confidence of language detection
def calculate_confidence(detected_languages):
    total_languages = len(detected_languages)
    lang_counts = {lang: detected_languages.count(lang) for lang in set(detected_languages)}
    language_confidences = {lang: (lang_counts[lang] / total_languages) * 100 for lang in set(detected_languages)}

    return language_confidences

# Function to process the image and return the combined result
def process_image(input_image_path):
    start_time = time.time()  # Record the start time
    detected_languages, extracted_texts = extract_text(input_image_path)

    if detected_languages:
        # Calculate the confidence of language detection
        language_confidences = calculate_confidence(detected_languages)

        max_confidence_language = max(language_confidences, key=language_confidences.get)

        ml_nationality, ml_resized_image, ml_confidence = process_ml(input_image_path)
        text_nationality = choose_text_nationality(max_confidence_language, extracted_texts)
        max_confidence = max(ml_confidence, language_confidences[max_confidence_language])

        if max_confidence > 35:  # You can adjust this threshold
            if ml_confidence > max_confidence:
                prediction_result, resized_image = ml_nationality, ml_resized_image
            else:
                prediction_result, resized_image = text_nationality, ml_resized_image
        else:
            prediction_result, resized_image = "Location is out of range!!", None

        # Record prediction time
        elapsed_time = measure_prediction_time(start_time)
        prediction_result_with_time = f"{prediction_result} (Time: {elapsed_time:.2f} seconds)"

        update_gui_with_result(prediction_result_with_time, resized_image, elapsed_time)

def measure_prediction_time(start_time):
    end_time = time.time()
    elapsed_time = end_time - start_time
    return elapsed_time

def update_gui_with_result(result, image, elapsed_time):
    label.config(text=result)
    if image is not None:
        display_image(image)
    confidence_label.config(text=f"Time taken for prediction: {elapsed_time:.2f} seconds")
    reset_loading_message()

# Function to reset the loading message and button state
def reset_loading_message():
    loading_label.config(text="")
    button.config(state="active")  # Enable the button after processing

# Function to display the image on the GUI
def display_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    photo = ImageTk.PhotoImage(image=image)
    image_label.config(image=photo)
    image_label.image = photo

# Function to update the loading message
def update_loading_message():
    loading_label.config(text="Processing the image... Please wait.")
    button.config(state="disabled")  # Disable the button while processing

# Function to perform image processing in a separate thread
def process_image_thread():
    update_loading_message()  # Show loading message
    input_image_path = filedialog.askopenfilename()
    thread = threading.Thread(target=process_image, args=(input_image_path,))
    thread.start()

# Create the main GUI window
window = tk.Tk()
window.title("Nationality Prediction")

window.geometry("800x700")  # Set the initial window size

# Configure row and column weights for center alignment
window.grid_rowconfigure(0, weight=1)
window.grid_rowconfigure(1, weight=1)
window.grid_rowconfigure(2, weight=1)
window.grid_rowconfigure(3, weight=1)
window.grid_columnconfigure(0, weight=1)

# Create a label for loading message
loading_label = tk.Label(window, text="", font=("Helvetica", 12))
loading_label.grid(row=0, column=0, pady=20)

# Create a label for displaying the image with an even larger size
image_label = tk.Label(window)
image_label.grid(row=2, column=0, pady=20)  # Use grid manager and adjust pady

# Create labels for predicted nationality and confidence level
label = tk.Label(window, text="", font=("Helvetica", 16))
label.grid(row=3, column=0, pady=20)  # Use grid manager and adjust pady

# Create a button for image selection with optimized size
button = tk.Button(window, text="Select Image", command=process_image_thread, width=15, height=2)
button.grid(row=1, column=0, pady=20)  # Use grid manager and adjust pady

confidence_label = tk.Label(window, text="", font=("Helvetica", 16))
confidence_label.grid(row=4, column=0, pady=20

)  # Use grid manager and adjust pady

window.mainloop()