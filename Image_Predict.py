import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the MobileNetV2 model pre-trained on ImageNet
model = MobileNetV2(weights="imagenet")

# Load and preprocess an image
image_path = "pic.jpg"  # Replace with the path to your image
image = load_img(image_path, target_size=(224, 224))  # Resize image to 224x224 pixels
image_array = img_to_array(image)  # Convert to a NumPy array
image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
image_array = preprocess_input(image_array)  # Preprocess the image for the model

# Perform prediction
predictions = model.predict(image_array)

# Decode the predictions to get class names and confidence scores
decoded_predictions = decode_predictions(predictions, top=3)  # Get top 3 predictions
print("Predictions:")
for i, (imagenet_id, label, score) in enumerate(decoded_predictions[0]):
    print(f"{i + 1}: {label} ({score:.2f})")
