"""Test script to verify model predictions match training"""
import numpy as np
import cv2
from tensorflow import keras
import os

# Load model
model_path = "my_model.keras"
print(f"Loading model from: {model_path}")
model = keras.models.load_model(model_path)
print("✅ Model loaded successfully!")
print(f"Model input shape: {model.input_shape}")
print(f"Model output shape: {model.output_shape}")

# Class names - alphabetically sorted folder names from train directory
class_names = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "PlantVillage",
    "Potato___Early_blight",
    "Potato___healthy",
    "Potato___Late_blight",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_mosaic_virus",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
]

print(f"\nTotal classes: {len(class_names)}")
print("\nClass names:")
for i, name in enumerate(class_names):
    print(f"  {i}: {name}")

# Test image - Potato Late Blight
test_image_path = r"E:\CNN_project\splitted_dataset\test\Potato___Late_blight\5e356292-3046-4967-b1e7-f3e5cf7d0f75___RS_LB 4459.JPG"

print(f"\n{'='*80}")
print(f"Testing image: {os.path.basename(test_image_path)}")
print(f"Expected class: Potato___Late_blight (index {class_names.index('Potato___Late_blight')})")
print(f"{'='*80}")

# Load and preprocess image (EXACTLY like the CNN notebook)
img = cv2.imread(test_image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR to RGB
img = cv2.resize(img, (256, 256))
print(f"\nImage shape after resize: {img.shape}")
print(f"Image dtype: {img.dtype}")
print(f"Image value range: [{img.min()}, {img.max()}]")

# NO normalization - use raw pixel values [0-255]
img_array = np.expand_dims(img, axis=0)
print(f"Final array shape: {img_array.shape}")

# Predict
print("\nMaking prediction...")
predictions = model.predict(img_array, verbose=0)[0]
print(f"Predictions shape: {predictions.shape}")
print(f"Predictions sum: {predictions.sum():.6f}")
print(f"Predictions min: {predictions.min():.6f}")
print(f"Predictions max: {predictions.max():.6f}")

# Get top 5
pred_index = np.argmax(predictions)
predicted_class = class_names[pred_index]
confidence = predictions[pred_index] * 100

print(f"\n{'='*80}")
print(f"🎯 PREDICTION RESULT:")
print(f"{'='*80}")
print(f"Predicted class: {predicted_class} (index {pred_index})")
print(f"Confidence: {confidence:.2f}%")
print(f"\nTop 5 predictions:")
top_indices = np.argsort(predictions)[::-1][:5]
for i, idx in enumerate(top_indices, 1):
    print(f"  {i}. {class_names[idx]}: {predictions[idx]*100:.2f}%")

print(f"\n{'='*80}")
if predicted_class == "Potato___Late_blight":
    print("✅ CORRECT! Model predicted the right class!")
else:
    print("❌ WRONG! Model predicted incorrectly!")
    print(f"   Expected: Potato___Late_blight")
    print(f"   Got: {predicted_class}")
print(f"{'='*80}")
