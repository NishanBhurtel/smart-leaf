"""
Test to exactly replicate Colab's prediction workflow
"""
import cv2
import numpy as np
from tensorflow import keras
import sys

# Load model
model = keras.models.load_model("my_model.keras")
print("✅ Model loaded")

# Class names (alphabetically sorted)
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

# Test image
image_path = r"E:\CNN_project\splitted_dataset\test\Potato___Late_blight\5e356292-3046-4967-b1e7-f3e5cf7d0f75___RS_LB 4459.JPG"

print(f"\nTesting: {image_path}")
print(f"Expected: Potato___Late_blight (index 5)\n")

# EXACTLY like Colab notebook
img = cv2.imread(image_path)
print(f"After cv2.imread: shape={img.shape}, dtype={img.dtype}, range=[{img.min()}, {img.max()}]")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(f"After BGR→RGB: shape={img.shape}, dtype={img.dtype}, range=[{img.min()}, {img.max()}]")

img = cv2.resize(img, (256, 256))
print(f"After resize: shape={img.shape}, dtype={img.dtype}, range=[{img.min()}, {img.max()}]")

img_array = np.expand_dims(img, axis=0)
print(f"After expand_dims: shape={img_array.shape}, dtype={img_array.dtype}\n")

# Predict
pred = model.predict(img_array, verbose=0)
pred_class = np.argmax(pred)

print(f"Predicted class index: {pred_class}")
print(f"Predicted class name: {class_names[pred_class]}")
print(f"Confidence: {pred[0][pred_class]*100:.2f}%")

print(f"\nTop 5:")
top_indices = np.argsort(pred[0])[::-1][:5]
for i, idx in enumerate(top_indices, 1):
    print(f"  {i}. {class_names[idx]}: {pred[0][idx]*100:.2f}%")

if class_names[pred_class] == "Potato___Late_blight":
    print("\n✅ CORRECT!")
else:
    print(f"\n❌ WRONG! Expected Potato___Late_blight, got {class_names[pred_class]}")
