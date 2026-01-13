"""Test with different dtypes to find what works"""
import cv2
import numpy as np
from tensorflow import keras

model = keras.models.load_model("plant_disease_model.keras")
class_names = [
    "Pepper__bell___Bacterial_spot", "Pepper__bell___healthy", "PlantVillage",
    "Potato___Early_blight", "Potato___healthy", "Potato___Late_blight",
    "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_healthy",
    "Tomato_Late_blight", "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite", "Tomato__Target_Spot",
    "Tomato__Tomato_mosaic_virus", "Tomato__Tomato_YellowLeaf__Curl_Virus",
]

image_path = r"E:\CNN_project\splitted_dataset\test\Potato___Late_blight\5e356292-3046-4967-b1e7-f3e5cf7d0f75___RS_LB 4459.JPG"

# Load image
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (256, 256))

print("Testing different data types and normalizations:\n")

# Test 1: uint8, no normalization [0-255]
print("1. uint8, range [0-255] (current approach):")
img_test = np.expand_dims(img, axis=0)
pred = model.predict(img_test, verbose=0)[0]
idx = np.argmax(pred)
print(f"   → {class_names[idx]} ({pred[idx]*100:.2f}%)\n")

# Test 2: float32, no normalization [0-255]
print("2. float32, range [0-255]:")
img_test = np.expand_dims(img.astype(np.float32), axis=0)
pred = model.predict(img_test, verbose=0)[0]
idx = np.argmax(pred)
print(f"   → {class_names[idx]} ({pred[idx]*100:.2f}%)\n")

# Test 3: float32, normalized [0-1]
print("3. float32, normalized [0-1]:")
img_test = np.expand_dims(img.astype(np.float32) / 255.0, axis=0)
pred = model.predict(img_test, verbose=0)[0]
idx = np.argmax(pred)
print(f"   → {class_names[idx]} ({pred[idx]*100:.2f}%)\n")

# Test 4: float32, normalized [-1,1]
print("4. float32, normalized [-1,1]:")
img_test = np.expand_dims((img.astype(np.float32) / 127.5) - 1.0, axis=0)
pred = model.predict(img_test, verbose=0)[0]
idx = np.argmax(pred)
print(f"   → {class_names[idx]} ({pred[idx]*100:.2f}%)\n")

print(f"Expected: Potato___Late_blight (index 5)")
