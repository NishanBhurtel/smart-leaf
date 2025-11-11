"""
Test script to diagnose preprocessing and prediction issues
Compare direct model inference vs Flask API results
"""
import numpy as np
from PIL import Image
from tensorflow import keras
import os

# Load model
model_path = "my_model.keras"
print(f"Loading model from: {model_path}")
model = keras.models.load_model(model_path)
print("✅ Model loaded successfully!")

# Class names (must match training)
class_names = [
    "Pepper_bell_Bacterial_spot",
    "Pepper_bell_healthy",
    "PlantVillage",
    "Potato_Early_blight",
    "Potato_healthy",
    "Potato_Late_blight",
    "Tomato_Target_Spot",
    "Tomato_Tomato_mosaic_virus",
    "Tomato_Tomato_YellowLeaf_Curl_Virus",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septorial_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_leaf",
]

def preprocess_image_v1(image_path):
    """Current preprocessing (what Flask is using)"""
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((256, 256))
    img_array = np.array(image)
    img_array = img_array.astype('float32')
    img_array /= 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def preprocess_image_v2(image_path):
    """Alternative preprocessing (might match your CNN training better)"""
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((256, 256))
    img_array = np.array(image, dtype=np.float32)
    # Try different normalization
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def test_image(image_path):
    """Test an image with different preprocessing methods"""
    print(f"\n{'='*80}")
    print(f"Testing: {os.path.basename(image_path)}")
    print(f"{'='*80}")
    
    # Test V1 (current Flask method)
    print("\n📊 Method 1: Current Flask preprocessing")
    img_array_v1 = preprocess_image_v1(image_path)
    print(f"Shape: {img_array_v1.shape}")
    print(f"Dtype: {img_array_v1.dtype}")
    print(f"Range: [{img_array_v1.min():.4f}, {img_array_v1.max():.4f}]")
    
    predictions_v1 = model.predict(img_array_v1, verbose=0)[0]
    print(f"\nRaw predictions (first 5): {predictions_v1[:5]}")
    print(f"Raw predictions (shape): {predictions_v1.shape}")
    
    # Apply softmax
    exp_preds = np.exp(predictions_v1 - np.max(predictions_v1))
    probabilities = exp_preds / np.sum(exp_preds)
    
    pred_index = np.argmax(probabilities)
    predicted_class = class_names[pred_index]
    confidence = float(probabilities[pred_index] * 100)
    
    print(f"\n🎯 Prediction: {predicted_class}")
    print(f"🎯 Confidence: {confidence:.2f}%")
    
    # Show top 5
    top_indices = np.argsort(probabilities)[::-1][:5]
    print("\nTop 5 predictions:")
    for i, idx in enumerate(top_indices, 1):
        print(f"  {i}. {class_names[idx]}: {probabilities[idx]*100:.2f}%")
    
    # Check if model already has softmax
    print("\n🔍 Diagnostics:")
    print(f"Sum of raw predictions: {predictions_v1.sum():.4f}")
    if abs(predictions_v1.sum() - 1.0) < 0.01:
        print("⚠️  Model output already sums to 1 - it has softmax built-in!")
        print("   No need to apply softmax again!")
        
        # Recalculate without softmax
        pred_index_no_softmax = np.argmax(predictions_v1)
        predicted_class_no_softmax = class_names[pred_index_no_softmax]
        confidence_no_softmax = float(predictions_v1[pred_index_no_softmax] * 100)
        
        print(f"\n✅ Corrected prediction (no extra softmax):")
        print(f"   Class: {predicted_class_no_softmax}")
        print(f"   Confidence: {confidence_no_softmax:.2f}%")
        
        # Show top 5 without extra softmax
        top_indices_no_softmax = np.argsort(predictions_v1)[::-1][:5]
        print("\n   Top 5:")
        for i, idx in enumerate(top_indices_no_softmax, 1):
            print(f"     {i}. {class_names[idx]}: {predictions_v1[idx]*100:.2f}%")

if __name__ == "__main__":
    # Test with a Potato healthy image
    test_image_path = r"E:\CNN_project\splitted_dataset\test\Potato___healthy\0be9d721-82f5-42c3-b535-7494afe01dbe___RS_HL 1814.JPG"
    
    if os.path.exists(test_image_path):
        test_image(test_image_path)
    else:
        print(f"❌ Test image not found: {test_image_path}")
        print("Please update the test_image_path variable with a valid image path")
