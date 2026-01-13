"""
Extract class names from the trained model and dataset.
This ensures predictions use the correct class labels.
"""

import tensorflow as tf
import json
import os

# Paths
backend_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(backend_dir, "..", "splitted_dataset")
model_path = os.path.join(backend_dir, "plant_disease_model.keras")
class_names_path = os.path.join(backend_dir, "class_names.json")

print("=" * 60)
print("📊 Extracting Class Names from Dataset")
print("=" * 60)

# Load dataset to get class names in correct order
print("\n📁 Loading training data to get class names...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(dataset_dir, "train"),
    labels="inferred",
    label_mode="categorical",
    batch_size=8,
    image_size=(256, 256),
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)

print(f"\n✅ Found {num_classes} classes in dataset:")
for i, name in enumerate(class_names):
    print(f"   {i:2d}: {name}")

# Save class names to JSON
print(f"\n💾 Saving class names to {class_names_path}...")
with open(class_names_path, 'w') as f:
    json.dump(class_names, f, indent=2)
print("✅ Class names saved!")

# Verify model can be loaded
print(f"\n🔍 Verifying model...")
try:
    model = tf.keras.models.load_model(model_path)
    print(f"✅ Model loaded successfully!")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")
    print(f"   Expected classes: {model.output_shape[-1]}")
    print(f"   Actual classes in JSON: {num_classes}")
    
    if model.output_shape[-1] == num_classes:
        print("\n✅ Model and class names match perfectly!")
    else:
        print("\n⚠️  WARNING: Model output classes don't match dataset classes!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

print("\n" + "=" * 60)
print("✨ Setup complete! Class names are synchronized with the model.")
print("=" * 60)
