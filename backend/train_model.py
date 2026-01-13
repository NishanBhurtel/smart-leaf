"""
Train the plant disease detection CNN model.
This script loads the training data and trains a model to replace the mock model.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import json

# Define paths
backend_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(backend_dir, "..", "splitted_dataset")
model_output_path = os.path.join(backend_dir, "plant_disease_model.keras")
class_names_path = os.path.join(backend_dir, "class_names.json")

# Configuration
IMG_SIZE = (256, 256)
BATCH_SIZE = 8
EPOCHS = 20

print("=" * 60)
print("🌿 Plant Disease Model Training")
print("=" * 60)

# Load training data
print("\n📁 Loading training data...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(dataset_dir, "train"),
    labels="inferred",
    label_mode="categorical",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True
)

print("\n📁 Loading validation data...")
val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(dataset_dir, "val"),
    labels="inferred",
    label_mode="categorical",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True
)

# Get class names from the training dataset
class_names = train_ds.class_names
num_classes = len(class_names)

print(f"\n✅ Found {num_classes} classes:")
for i, name in enumerate(class_names):
    print(f"  {i:2d}: {name}")

# Save class names to JSON
print(f"\n💾 Saving class names to {class_names_path}...")
with open(class_names_path, 'w') as f:
    json.dump(class_names, f, indent=2)
print("✅ Class names saved!")

# Build the model
print("\n🏗️  Building the CNN model...")
model = Sequential([
    Conv2D(32, (3, 3), input_shape=(256, 256, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(128, (3, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(256, (3, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(512, (3, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Dropout(0.25),
    Flatten(),
    Dense(512),
    Activation('relu'),
    
    Dropout(0.5),
    Dense(num_classes),
    Activation('softmax')
])

# Compile the model
print("\n⚙️  Compiling the model...")
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.0001),
    metrics=['accuracy']
)

print("\n📊 Model summary:")
model.summary()

# Set up callbacks
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6),
]

# Train the model
print("\n🚀 Training the model...")
print(f"   Epochs: {EPOCHS}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Training samples: {len(train_ds) * BATCH_SIZE}")
print(f"   Validation samples: {len(val_ds) * BATCH_SIZE}")

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=callbacks,
    verbose=1
)

# Evaluate on test set if available
test_dir = os.path.join(dataset_dir, "test")
if os.path.exists(test_dir):
    print("\n📁 Loading test data...")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="categorical",
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        shuffle=False
    )
    
    print("\n🧪 Evaluating on test set...")
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"   Test Accuracy: {test_acc:.4f}")
    print(f"   Test Loss: {test_loss:.4f}")

# Save the model
print(f"\n💾 Saving trained model to {model_output_path}...")
model.save(model_output_path)
print("✅ Model trained and saved successfully!")

print("\n" + "=" * 60)
print("✨ Training complete! Restart your Flask app to use the new model.")
print("=" * 60)
