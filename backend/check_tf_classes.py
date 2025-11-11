"""Check exact class order from TensorFlow dataset"""
import tensorflow as tf

output_folder = r"E:\smart-leaf\splitted_dataset"
img_size = (256, 256)
batch_size = 8

train_ds = tf.keras.utils.image_dataset_from_directory(
    output_folder + "/train",
    labels="inferred",
    label_mode="categorical",
    batch_size=batch_size,
    image_size=img_size,
    shuffle=False
)

print("Class names from TensorFlow image_dataset_from_directory:")
print(f"Total classes: {len(train_ds.class_names)}\n")
for i, name in enumerate(train_ds.class_names):
    print(f"{i:2d} : {name}")
