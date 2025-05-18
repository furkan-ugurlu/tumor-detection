"""
Brain Tumor Detection using Convolutional Neural Networks (CNN)

This script loads, preprocesses, and trains a CNN model to classify brain tumor images
into multiple categories. It includes data loading, resizing, normalization, model building,
training, evaluation, and visualization of training history.

The dataset should be organized in subfolders per class for both training and testing.
"""

# Read the data
# Preprocess the data
#    - Data cleaning
#    - Resize to the same dimensions
#    - Apply padding/train
# Build the Neural Network Model
#    - Layers
#    - Activation functions

# Library imports

import os, warnings
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import shutil

# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='magma')
warnings.filterwarnings("ignore") # Suppress warnings to clean up output cells


# The image dimensions are highly inconsistent, so resizing is necessary 
# to ensure uniformity. Additionally, an interpolation strategy is applied 
# to prevent image quality loss during resizing.


filtered_train_dir = "Brain Tumor data/Brain Tumor data/Training"
filtered_test_dir = "Brain Tumor data/Brain Tumor data/Testing"


# Only supported files will be used when calling image_dataset_from_directory
ds_train_ = tf.keras.utils.image_dataset_from_directory(
    filtered_train_dir, # Directory containing the training images
    labels='inferred', # Automatically infer labels from subdirectory names
    label_mode='categorical', # Use categorical labels (one-hot encoding) we have 4 type cancer data
    image_size=[128, 128], # Resize images to 128x128 pixels
    interpolation='nearest', # Use nearest neighbor interpolation for resizing
    batch_size=64, # Set the batch size to 64
    shuffle=True  # Shuffle the training dataset
)

ds_valid_ = tf.keras.utils.image_dataset_from_directory(
    filtered_test_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=[128, 128],
    interpolation='nearest',
    batch_size=64,
    shuffle=False  # Do not shuffle the validation dataset
)

print("Train Set Size", ds_train_.cardinality())
print("Test Set Size", ds_valid_.cardinality())

# Data Pipeline
def convert_to_float(image, label): # Converts the data to the appropriate format
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image, label

AUTOTUNE = tf.data.AUTOTUNE # Enables automatic tuning for performance optimization

ds_train = (
    ds_train_
    .map(convert_to_float) # Apply the conversion function
    .cache()   # Cache the data in memory
    .prefetch(buffer_size=AUTOTUNE)  # Load data in parallel to prevent bottlenecks
)

ds_valid = (
    ds_valid_
    .map(convert_to_float)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)

# Determine the number of classes from the dataset
num_classes = len(ds_train_.class_names)
print(f"Number of classes: {num_classes}")

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (5, 5), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (5, 5), activation='relu'),
    tf.keras.layers.Conv2D(128, (5, 5), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),  # Add a dropout layer to prevent overfitting
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')  # Match the number of classes
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy', # Used for multi-class classification
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    ds_train,
    validation_data=ds_valid,
    epochs=10, # Number of epochs
    verbose=1, # Display training progress
)

# Evaluate the model
loss, accuracy = model.evaluate(ds_valid)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Visualize the model's performance
def plot_training_history(history):
    fig = plt.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    
    # Loss
    ax0.plot(history.history['loss'], label='Train Loss')
    ax0.plot(history.history['val_loss'], label='Validation Loss')
    ax0.set_title('Loss')
    ax0.set_xlabel('Epochs')
    ax0.set_ylabel('Loss')
    ax0.legend()
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    plt.show()

plot_training_history(history)

