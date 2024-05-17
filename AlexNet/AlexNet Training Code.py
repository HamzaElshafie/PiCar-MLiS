import pandas as pd
import numpy as np
import glob
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, Input, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

# Load features
csv_file_path = '/kaggle/input/machine-learning-in-science-ii-2024/training_norm.csv'
features_df = pd.read_csv(csv_file_path)

# Get image file paths
image_folder_path = '/kaggle/input/machine-learning-in-science-ii-2024/training_data/training_data'
image_file_paths = glob.glob(f'{image_folder_path}/*.png')

# Define target image size
target_size = (227, 227)

# Manually load and preprocess images
def load_and_preprocess_image(image_path, target_size):
    try:
        # Read and preprocess the image
        image = cv2.imread(image_path)
        image = cv2.resize(image, target_size)  # Resize images to a common size
        image = image / 255.0  # Normalize pixel values to the range [0, 1]
        return image
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}. Skipping this example.")
        return None

# Create a dataset generator
def image_data_generator(image_ids, features_df, image_folder_path, target_size):
    for image_id in image_ids:
        image_path = f'{image_folder_path}/{image_id}.png'
        image = load_and_preprocess_image(image_path, target_size)
        if image is not None:
            features = features_df[features_df['image_id'] == image_id]
            if not features.empty:
                angle = features['angle'].values[0]
                speed = features['speed'].values[0]
                yield image, [angle, speed]

# Split data into training (70%), validation (20%), and test (10%) sets
train_ids, test_ids = train_test_split(features_df['image_id'], test_size=0.1, random_state=42)
train_ids, val_ids = train_test_split(train_ids, test_size=0.2222, random_state=42)  # 0.2222 * 90% = 20%

# Create tf.data.Datasets
batch_size = 64

train_dataset = tf.data.Dataset.from_generator(
    lambda: image_data_generator(train_ids, features_df, image_folder_path, target_size),
    output_types=(tf.float32, tf.float32),
    output_shapes=((227, 227, 3), (2,))
).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

val_dataset = tf.data.Dataset.from_generator(
    lambda: image_data_generator(val_ids, features_df, image_folder_path, target_size),
    output_types=(tf.float32, tf.float32),
    output_shapes=((227, 227, 3), (2,))
).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

test_dataset = tf.data.Dataset.from_generator(
    lambda: image_data_generator(test_ids, features_df, image_folder_path, target_size),
    output_types=(tf.float32, tf.float32),
    output_shapes=((227, 227, 3), (2,))
).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

# Manually define AlexNet architecture with a combined output layer
def create_alexnet(input_shape):
    # Input layer
    input_layer = Input(shape=input_shape, name='input_image')

    # Convolutional layers
    x = layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu')(input_layer)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.Conv2D(256, (5, 5), activation='relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.Conv2D(384, (3, 3), activation='relu')(x)
    x = layers.Conv2D(384, (3, 3), activation='relu')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Flatten layer
    x = layers.Flatten()(x)

    # Fully connected layers
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    # Combined output layer for steering angle and speed
    output = layers.Dense(2, activation='linear', name='output')(x)

    # Create the model
    model = Model(inputs=input_layer, outputs=output)

    return model

# Create the AlexNet model
alexnet_model = create_alexnet((target_size[0], target_size[1], 3))

# Compile the model with appropriate loss functions for regression tasks
alexnet_model.compile(optimizer=Adam(), loss='mse')

# Define the model checkpoint callback to save the best model based on validation loss
checkpoint_callback = ModelCheckpoint(
    filepath='best_model.h5', 
    monitor='val_loss', 
    save_best_only=True, 
    save_weights_only=False, 
    mode='min', 
    verbose=1
)

# Training loop
history = alexnet_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=60,
    verbose=2,
    callbacks=[checkpoint_callback]
)

# Evaluate the model on the test set
best_model = tf.keras.models.load_model('best_model.h5')

test_loss = best_model.evaluate(test_dataset)
print(f"Test Loss: {test_loss}")
