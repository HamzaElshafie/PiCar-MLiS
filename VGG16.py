import os
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, callbacks


# Load your dataset
class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, targets, root_dir_original, root_dir_darkened, batch_size=32, shuffle=True):
        self.targets = targets
        self.root_dir_original = root_dir_original
        self.root_dir_darkened = root_dir_darkened
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return len(self.targets) // self.batch_size

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        batch_targets = self.targets.iloc[indexes]
        X, y = self.__data_generation(batch_targets)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.targets))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_targets):
        X = np.empty((self.batch_size, *image_shape))
        y = np.empty((self.batch_size, 2))

        for i, target in enumerate(batch_targets.itertuples()):
            img_name_original = os.path.join(self.root_dir_original, f"{target.image_id}.png")
            img_name_darkened = os.path.join(self.root_dir_darkened, f"{target.image_id}.png")

            # Choose randomly between original and darkened images
            if random.choice([True, False]):
                img_name = img_name_original
            else:
                img_name = img_name_darkened

            image = tf.keras.preprocessing.image.load_img(img_name, target_size=image_shape)
            X[i,] = tf.keras.preprocessing.image.img_to_array(image)
            y[i,] = [target.angle, target.speed]

        return X / 255.0, y  # normalize image

# Load your data
targets_csv = '/kaggle/input/mlis-kaggle/training_norm.csv'  # Replace with the correct CSV file
root_dir_original = '/kaggle/input/mlis-kaggle/training_data/training_data/'
root_dir_darkened = '/kaggle/input/mlis-kaggle/training_data/darkened_data/'
image_shape = (224, 224, 3) 
# Load targets
targets = pd.read_csv(targets_csv)

# Split the dataset
train_targets, test_targets = train_test_split(targets, test_size=0.2, random_state=42)
train_targets, val_targets = train_test_split(train_targets, test_size=0.2, random_state=42)

# Create data generators for training, validation, and test sets
train_dataset = CustomDataGenerator(targets=train_targets, root_dir_original=root_dir_original, root_dir_darkened=root_dir_darkened)
val_dataset = CustomDataGenerator(targets=val_targets, root_dir_original=root_dir_original, root_dir_darkened=root_dir_darkened)
test_dataset = CustomDataGenerator(targets=test_targets, root_dir_original=root_dir_original, root_dir_darkened=root_dir_darkened)

# Load pre-trained model (VGG16) and replace the final layer
base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=image_shape)
for layer in base_model.layers[:-4]:
    layer.trainable = False
    
# Create your custom model head with self-attention
inputs = layers.Input(shape=image_shape)
x = base_model(inputs)
x = layers.Conv2D(filters=187, kernel_size=(3, 3), padding="same", activation='relu')(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Conv2D(filters=432, kernel_size=(3, 3), padding="same", activation='relu')(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Conv2D(filters=435, kernel_size=(3, 3), padding="same", activation='relu')(x)
x = layers.MultiHeadAttention(num_heads=8, key_dim=2)(x, x)  # Self-attention
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(217, activation='relu')(x)
x = layers.Dropout(0.2484014519192801)(x)
x = layers.Dense(357, activation='relu')(x)
x = layers.Dropout(0.3562685197944236)(x)
x = layers.Dense(138)(x)
outputs = layers.Dense(2)(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Define loss function and optimizer
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

# Define a ModelCheckpoint callback
checkpoint_filepath = '/kaggle/working/weight.h5'
model_checkpoint_callback = callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose=1
)

# Training loop with ModelCheckpoint
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=100,
    callbacks=[model_checkpoint_callback]
)

# Load the best weights
model.load_weights(checkpoint_filepath)

# Evaluate on the test set
model.evaluate(test_dataset)

print('Finished Training')
