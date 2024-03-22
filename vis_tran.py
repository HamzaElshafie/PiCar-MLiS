import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from vit_keras import vit

# Define the image shape
image_shape = (224, 224, 3)  # Example shape, change as per your requirements

class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, targets, root_dir_original, batch_size=32, shuffle=True, n_workers=4):
        self.targets = targets
        self.root_dir_original = root_dir_original
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_workers = n_workers
        self.on_epoch_end()

    def __len__(self):
        return len(self.targets) // self.batch_size

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        X, y = self.__data_generation(indexes)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.targets))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        X = np.empty((len(indexes), *image_shape))
        y = np.empty((len(indexes), 2))

        for i, idx in enumerate(indexes):
            target = self.targets.iloc[idx]
            img_id = str(int(target.image_id))  # Convert to integer and then to string to remove ".0" suffix
            img_name_original = os.path.join(self.root_dir_original, f"{img_id}.png")

            # Load and preprocess the image
            image = tf.keras.preprocessing.image.load_img(img_name_original, target_size=image_shape[:2])
            image = tf.keras.preprocessing.image.img_to_array(image)
            image /= 255.0

            # Store the image and target
            X[i,] = image
            y[i,] = [target.angle, target.speed]

        return X, y

# Load your data
targets_csv = '/content/mlis_kagg/training_norm.csv'  # Replace with the correct CSV file
root_dir_original = '/content/mlis_kagg/training_data/training_data'

# Load targets
targets = pd.read_csv(targets_csv)

# Split the dataset
train_targets, test_val_targets = train_test_split(targets, test_size=0.1, random_state=42)
val_targets, test_targets = train_test_split(test_val_targets, test_size=0.1, random_state=42)

# Create data generators for training, validation, and test sets
train_dataset = CustomDataGenerator(targets=train_targets, root_dir_original=root_dir_original)
val_dataset = CustomDataGenerator(targets=val_targets, root_dir_original=root_dir_original)
test_dataset = CustomDataGenerator(targets=test_targets, root_dir_original=root_dir_original)

# Create a vision transformer model
model = vit.vit_b32(
    image_size=image_shape[0],
    activation='relu',
    pretrained=True,
    include_top=True,  # Set to True for regression
    pretrained_top=False,
    classes=2,  # Set the number of output neurons to 2 for angle and speed
    regression=True  # Specify that it's a regression task
)

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Define a ModelCheckpoint callback to save the model with the best validation loss
checkpoint_filepath = 'ViT_regression_model_checkpoint.keras'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_best_only=True,
    monitor='val_loss',
    mode='min',
    verbose=1
)

# Training loop
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=1,
    callbacks=[model_checkpoint_callback]  # Include the ModelCheckpoint callback
)

# Load the best model based on validation loss
best_model = tf.keras.models.load_model(checkpoint_filepath)

# Evaluate on the test set
best_model.evaluate(test_dataset)

print('Finished Training')
