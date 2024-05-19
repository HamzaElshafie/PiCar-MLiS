import os
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from hyperopt import fmin, tpe, hp, Trials

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
train_targets, test_targets = train_test_split(targets, test_size=0.1, random_state=42)
train_targets, val_targets = train_test_split(train_targets, test_size=0.1, random_state=42)

# Create data generators for training, validation, and test sets
train_dataset = CustomDataGenerator(targets=train_targets, root_dir_original=root_dir_original, root_dir_darkened=root_dir_darkened)
val_dataset = CustomDataGenerator(targets=val_targets, root_dir_original=root_dir_original, root_dir_darkened=root_dir_darkened)
test_dataset = CustomDataGenerator(targets=test_targets, root_dir_original=root_dir_original, root_dir_darkened=root_dir_darkened)

# Define the search space for hyperparameters
space = {
    'learning_rate': hp.loguniform('learning_rate', np.log(1e-5), np.log(1e-2)),
    'num_filters': hp.quniform('num_filters', 32, 512, 1),
    'dropout_rate': hp.uniform('dropout_rate', 0.2, 0.5)
}

def train_model(params):
    # Load your models
    base_model_vgg16 = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=image_shape)
    base_model_vgg19 = tf.keras.applications.VGG19(weights='imagenet', include_top=False, input_shape=image_shape)

    # Freeze the base model layers
    for base_model in [base_model_vgg16, base_model_vgg19]:
        for layer in base_model.layers:
            layer.trainable = False

    # Create branches for each base model
    input_layer = layers.Input(shape=image_shape)
    vgg16_output = base_model_vgg16(input_layer)
    vgg19_output = base_model_vgg19(input_layer)

    # Concatenate the outputs of the base models
    concatenated_output = layers.Concatenate()([vgg16_output, vgg19_output])

    # Apply Conv2D and Dense layers for further processing
    x = layers.Conv2D(int(params['num_filters']), padding="same", kernel_size=(3, 3), activation='relu')(concatenated_output)
    x = layers.Conv2D(int(params['num_filters']), padding="same", kernel_size=(3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(int(params['num_filters']), padding="same", kernel_size=(3, 3), activation='relu')(x)
    x = layers.Conv2D(int(params['num_filters']), padding="same", kernel_size=(3, 3), activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(params['dropout_rate'])(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(params['dropout_rate'])(x)
    outputs = layers.Dense(2, activation='relu')(x)

    # Create and compile the model
    model = tf.keras.Model(inputs=input_layer, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    # Training loop
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=5,
        #callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
        verbose=0
    )

    # Evaluate on the validation set
    val_loss = history.history['val_loss'][-1]
    return val_loss

# Define a function to minimize (validation loss)
def objective(params):
    val_loss = train_model(params)
    return val_loss

# Perform Bayesian optimization
trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=10, trials=trials)

print("Best hyperparameters:", best)

# Train the model with the best hyperparameters
best_params = space_eval(space, best)
best_val_loss = train_model(best_params)

# Load the best model based on validation loss
model = train_model(best_params)

# Evaluate on the test set
test_loss = model.evaluate(test_dataset)
print("Test loss:", test_loss)

print('Finished Training')
