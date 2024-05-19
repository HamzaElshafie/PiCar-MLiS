import os
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
import optuna

# Define the search space for hyperparameters
def objective(trial):
    # Define the search space for hyperparameters
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    dense_units = trial.suggest_int('dense_units', 32, 1020)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.2, 0.5)
    momentum = trial.suggest_uniform('momentum', 0.0, 0.9)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)

    # Load your model
    base_model_mbnet = tf.keras.applications.MobileNetV3Large(weights='imagenet', include_top=False, input_shape=image_shape)

    # Freeze the base model layers
    for layer in base_model_mbnet.layers:
        layer.trainable = True

    # Create branches for each base model
    input_layer = layers.Input(shape=image_shape)
    mbnetv3_output = base_model_mbnet(input_layer, training=True)

    # Apply GlobalAveragePooling2D
    x = layers.GlobalAveragePooling2D()(mbnetv3_output)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(dense_units, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(2, activation='linear')(x)

    # Create and compile the model
    model = tf.keras.Model(inputs=input_layer, outputs=outputs)
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate,
        momentum=momentum,
        decay=weight_decay
    )
    model.compile(optimizer=optimizer, loss='mse', weighted_metrics=['mae'])

    # Training loop
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=20,
        verbose=0,
        callbacks=[ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', mode='min')]
    )

    # Return the validation loss
    val_loss = history.history['val_loss'][-1]
    return val_loss

# Run the hyperparameter optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

# Get the best hyperparameters
best_params = study.best_params
print("Best hyperparameters:", best_params)
