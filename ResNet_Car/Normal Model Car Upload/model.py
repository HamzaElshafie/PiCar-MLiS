import numpy as np
import tensorflow as tf
import os

class Model:
    saved_model = 'MobileNet/'
    
    def __init__(self):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.saved_model)
        self.model = tf.keras.models.load_model(model_path)

    def preprocess(self, image):
        # Convert image to the appropriate data type, resize it, and scale according to ResNet50's requirements
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [224, 224])
        image = tf.keras.applications.resnet50.preprocess_input(image)
        image = tf.expand_dims(image, axis=0)  # Ensure image has shape (1, 224, 224, 3)
        return image

    def predict(self, image):
        image = self.preprocess(image)
        angle, speed = self.model.predict(image)[0]  # Directly use 'image' without wrapping it in another list
        angle = 80 * np.clip(angle, 0, 1) + 50
        speed = 35 * np.clip(speed, 0, 1)
        return angle, speed
