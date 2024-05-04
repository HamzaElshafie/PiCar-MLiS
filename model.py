import os
import numpy as np
import tensorflow as tf
import cv2

class Model:
    saved_model = 'MobileNet2/'  # Path to the directory containing your MobileNetV3 model

    def __init__(self):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.saved_model)
        try:
            self.model = tf.keras.models.load_model(model_path)
            if self.model:
                print("Model loaded successfully.")
                print("Model type:", type(self.model))
            else:
                print("Model loaded is None, check the model path and compatibility.")
        except Exception as e:
            print(f"Failed to load model from {model_path}. Error: {e}")
            self.model = None


    def preprocess(self, image):
        # Convert the image to floating point and resize
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = tf.image.resize(image, [108, 108])  # Adjusted to match the model's expected input size
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = tf.keras.applications.mobilenet_v3.preprocess_input(image)
        
        # Add a batch dimension
        image = tf.expand_dims(image, axis=0)  # Add batch dimension
        return image

    def predict(self, image):
        # Predict angle and speed from the processed image if the model is loaded
        if self.model is not None:
            image = self.preprocess(image)
            predictions = self.model.predict(image)[0]  # Assume the model outputs a batch, and we take the first
##            angle = 80 * np.clip(predictions[0], 0, 1) + 50  # Scale the angle output
##            angles = np.arange(17)*80+50
            print(f"Angle before conversion: {predictions[0]}")
            print(f"Speed before conversion: {predictions[1]}")
            angle = predictions[0]*80+50
##            speed = 35 * np.clip(predictions[1], 0, 1)  # Scale the speed output
            speed = speed = np.around(predictions[1]).astype(int)*35
            return angle, speed
        else:
            # Output default values or handle the case when the model is not loaded
            print("Model is not loaded, cannot predict.")
            return None, None

# Usage of this class should be done in a context where the environment is prepared
# and the necessary libraries and models are correctly set up.
