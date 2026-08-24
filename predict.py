import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import img_to_array
import cv2


def auto_crop_scope_region(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    height, width, _ = img.shape

    # light crop
    top = int(height * 0.01)
    bottom = int(height * 0.98)
    left = int(width * 0.01)
    right = int(width * 0.99)

    cropped = img[top:bottom, left:right]

    return cropped


MODEL_PATH = "waveform_classifier.keras"
IMAGE_PATH = "waveform_testingset/triangle_predict_0002.png"

IMG_HEIGHT = 300
IMG_WIDTH = 600

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# load class names
with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]


img = auto_crop_scope_region(IMAGE_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

# show new CV2 image
cv2.imshow("Image Sent to CNN", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()

# conv image
img_array = img_to_array(img)
img_array = tf.expand_dims(img_array, axis=0)

# Predict waveform
predictions = model.predict(img_array)
score = predictions[0]

print("Prediction output:", score)
print("Number of model outputs:", len(score))
print("Number of class names:", len(class_names))
print("Argmax index:", np.argmax(score))
print("Class names:", class_names)

if len(score) != len(class_names):
    print("ERROR: Model output count does not match class_names.")
else:
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    print(f"Predicted waveform: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")