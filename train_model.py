# documentation: https://www.tensorflow.org/tutorials/images/classification
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import image_dataset_from_directory

# dataset settings
DATASET_PATH = "waveform_dataset"
IMG_HEIGHT = 300
IMG_WIDTH = 600
BATCH_SIZE = 32

# load dataset
train_ds = image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

validation_ds = image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

# class names
class_names = train_ds.class_names
print("Class names:", class_names)

with open("class_names.txt", "w") as f:
    for name in class_names:
        f.write(name + "\n")

# speed up dataset loading
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size = AUTOTUNE)
validation_ds = validation_ds.cache().prefetch(buffer_size = AUTOTUNE)

# light data augmentation
data_augmentation = tf.keras.Sequential([layers.RandomContrast(0.05)])

# build CNN model
model = models.Sequential([
    layers.Rescaling(1./255),

    data_augmentation,

    layers.Conv2D(16, (3, 3), activation = 'relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(32, (3, 3), activation = 'relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), activation = 'relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128, activation = 'relu'),
    layers.Dropout(0.2),

    layers.Dense(len(class_names), activation = 'softmax')
])

# compile model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# train model
history = model.fit(train_ds, validation_data = validation_ds, epochs = 20)

model.save("waveform_classifier.keras")
print("Model saved as waveform_classifier.keras")