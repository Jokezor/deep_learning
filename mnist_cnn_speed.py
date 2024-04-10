import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
import os
import datetime

# Enable mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Load and prepare the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Reshape images to fit the CNN input shape and convert labels to one-hot encoding
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Define the model architecture
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax', dtype='float32')  # Ensure the output layer dtype is float32 for mixed precision
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Prepare the data using tf.data for optimal loading
AUTOTUNE = tf.data.experimental.AUTOTUNE
batch_size = 128  # Adjust based on your GPU memory

def prepare_dataset(images, labels, training=True):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if training:
        dataset = dataset.shuffle(10000)
    dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)
    return dataset

train_dataset = prepare_dataset(train_images, train_labels)
test_dataset = prepare_dataset(test_images, test_labels, training=False)

# Set up TensorBoard logging
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train the model
model.fit(train_dataset, epochs=5, validation_data=test_dataset, callbacks=[tensorboard_callback])

# Evaluate the model
test_loss, test_acc = model.evaluate(test_dataset, verbose=2)
print('\nTest accuracy:', test_acc)
