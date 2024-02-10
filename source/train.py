import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.losses import Dice
from data import load_and_preprocess_data

# Define U-Net model
def build_unet(input_shape):


# Load and preprocess data
x_train, y_train = load_and_preprocess_data()

# Build the model
input_shape = x_train[0].shape
model = build_unet(input_shape)

# Compile the model
model.compile(optimizer='adam', loss=Dice(), metrics=[MeanIoU(num_classes=2)])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Save the model
model.save("semantic_segmentation_model.h5")
