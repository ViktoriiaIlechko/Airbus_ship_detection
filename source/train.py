import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.metrics import MeanIoU
from source.data import load_and_preprocess_data
from tensorflow.keras.models import load_model


# Define U-Net model
def build_unet(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    # Encoder
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    # Middle
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    # Decoder
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    up1 = layers.UpSampling2D(size=(2, 2))(conv3)

    # Concatenate encoder and decoder paths
    concat1 = layers.Concatenate(axis=-1)([conv2, up1])
    conv4 = layers.Conv2D(128, 3, activation='relu', padding='same')(concat1)
    conv4 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv4)
    up2 = layers.UpSampling2D(size=(2, 2))(conv4)

    concat2 = layers.Concatenate(axis=-1)([conv1, up2])
    conv5 = layers.Conv2D(64, 3, activation='relu', padding='same')(concat2)
    conv5 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv5)

    # Output layer
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv5)

    mod = models.Model(inputs=inputs, outputs=outputs)
    return mod


def dice_coefficient(y_true, y_pred, smooth=1):
    y_true = tf.cast(y_true, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice


def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)


if __name__ == "__main__":
    # Load and preprocess data
    data_path = 'C:\Viktoria\wthRmn\pythonProject\Airbus_ship\data_airbus'
    labels_file = 'C:\\Viktoria\\wthRmn\\pythonProject\\Airbus_ship\\data_airbus\\train_ship_segmentations_v2.csv'
    x_train, y_train, x_val, y_val = load_and_preprocess_data(data_dir=data_path, labels_file=labels_file,
                                                              test_size=0.2,
                                                              random_state=42)

    # Build the model
    input_shape = x_train[0].shape
    model = build_unet(input_shape)

    # Compile the model
    # model.compile(optimizer='adam', loss=dice_loss, metrics=[MeanIoU(num_classes=2)])
    model.compile(optimizer='adam', loss=dice_loss, metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

    # Save the model
    model.save("C:\Viktoria\wthRmn\pythonProject\Airbus_ship\source\s_s_model.h5")
