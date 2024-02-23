from source.data import load_and_preprocess_data
from source.train import dice_loss
import tensorflow as tf
import pathlib
import os

if __name__ == "__main__":
    # Load the trained model
    current_d = pathlib.Path(__file__).parent
    model_path = os.path.abspath(os.path.join(current_d, 'models\s_s_model.h5'))
    model = tf.keras.models.load_model(model_path, custom_objects={'dice_loss': dice_loss})
    data_dir = os.path.abspath(os.path.join(current_d, 'data_airbus'))
    labels_file = os.path.abspath(os.path.join(current_d, 'data_airbus\\train_ship_segmentations_v2.csv'))

    # Evaluate the model
    x_train, y_train, x_val, y_val = load_and_preprocess_data(data_dir=data_dir, labels_file=labels_file, test_size=0.2,
                                                              random_state=42)
    loss, accuracy = model.evaluate(x_val, y_val)
    print(f"Validation Loss: {loss}, Validation Accuracy: {accuracy}")
