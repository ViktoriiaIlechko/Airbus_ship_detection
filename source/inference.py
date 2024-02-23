import os
import tensorflow as tf
from source.data import load_and_preprocess_data
from source.data import load_new_data
from source.train import dice_loss
from PIL import Image
import matplotlib.pyplot as plt
import pathlib

if __name__ == "__main__":
    # Load the trained model
    current_dir = pathlib.Path(__file__).parent
    model_dir = os.path.abspath(os.path.join(current_dir, 'models\s_s_model.h5'))
    model = tf.keras.models.load_model(model_dir, custom_objects={'dice_loss': dice_loss})

    # Load new data for inference
    # new_data_dir = os.path.join(data_dir, 'test_v2')
    # x_new = load_new_data(new_data_dir=new_data_dir)
    new_data_dir = os.path.abspath(os.path.join(current_dir, 'data_airbus\\train_test_img.jpg '))
    print('new_data_dir', new_data_dir)
    x_new = load_new_data(new_data_dir=new_data_dir)
    # Make predictions
    predictions = model.predict(x_new)

    # Visualize predictions and save images
    output_dir = os.path.join(current_dir, '..', 'visualisations')

    os.makedirs(output_dir, exist_ok=True)

    for i in range(len(x_new)):
        # Example: Assuming predictions are binary masks
        predicted_mask = (predictions[i] > 0.5).astype('uint8')

        # Visualize and save the original image
        plt.subplot(1, 2, 1)
        plt.imshow(x_new[i][0])  # Assuming x_new[i] is of shape (1, height, width, channels)
        plt.title('Input Image')

        # Visualize and save the predicted mask
        plt.subplot(1, 2, 2)
        plt.imshow(predicted_mask[:, :, 0], cmap='gray')  # Assuming channels=1
        plt.title('Predicted Mask')

        # Save the visualization to disk
        output_path = os.path.join(output_dir, f'prediction_{i}.png')
        plt.savefig(output_path)

        # Clear the current figure for the next iteration
        plt.clf()

    # Close any remaining plots
    plt.close()
