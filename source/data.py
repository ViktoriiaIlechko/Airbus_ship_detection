import os
import pandas as pd
import numpy as np
from skimage import io
from sklearn.model_selection import train_test_split


def rle_decode(rle_string, shape=(768, 768)):
    """
    Decode a generic RLE-encoded mask into a binary mask.

    Parameters:
        - rle_string (str): RLE-encoded mask string.
        - shape (tuple): Shape of the binary mask (height, width).

    Returns:
        - binary_mask (numpy array): Decoded binary mask.
    """
    # Check if the input is already a binary mask (numpy array)
    if isinstance(rle_string, np.ndarray):
        return rle_string

    # Check if the input is not a string (e.g., float)
    if not isinstance(rle_string, str) or rle_string.lower() == 'nan':
        return np.zeros(shape, dtype=np.uint8)

    # Initialize an array of zeros with the specified shape
    binary_mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    # Split the RLE-encoded string into pairs (start, length)
    pairs = np.array(rle_string.split(), dtype=np.uint)
    # Check if pairs is not empty
    if pairs.size > 0:
        starts, lengths = pairs[::2] - 1, pairs[1::2]
        # Iterate over pairs and set corresponding pixels to 1 in the binary mask
        for start, length in zip(starts, lengths):
            binary_mask[start:start + length] = 1

    # Reshape the 1D array to the specified shape
    binary_mask = binary_mask.reshape(shape).T  # Transpose to match expected shape

    return binary_mask


def load_and_preprocess_data(data_dir, labels_file, test_size=0.2, random_state=42):
    """
    Load and preprocess data for semantic segmentation.

    Parameters:
        - data_dir (str): Root directory containing the input images.
        - labels_file (str): Path to the labeled dataset CSV file.
        - val_size (float): Proportion of the dataset to include in the validation split.
        - random_state (int): Seed for random number generation.

    Returns:
        - x_train (numpy array): Input images for training.
        - y_train (numpy array): Corresponding mask images for training.
        - x_val (numpy array): Input images for validation.
        - y_val (numpy array): Corresponding mask images for validation.
    """
    # Load file paths for images
    image_files = [os.path.join(data_dir, 'train_v2', f) for f in os.listdir(os.path.join(data_dir, 'train_v2')) if
                   f.endswith('.jpg')]

    # Load labeled dataset
    labels_df = pd.read_csv(labels_file)
    # Merge image files with corresponding labels
    labeled_images = pd.merge(pd.DataFrame({'ImageId': image_files}), labels_df, how='left', left_on='ImageId',
                              right_on='ImageId')

    # Assuming labeled_images is your DataFrame
    # Convert the 'EncodedPixels' column to strings
    labeled_images['EncodedPixels'] = labeled_images['EncodedPixels'].astype(str)
    # Handle NaN values
    # For example, replace NaN with an empty string
    labeled_images['EncodedPixels'].fillna('', inplace=True)
    # Extract image paths and labels
    image_paths = labeled_images['ImageId'].values
    labels = labeled_images['EncodedPixels'].values
    # Load images and masks
    images = [io.imread(f) for f in image_paths]
    masks = [rle_decode(label) for label in labels]
    # Preprocess and normalize images and masks as needed
    images = np.array(images) / 255.0  # Normalize pixel values to the range [0, 1]
    masks = np.array(masks)
    # Split the dataset into training and  validation sets
    # x_train, x_val = train_test_split(images, test_size=test_size, random_state=random_state)
    x_train, x_val, y_train, y_val = train_test_split(images, masks, test_size=test_size, random_state=random_state)

    return x_train, y_train, x_val, y_val


def load_new_data(new_data_dir):
    """
    Load and preprocess new data for inference.

    Parameters:
        - new_data_dir (str): Directory containing the new input images for inference.

    Returns:
        - x_new (numpy array): Input images for inference.
    """
    # Load file paths for new images
    if os.path.isdir(new_data_dir):
        new_image_files = [os.path.join(new_data_dir, f) for f in os.listdir(new_data_dir) if f.endswith('.jpg')]

        # Load new images
        x_new = [io.imread(f) for f in new_image_files]
    else:
        # Load a single image
        x_new = [io.imread(new_data_dir)]

    # Preprocess and normalize images as needed
    x_new = np.array(x_new) / 255.0  # Normalize pixel values to the range [0, 1]

    return x_new
