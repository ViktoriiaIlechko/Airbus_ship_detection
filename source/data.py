import numpy as np
from skimage import io
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(data_dir, mask_dir, test_size=0.2, random_state=42):
    """
    Load and preprocess data for semantic segmentation.

    Parameters:
        - data_dir (str): Directory containing the input images.
        - mask_dir (str): Directory containing the corresponding mask images.
        - test_size (float): Proportion of the dataset to include in the test split.
        - random_state (int): Seed for random number generation.

    Returns:
        - x_train (numpy array): Input images for training.
        - y_train (numpy array): Corresponding mask images for training.
        - x_test (numpy array): Input images for testing.
        - y_test (numpy array): Corresponding mask images for testing.
    """
    # Load file paths for images and masks
    image_files = [f for f in os.listdir(data_dir) if f.endswith('.png')]
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]

    # Ensure images and masks have corresponding files
    image_files.sort()
    mask_files.sort()
    assert len(image_files) == len(mask_files), "Number of images and masks must be the same."

    # Load images and masks
    images = [io.imread(os.path.join(data_dir, f)) for f in image_files]
    masks = [io.imread(os.path.join(mask_dir, f), as_gray=True) for f in mask_files]

    # Preprocess and normalize images and masks as needed
    images = np.array(images) / 255.0  # Normalize pixel values to the range [0, 1]
    masks = np.array(masks)  # Masks may need additional preprocessing based on your data

    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(images, masks, test_size=test_size, random_state=random_state)

    return x_train, y_train, x_test, y_test
