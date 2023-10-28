import os

from skimage import io
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np


class SimplePCA:

    """
    A class for performing Principal Component Analysis (PCA) on a collection of images.

    Parameters:
    image_collection_path (str): The path to the collection of images.

    Attributes:
    __image_collection_path (str): The path to the image collection.
    __images (list): List to store flattened images.
    __pcs (list): List to store PCA components.
    __default_shape (tuple): Default shape of the images in the collection.

    Methods:
    apply_pca(): Applies PCA on the loaded images.
    get_pca_list() -> list: Returns the list of PCA components.
    show_pca_images(): Displays the PCA components and their histograms.
    save_pca_images(path): Saves the PCA components as TIFF images in the specified path.
    """

    def __init__(self, image_collection_path):
        """
        Initialize the SimplePCA object with the provided image collection path.

        Args:
        image_collection_path (str): The path to the collection of images.
        """
        self.__image_collection_path = image_collection_path
        self.__images = []
        self.__pcs = []
        self.__default_shape = None

    def __load_images(self):
        ic = io.imread_collection(self.__image_collection_path)
        self.__default_shape = ic[0].shape
        for img in ic:
            self.__images.append(img.flatten())

    def apply_pca(self):
        """
         Apply Principal Component Analysis (PCA) on the loaded images.
         """
        self.__load_images()
        pca = PCA()
        pca.fit(self.__images)
        self.__pcs = [component.reshape(self.__default_shape) for component in pca.components_]

    def get_pca_list(self) -> list:
        """
        Get the list of PCA components.

        Returns:
        list: List of PCA components.
        """
        return self.__pcs

    def show_pca_images(self):
        """
        Display the PCA components and their histograms using Matplotlib.
        """
        if not self.__pcs:
            raise ValueError("PCA components not calculated. Call apply_pca method first.")

        fig, ax = plt.subplots(nrows=len(self.__pcs), ncols=2, figsize=(20, 30))
        for i in range(len(self.__images)):
            # Display PCA component as image
            ax[i, 0].imshow(self.__pcs[i], cmap='gray')
            ax[i, 0].set_title(f'PCA Band {i + 1}')
            ax[i, 0].axis('off')
            ax[i, 0].grid(False)
            cbar = ax[i, 0].imshow(self.__pcs[i], cmap='gray')
            plt.colorbar(cbar, ax=ax[i, 0], orientation='vertical')

            # Display histogram of PCA component
            ax[i, 1].hist(self.__pcs[i].ravel(), bins=256, density=True, histtype='bar', color='black')
            ax[i, 1].set_title(f'Histogram of PCA Band {i + 1}')
            ax[i, 1].set_xlabel('Pixel Intensity')
            ax[i, 1].set_ylabel('Frequency')
        plt.show()

    def save_pca_images(self, path):
        """
        Save the PCA components as TIFF images in the specified directory.

        Args:
        path (str): The directory where the PCA components will be saved.

        Raises:
        FileNotFoundError: If the specified directory does not exist.
        PermissionError: If the program does not have permission to write in the specified directory.
        """
        # Check if the specified directory exists
        if not os.path.exists(path):
            raise FileNotFoundError(f"Directory not found: {path}")

        # Check if the directory is writable
        if not os.access(path, os.W_OK):
            raise PermissionError(f"No write permissions in the directory: {path}")

        # Save PCA components as TIFF images
        for i, component in enumerate(self.__pcs, start=1):
            file_path = os.path.join(path, f'PCA{i}.TIFF')
            plt.imsave(file_path, component)
            print(f"PCA component {i} saved at: {file_path}")
