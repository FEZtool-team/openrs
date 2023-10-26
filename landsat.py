import numpy as np
import matplotlib.pyplot as plt


class Landsat:

    def __init__(self, *bands):
        for band in bands:
            if not isinstance(band, np.ndarray):
                raise ValueError("Invalid band data. Must be a NumPy array.")
        self.bands = bands
        self.stack = None

    def get_multi_spectral_img(self):
        band_list = []
        for band in self.bands:
            normalize_band = (band - np.min(band)) / (np.max(band) - np.min(band))
            band_list.append(normalize_band)
        stack = np.stack(band_list, axis=2)
        self.stack = stack
        return stack

    def show_img(self, title):
        if self.stack is None:
            self.get_multi_spectral_img()
        plt.imshow(self.stack)
        plt.title(title)
        plt.axis("off")
        plt.show()
