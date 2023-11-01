import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from skimage import io
from abc import ABC, abstractmethod


class ImageProcess:
    def __init__(self, path):
        self.__path = path

    def read_image(self):
        return io.imread(self.__path)

    @staticmethod
    def normalize_band(band):
        return (band - np.min(band)) / (np.max(band) - np.min(band))


def display_band(title: str, band: np.ndarray, cmap: cm):
    plt.figure(figsize=(15, 10))
    plt.imshow(band, cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.axis('off')
    plt.show()


class BandCalculate(ABC):
    def __init__(self, band1: ImageProcess, nir_band: ImageProcess):
        self.__result = None
        self._band1 = ImageProcess.normalize_band(np.array(band1.read_image()).astype(float))
        self._nir_band = ImageProcess.normalize_band(np.array(nir_band.read_image()).astype(float))

    @abstractmethod
    def calculate_band(self) -> np.ndarray:
        pass

    @abstractmethod
    def show(self, title):
        pass


class NDVI(BandCalculate):

    def __init__(self, red_band: ImageProcess, nir_band: ImageProcess):
        super().__init__(red_band, nir_band)
        self.__ndvi = None

    def calculate_band(self) -> np.ndarray:
        self.__ndvi = (self._nir_band - self._band1) / (self._nir_band + self._band1)
        return self.__ndvi

    def show(self, title: str):
        if self.__ndvi is None:
            raise ValueError("NDVI not calculated. Please calculate NDVI using calculate_band() method before "
                             "displaying.")
        display_band(title, self.__ndvi, cmap=cm.Grays)


class NDWI(BandCalculate):
    def __init__(self, green_band: ImageProcess, nir_band: ImageProcess):
        super().__init__(green_band, nir_band)
        self.__ndwi = None

    def calculate_band(self) -> np.ndarray:
        self.__ndwi = (self._band1 - self._nir_band) / (self._band1 + self._nir_band)
        return self.__ndwi

    def show(self, title: str):
        if self.__ndwi is None:
            raise ValueError("NDWI not calculated. Please calculate NDWI using calculate_band() method before "
                             "displaying.")
        display_band(title, self.__ndwi, cmap=cm.Grays)


class SAVI(BandCalculate):
    def __init__(self, red_band: ImageProcess, nir_band: ImageProcess):
        super().__init__(red_band, nir_band)
        self.__savi = None

    def calculate_band(self) -> np.ndarray:
        self.__savi = ((self._nir_band - self._band1) / (self._nir_band + self._band1 + 0.5)) * (1 + 0.5)
        return self.__savi

    def show(self, title):
        if self.__savi is None:
            raise ValueError("SAVI not calculated. Please calculate SAVI using calculate_band() method before "
                             "displaying.")
        display_band(title, self.__savi, cmap=cm.gray)


class AFVI(BandCalculate):
    def __init__(self, swir1_band, nir_band):
        super().__init__(swir1_band, nir_band)
        self.__afvi = None

    def calculate_band(self) -> np.ndarray:
        self.__afvi = ((self._nir_band - 0.66) * (self._band1 / (self._nir_band + (0.66 * self._band1))))
        return self.__afvi

    def show(self, title):
        if self.__afvi is None:
            raise ValueError("AFVI not calculated. Please calculate AFVI using calculate_band() method before "
                             "displaying.")
        display_band(title, self.__afvi, cmap=cm.gray)


class UI(BandCalculate):
    def __init__(self, swir2_band, nir_band):
        super().__init__(swir2_band, nir_band)
        self.__ui = None

    def calculate_band(self) -> np.ndarray:
        self.__ui = (self._band1 - self._nir_band) / (self._band1 + self._nir_band)
        return self.__ui

    def show(self, title):
        if self.__ui is None:
            raise ValueError("UI not calculated. Please calculate UI using calculate_band() method before "
                             "displaying.")
        display_band(title, self.__ui, cmap=cm.gray)


class BI(BandCalculate):
    def __init__(self, green_band, red_band, nir_band):
        super().__init__(green_band, nir_band)
        self._red_band = ImageProcess.normalize_band(np.array(red_band.read_image()).astype(float))
        self.__bi = None

    def calculate_band(self) -> np.ndarray:
        self.__bi = ((self._nir_band - self._band1) - self._red_band) / (
                (self._nir_band + self._band1) + self._red_band)
        return self.__bi

    def show(self, title):
        if self.__bi is None:
            raise ValueError("BI not calculated. Please calculate BI using calculate_band() method before "
                             "displaying.")
        display_band(title, self.__bi, cmap=cm.gray)
