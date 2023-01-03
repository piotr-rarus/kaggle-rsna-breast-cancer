import numpy as np
from numpy.typing import NDArray
from pydicom import FileDataset


def normalize_dicom(dicom: FileDataset) -> NDArray[np.float64]:
    image = dicom.pixel_array
    image = image - image.min()
    image = image / image.max()

    if dicom.PhotometricInterpretation == "MONOCHROME1":
        image = 1 - image

    return image
