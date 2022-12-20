from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydicom import FileDataset, dcmread


def normalize_dicom(dicom: FileDataset) -> NDArray[np.float64]:
    image = dicom.pixel_array
    image = image - image.min()
    image = image / image.max()

    if dicom.PhotometricInterpretation == "MONOCHROME1":
        image = 1 - image

    return image


def read_dicom_and_normalize(dicom_path: Path) -> NDArray[np.float64]:
    dicom = dcmread(dicom_path)
    image = normalize_dicom(dicom)
    image *= 255
    return image
