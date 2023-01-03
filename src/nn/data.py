from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from pydicom import dcmread
from torch.utils.data import Dataset

from src.lib.cv import normalize_dicom


def _get_tensor_image(image_dir: Path, patient_id: int, image_id: int) -> torch.Tensor:
    patient_dir = image_dir / str(patient_id)
    image_path = next(patient_dir.glob(f"{image_id}.*"))
    numpy_image: NDArray[np.float64]

    if image_path.suffix == ".dcm":
        numpy_image = 255 * normalize_dicom(dcmread(image_path))
    else:
        numpy_image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    return torch.tensor(numpy_image, dtype=torch.float)


class RSNABreastCancerTrainDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, labels_csv_path: Path, images_folder: Path) -> None:
        """Dataset of breast cancer samples from RSNA Kaggle challenge.
        challenge link: https://www.kaggle.com/competitions/rsna-breast-cancer-detection
        This dataset operates on either dicom (.dcm) or cv2.imread compatible extensions
        (like png, jpeg, ...).
        For train dataset, a tuple of image, target is returned. For test dataset,
        a dummy label `0` is returned next to the test image. The returned image
        is a single channel float tensor, i.e. expect the image.shape to be a tuple like
        (320, 180), not (320, 180 , 3). The pixel values are rescaled to 0-255 range.
        The dicom images provided by contest organizers have more than 256 possible
        pixel values, thus if images_folder contains dicom (.dcm) files, expect
        fractional pixel values.

        Parameters
        ----------
        labels_csv_path : Path
            path to csv file with samples metadata
        images_folder : Path
            folder where the images are stored (dcm/png/jpeg)
        """
        super().__init__()
        self.metadata = pd.read_csv(labels_csv_path)
        self.images_folder = Path(images_folder)
        assert "cancer" in self.metadata.columns

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return _get_tensor_image(
            self.images_folder,
            self.metadata.patient_id[index],
            self.metadata.image_id[index],
        ), int(self.metadata.cancer[index])


class RSNABreastCancerTestDataset(Dataset[torch.Tensor]):
    def __init__(self, labels_csv_path: Path, images_folder: Path) -> None:
        """Dataset of breast cancer samples from RSNA Kaggle challenge.
        challenge link: https://www.kaggle.com/competitions/rsna-breast-cancer-detection
        This dataset operates on either dicom (.dcm) or cv2.imread compatible extensions
        (like png, jpeg, ...).
        For train dataset, a tuple of image, target is returned. For test dataset,
        a dummy label `0` is returned next to the test image. The returned image
        is a single channel float tensor, i.e. expect the image.shape to be a tuple like
        (320, 180), not (320, 180 , 3). The pixel values are rescaled to 0-255 range.
        The dicom images provided by contest organizers have more than 256 possible
        pixel values, thus if images_folder contains dicom (.dcm) files, expect
        fractional pixel values.

        Parameters
        ----------
        labels_csv_path : Path
            path to csv file with samples metadata
        images_folder : Path
            folder where the images are stored (dcm/png/jpeg)
        """
        super().__init__()
        self.metadata = pd.read_csv(labels_csv_path)
        self.images_folder = Path(images_folder)
        assert "cancer" not in self.metadata.columns

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> torch.Tensor:
        return _get_tensor_image(
            self.images_folder,
            self.metadata.patient_id[index],
            self.metadata.image_id[index],
        )
