from pathlib import Path

import cv2
import pandas as pd
import torch
from pydicom import dcmread
from torch.utils.data import Dataset

from src.lib.cv import normalize_dicom


class RSNABreastCancerDataset(Dataset[tuple[torch.Tensor, int]]):
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

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        impath: Path = next(
            self.images_folder.rglob(f"{self.metadata.image_id[index]}.*")
        )
        if impath.suffix == ".dcm":
            numpy_image = 255 * normalize_dicom(dcmread(impath))
        else:
            numpy_image = cv2.imread(str(impath), cv2.IMREAD_GRAYSCALE)
        image = torch.tensor(numpy_image, dtype=torch.float)
        if "cancer" not in self.metadata.columns:
            # for test dataset, the "cancer" column is not present
            return image, 0
        return image, int(self.metadata.cancer[index])
