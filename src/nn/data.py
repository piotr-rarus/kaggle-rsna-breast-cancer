from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from imblearn.over_sampling import RandomOverSampler
from numpy.typing import NDArray
from pydicom import dcmread
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

from src.lib.cv import normalize_dicom


def _get_tensor_image(image_dir: Path, patient_id: int, image_id: int) -> torch.Tensor:
    patient_dir = image_dir / str(patient_id)
    image_path = next(patient_dir.glob(f"{image_id}.*"))
    numpy_image: NDArray[np.float64]

    if image_path.suffix == ".dcm":
        numpy_image = 255 * normalize_dicom(dcmread(image_path))
    else:
        numpy_image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    return torch.tensor(numpy_image, dtype=torch.float).unsqueeze(0)


class RSNABreastCancerTrainDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, metadata_csv_path: Path, images_dir: Path) -> None:
        """Dataset of breast cancer samples from RSNA Kaggle challenge.
        challenge link: https://www.kaggle.com/competitions/rsna-breast-cancer-detection
        This dataset operates on either dicom (.dcm) or cv2.imread compatible extensions
        (like png, jpeg, ...).
        For train dataset, a tuple of image, target is returned. For test dataset,
        a dummy label `0` is returned next to the test image. The returned image
        is a single channel float tensor, i.e. expect the image.shape to be a tuple like
        (1, 320, 180), not (3, 320, 180). The pixel values are rescaled to 0-255 range.
        The dicom images provided by contest organizers have more than 256 possible
        pixel values, thus if images_folder contains dicom (.dcm) files, expect
        fractional pixel values.

        Parameters
        ----------
        metadata_csv_path : Path
            path to csv file with samples metadata
        images_dir : Path
            folder where the images are stored (dcm/png/jpeg)
        """
        super().__init__()
        self.metadata = pd.read_csv(metadata_csv_path)
        self.images_dir = Path(images_dir)
        assert "cancer" in self.metadata.columns

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return _get_tensor_image(
            self.images_dir,
            self.metadata.patient_id[index],
            self.metadata.image_id[index],
        ), torch.tensor(self.metadata.cancer[index], dtype=torch.int)


class RSNABreastCancerTestDataset(Dataset[torch.Tensor]):
    def __init__(self, metadata_csv_path: Path, images_dir: Path) -> None:
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
        metadata_csv_path : Path
            path to csv file with samples metadata
        images_dir : Path
            folder where the images are stored (dcm/png/jpeg)
        """
        super().__init__()
        self.metadata = pd.read_csv(metadata_csv_path)
        self.images_dir = Path(images_dir)
        assert "cancer" not in self.metadata.columns

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> torch.Tensor:
        return _get_tensor_image(
            self.images_dir,
            self.metadata.patient_id[index],
            self.metadata.image_id[index],
        )


class LightningDataModule(pl.LightningDataModule):
    def __init__(
        self,
        images_dir: Path,
        metadata_csv_path: Path,
        batch_size: int = 32,
        oversample_train_dataset: bool = False,
        val_split_size_factor: float = 0.1,
        split_random_state: int = 0,
        resampling_random_state: int = 0,
    ):
        super().__init__()

        assert 0 <= val_split_size_factor < 1
        self.batch_size = batch_size
        self.oversample_train_dataset = oversample_train_dataset
        self.val_split_size_factor = val_split_size_factor
        self.split_random_state = split_random_state
        self.resampling_random_state = resampling_random_state

        self.dataset = RSNABreastCancerTrainDataset(
            images_dir=images_dir,
            metadata_csv_path=metadata_csv_path,
        )

        train_idx, val_idx = self.__get_train_val_splits(
            dataset=self.dataset,
            val_split_size_factor=self.val_split_size_factor,
            split_random_state=self.split_random_state,
            oversample_train_dataset=self.oversample_train_dataset,
            resampling_random_state=self.resampling_random_state,
        )
        self.train_dataset = Subset(self.dataset, train_idx)
        self.val_dataset = Subset(self.dataset, val_idx)

    def __get_train_val_splits(
        self,
        dataset: RSNABreastCancerTrainDataset,
        val_split_size_factor: float,
        split_random_state: int,
        oversample_train_dataset: bool,
        resampling_random_state: int,
    ) -> tuple[list[int], list[int]]:
        # we can't just split indexes, we should split by patients:
        patients_to_stratify = dataset.metadata.groupby(by="patient_id")[
            ["cancer"]
        ].max()  # a dataframe: index: "patient_id"; columns: ["cancer"]

        train_patients_idx, val_patients_idx = train_test_split(
            patients_to_stratify.index,
            test_size=val_split_size_factor,
            random_state=split_random_state,
            stratify=patients_to_stratify.cancer,
            shuffle=True,
        )

        train_idx = dataset.metadata[
            dataset.metadata.patient_id.isin(train_patients_idx)
        ].index.to_numpy()

        val_idx = dataset.metadata[
            dataset.metadata.patient_id.isin(val_patients_idx)
        ].index.to_numpy()

        if oversample_train_dataset:
            random_over_sampler = RandomOverSampler(
                random_state=resampling_random_state
            )
            train_idx, _ = random_over_sampler.fit_resample(
                train_idx.reshape(-1, 1),
                dataset.metadata.iloc[train_idx].cancer,
            )
            train_idx = train_idx.squeeze()

        return train_idx.tolist(), val_idx.tolist()

    def train_dataloader(self) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
