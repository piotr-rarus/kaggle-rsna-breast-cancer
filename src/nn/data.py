from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from numpy.typing import NDArray
from pydicom import dcmread
from sklearn.model_selection import StratifiedKFold  # type: ignore
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
        train_val_split_id: int = 0,
        val_dataset_factor: int = 10,
        random_state: int = 0,
    ):
        """Datamodule for use with PytorchLightning.
        The training set is split into new training set and a validation set.
        The split is stratified based on patients that have cancer.
        The split is done like for cross validation (default 10-fold).
        The n-th fold from "cross validation" folds can be chosen
        using optional train_val_split_id (default 0).

        Parameters
        ----------
        images_dir : Path
            Folder where images are stored (like "data/images_64" or "data/images_512").
        metadata_csv_path : Path
            Path to csv file with training metadata. Defaults to "data/train.csv".
        batch_size : int, optional
            Number of samples in a batch, by default 32
        oversample_train_dataset : bool, optional
            If True, the newly separated train subset will oversample the cancer cases
            such that the apparent number of patients with diagnosed cancer will be
            close to number of patients without cancer, by default False
        train_val_split_id : int, optional
            The train-val split is done in k-fold manner. The train_val_split_id allows
            to choose different folds, by default 0
        val_dataset_factor : int, optional
            The train-val split is done in k-fold manner. The val_dataset_factor
            changes the size of K used in k-fold split, by default 10
        random_state : int, optional
            (seed) used only for shuffling the samples in newly separated train subset,
            by default 0
        """

        assert 0 <= train_val_split_id and train_val_split_id < val_dataset_factor
        super().__init__()
        self.batch_size = batch_size
        # instead of stratified train_test_split, we use stratified KFold
        # this way we may make a cross-validation if we'd like to
        full_train_dataset = RSNABreastCancerTrainDataset(
            images_dir=images_dir,
            metadata_csv_path=metadata_csv_path,
        )

        # we can't just split indexes, we should split by patients:
        patients_to_stratify = full_train_dataset.metadata.groupby(by="patient_id")[
            ["cancer"]
        ].max()  # a dataframe: index: "patient_id"; columns: ["cancer"]
        skf = StratifiedKFold(n_splits=val_dataset_factor)
        train_patients_idxs, val_patients_idxs = list(
            skf.split(patients_to_stratify, patients_to_stratify.cancer)
        )[train_val_split_id]
        train_patients = patients_to_stratify.iloc[train_patients_idxs]
        val_patients = patients_to_stratify.iloc[val_patients_idxs]
        train_idxs = full_train_dataset.metadata[
            full_train_dataset.metadata.patient_id.isin(train_patients.index)
        ].index.to_numpy()
        val_idxs = full_train_dataset.metadata[
            full_train_dataset.metadata.patient_id.isin(val_patients.index)
        ].index.to_numpy()

        if oversample_train_dataset:
            patients_to_oversample = train_patients[
                train_patients.cancer == 1
            ].index.to_numpy()
            oversample_multiplier = (
                len(train_patients) // len(patients_to_oversample) - 1
            )  # will raise value error if there are no patients with cancer
            train_idxs_to_oversample = full_train_dataset.metadata[
                full_train_dataset.metadata.patient_id.isin(patients_to_oversample)
            ].index.to_numpy()
            train_idxs = np.concatenate(
                [train_idxs] + [train_idxs_to_oversample] * oversample_multiplier
            )
        np.random.default_rng(seed=random_state).shuffle(train_idxs)
        self.train_dataset = Subset(full_train_dataset, train_idxs)
        self.val_dataset = Subset(full_train_dataset, val_idxs)

    def train_dataloader(self) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
