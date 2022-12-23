from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import StratifiedKFold  # type: ignore
from torch.utils.data import DataLoader, Dataset, Subset

from src.lib.cv import read_dicom_and_normalize


class RSNABreastCancerDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, labels_csv_path: str | Path, images_folder: str | Path) -> None:
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

        Args:
            labels_csv_path (Path): path to csv file with samples metadata
            images_folder (Path): folder where the images are stored (dcm/png/jpeg)
        """
        super().__init__()
        self.metadata = pd.read_csv(labels_csv_path)
        self.images_folder = Path(images_folder)
        self.labels = (
            torch.zeros(len(self.metadata)).int()
            if "cancer" not in self.metadata.columns
            else torch.from_numpy(self.metadata.cancer.to_numpy())
        )

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        impath: Path = next(
            self.images_folder.rglob(f"{self.metadata.image_id[index]}.*")
        )
        if impath.suffix == ".dcm":
            numpy_image = read_dicom_and_normalize(impath)
        else:
            numpy_image = cv2.imread(str(impath), cv2.IMREAD_GRAYSCALE)
        image = torch.tensor(numpy_image, dtype=torch.float)
        return image.unsqueeze(0), self.labels[index]


class LightningDataModule(pl.LightningDataModule):
    def __init__(
        self,
        images_folder: str | Path,
        labels_csv_path: str | Path,
        batch_size: int = 32,
        oversample_train_dataset: bool = False,
        train_val_split_id: int = 0,
        val_dataset_factor: int = 10,
        random_state: int | None = None,
    ):
        assert 0 <= train_val_split_id and train_val_split_id < val_dataset_factor
        super().__init__()
        self.batch_size = batch_size
        # instead of stratified train_test_split, we use stratified KFold
        # this way we may make a cross-validation if we'd like to
        full_train_dataset = RSNABreastCancerDataset(
            labels_csv_path=labels_csv_path,
            images_folder=images_folder,
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
