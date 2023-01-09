from pathlib import Path

import pandas as pd
from pydicom import FileDataset, dcmread
from pytest import fixture

from src.nn.data import (
    LightningDataModule,
    RSNABreastCancerTestDataset,
    RSNABreastCancerTrainDataset,
)


@fixture(scope="session")
def mock_dicom_filepath() -> Path:
    return Path("src/tests/mock_dicoms/10008/200779059.dcm")


@fixture(scope="session")
def mock_dicoms_dir() -> Path:
    return Path("src/tests/mock_dicoms/")


@fixture(scope="session")
def mock_train_metadata_filepath() -> Path:
    return Path("src/tests/mock_train_data.csv")


@fixture(scope="session")
def mock_test_metadata_filepath() -> Path:
    return Path("src/tests/mock_test_data.csv")


@fixture(scope="session")
def mock_dicom(mock_dicom_filepath: Path) -> FileDataset:
    return dcmread(mock_dicom_filepath)


@fixture(scope="session")
def mock_train_metadata(mock_train_metadata_filepath: Path) -> pd.DataFrame:
    return pd.read_csv(mock_train_metadata_filepath)


@fixture(scope="session")
def mock_train_dataset(
    mock_train_metadata_filepath: Path, mock_dicoms_dir: Path
) -> RSNABreastCancerTrainDataset:
    return RSNABreastCancerTrainDataset(
        metadata_csv_path=mock_train_metadata_filepath,
        images_dir=mock_dicoms_dir,
    )


@fixture(scope="session")
def mock_test_dataset(
    mock_test_metadata_filepath: Path, mock_dicoms_dir: Path
) -> RSNABreastCancerTestDataset:
    return RSNABreastCancerTestDataset(
        metadata_csv_path=mock_test_metadata_filepath,
        images_dir=mock_dicoms_dir,
    )


@fixture(scope="session")
def mock_lightning_data_module(
    mock_dicoms_dir: Path, mock_train_metadata_filepath: Path
) -> LightningDataModule:
    return LightningDataModule(
        images_dir=mock_dicoms_dir,
        metadata_csv_path=mock_train_metadata_filepath,
        batch_size=1,
        oversample_train_dataset=False,
        val_split_size_factor=0.5,
        split_random_state=0,
    )


@fixture(scope="session")
def mock_lightning_data_module_with_oversampling(
    mock_dicoms_dir: Path, mock_train_metadata_filepath: Path
) -> LightningDataModule:
    return LightningDataModule(
        images_dir=mock_dicoms_dir,
        metadata_csv_path=mock_train_metadata_filepath,
        batch_size=1,
        oversample_train_dataset=True,
        val_split_size_factor=0.5,
        split_random_state=0,
    )
