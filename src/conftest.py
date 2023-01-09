from pathlib import Path

import pandas as pd
from pydicom import FileDataset, dcmread
from pytest import fixture

from src.nn.data import (
    LightningDataModule,
    RSNABreastCancerTestDataset,
    RSNABreastCancerTrainDataset,
)

MOCK_DICOM_FILEPATH = Path("src/tests/mock_dicoms/10008/200779059.dcm")
MOCK_DICOMS_DIR = Path("src/tests/mock_dicoms/")
MOCK_TRAIN_METADATA_FILEPATH = Path("src/tests/mock_train_data.csv")
MOCK_TEST_METADATA_FILEPATH = Path("src/tests/mock_test_data.csv")


@fixture(scope="session")
def mock_dicom() -> FileDataset:
    return dcmread(MOCK_DICOM_FILEPATH)


@fixture(scope="session")
def mock_train_metadata() -> pd.DataFrame:
    return pd.read_csv(MOCK_TRAIN_METADATA_FILEPATH)


@fixture(scope="session")
def mock_train_dataset() -> RSNABreastCancerTrainDataset:
    return RSNABreastCancerTrainDataset(
        metadata_csv_path=MOCK_TRAIN_METADATA_FILEPATH,
        images_dir=MOCK_DICOMS_DIR,
    )


@fixture(scope="session")
def mock_test_dataset() -> RSNABreastCancerTestDataset:
    return RSNABreastCancerTestDataset(
        metadata_csv_path=MOCK_TEST_METADATA_FILEPATH,
        images_dir=MOCK_DICOMS_DIR,
    )


@fixture(scope="session")
def mock_lightning_data_module() -> LightningDataModule:
    return LightningDataModule(
        images_dir=MOCK_DICOMS_DIR,
        metadata_csv_path=MOCK_TRAIN_METADATA_FILEPATH,
        batch_size=1,
        oversample_train_dataset=False,
        val_split_size_factor=0.5,
        split_random_state=0,
    )


@fixture(scope="session")
def mock_lightning_data_module_with_oversampling() -> LightningDataModule:
    return LightningDataModule(
        images_dir=MOCK_DICOMS_DIR,
        metadata_csv_path=MOCK_TRAIN_METADATA_FILEPATH,
        batch_size=1,
        oversample_train_dataset=True,
        val_split_size_factor=0.5,
        split_random_state=0,
    )
