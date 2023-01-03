from pathlib import Path

import pandas as pd
from pydicom import FileDataset, dcmread
from pytest import fixture

from src.nn.data import RSNABreastCancerTestDataset, RSNABreastCancerTrainDataset

MOCK_DICOM_FILEPATH = Path("src/tests/mock_dicoms/10008/200779059.dcm")
MOCK_DICOMS_DIR = Path("src/tests/mock_dicoms/")
MOCK_TRAIN_METADATA_FILEPATH = Path("src/tests/mock_train_data.csv")
MOCK_TEST_METADATA_FILEPATH = Path("src/tests/mock_test_data.csv")


@fixture(scope="session")
def mock_dicom() -> FileDataset:
    return dcmread(MOCK_DICOM_FILEPATH)


@fixture(scope="session")
def mock_train_data() -> pd.DataFrame:
    return pd.read_csv(MOCK_TRAIN_METADATA_FILEPATH)


@fixture(scope="session")
def mock_train_dataset() -> RSNABreastCancerTrainDataset:
    return RSNABreastCancerTrainDataset(
        labels_csv_path=MOCK_TRAIN_METADATA_FILEPATH,
        images_folder=MOCK_DICOMS_DIR,
    )


@fixture(scope="session")
def mock_test_dataset() -> RSNABreastCancerTestDataset:
    return RSNABreastCancerTestDataset(
        labels_csv_path=MOCK_TEST_METADATA_FILEPATH,
        images_folder=MOCK_DICOMS_DIR,
    )
