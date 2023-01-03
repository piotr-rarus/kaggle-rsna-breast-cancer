from pathlib import Path

import pandas as pd
from pydicom import FileDataset, dcmread
from pytest import fixture

MOCK_DICOM_FILEPATH = Path("src/tests/mock_dicoms/200779059.dcm")
MOCK_TRAIN_DATA_FILEPATH = Path("src/tests/mock_train_data.csv")
MOCK_TEST_DATA_FILEPATH = Path("src/tests/mock_test_data.csv")


@fixture(scope="session")
def mock_dicom() -> FileDataset:
    return dcmread(MOCK_DICOM_FILEPATH)


@fixture(scope="session")
def mock_train_data() -> pd.DataFrame:
    return pd.read_csv(MOCK_TRAIN_DATA_FILEPATH)


@fixture(scope="session")
def mock_dicoms_folderpath() -> Path:
    return MOCK_DICOM_FILEPATH.parent


@fixture(scope="session")
def mock_test_csv_path() -> Path:
    return MOCK_TEST_DATA_FILEPATH


@fixture(scope="session")
def mock_train_csv_path() -> Path:
    return MOCK_TRAIN_DATA_FILEPATH
