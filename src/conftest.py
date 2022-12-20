from pathlib import Path

import pandas as pd
from pydicom import FileDataset, dcmread
from pytest import fixture

MOCK_DICOM_FILEPATH = Path("src/tests/200779059.dcm")
MOCK_TRAIN_DATA_FILEPATH = Path("src/tests/mock_train_data.csv")


@fixture(scope="session")
def mock_dicom() -> FileDataset:
    return dcmread(MOCK_DICOM_FILEPATH)


@fixture(scope="session")
def mock_train_data() -> pd.DataFrame:
    return pd.read_csv(MOCK_TRAIN_DATA_FILEPATH)
