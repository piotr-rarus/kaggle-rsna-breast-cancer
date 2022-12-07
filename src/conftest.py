from pathlib import Path

import pandas as pd
from pytest import fixture

MOCK_TRAIN_DATA_FILEPATH = Path("src//tests/mock_train_data.csv")


@fixture(scope="session")
def mock_train_data() -> pd.DataFrame:
    return pd.read_csv(MOCK_TRAIN_DATA_FILEPATH)
