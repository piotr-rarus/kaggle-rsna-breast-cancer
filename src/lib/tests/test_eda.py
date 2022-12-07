import pandas as pd
from _pytest.tmpdir import TempPathFactory

from src.lib.eda import profile


def test_profile(
    mock_train_data: pd.DataFrame, tmp_path_factory: TempPathFactory
) -> None:
    eda_dir = tmp_path_factory.mktemp("eda")
    profile(
        mock_train_data,
        name="profile_test",
        dump_dir=eda_dir,
        explorative=False,
        minimal=True,
    )
    profile_path = eda_dir / "profile_test.html"
    assert profile_path.exists()
