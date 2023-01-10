from pathlib import Path

import cv2
from _pytest.tmpdir import TempPathFactory

from src.preparation import prepare_data


def test_prepare_data(tmp_path_factory: TempPathFactory, mock_dicoms_dir: Path) -> None:
    output_dir = tmp_path_factory.mktemp("preparation_32")
    prepare_data(
        data_dir=mock_dicoms_dir, output_dir=output_dir, resolution=64, n_jobs=1
    )
    files = list(output_dir.rglob("*.png"))
    assert len(files) == 1
    image_filepath = files[0]
    assert image_filepath.stem == "200779059"
    assert image_filepath.suffix == ".png"
    prepared_image = cv2.imread(str(image_filepath), cv2.IMREAD_GRAYSCALE)
    assert prepared_image.shape == (64, 48)
    assert prepared_image.max() >= 220
