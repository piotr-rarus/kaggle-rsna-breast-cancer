from pathlib import Path

import torch

from src.nn.data import RSNABreastCancerDataset


def test_dataset(
    mock_dicoms_folderpath: Path, mock_test_csv_path: Path, mock_train_csv_path: Path
) -> None:
    # reading dicoms
    # no labels case (no "cancer" column in .csv)
    dicom_dataset = RSNABreastCancerDataset(
        labels_csv_path=mock_test_csv_path, images_folder=mock_dicoms_folderpath
    )
    assert len(dicom_dataset) == 1
    image = dicom_dataset[0]
    assert image.shape == (2776, 2082)
    assert image.dtype == torch.float
    assert torch.allclose(image.mean(), torch.tensor(7.7473), atol=1e-3)
    assert torch.allclose(image.std(), torch.tensor(25.9267), atol=1e-3)

    # reading pngs
    # labels present ("cancer" column in .csv)
    png64_train_dataset = RSNABreastCancerDataset(
        labels_csv_path=mock_train_csv_path, images_folder=mock_dicoms_folderpath
    )
    assert len(png64_train_dataset) == 13
    image, target = png64_train_dataset[0]
    assert target == 0
    assert image.shape == (64, 58)
    assert image.dtype == torch.float
    assert torch.allclose(image.mean(), torch.tensor(16.3004), atol=1e-3)
    assert torch.allclose(image.std(), torch.tensor(45.7319), atol=1e-3)
