import torch

from src.nn.data import RSNABreastCancerTestDataset, RSNABreastCancerTrainDataset


def test_dicom_test_dataset(
    mock_test_dataset: RSNABreastCancerTestDataset,
) -> None:
    # reading dicoms
    # no labels case (no "cancer" column in .csv)
    assert len(mock_test_dataset) == 1
    image = mock_test_dataset[0]
    assert image.shape == (2776, 2082)
    assert image.dtype == torch.float
    assert torch.allclose(image.mean(), torch.tensor(7.7473), atol=1e-3)
    assert torch.allclose(image.std(), torch.tensor(25.9267), atol=1e-3)


def test_png_train_dataset(
    mock_train_dataset: RSNABreastCancerTrainDataset,
) -> None:
    # reading pngs
    # labels present ("cancer" column in .csv)
    assert len(mock_train_dataset) == 13
    image, target = mock_train_dataset[0]
    assert target == 0
    assert image.shape == (64, 58)
    assert image.dtype == torch.float
    assert torch.allclose(image.mean(), torch.tensor(16.3004), atol=1e-3)
    assert torch.allclose(image.std(), torch.tensor(45.7319), atol=1e-3)
