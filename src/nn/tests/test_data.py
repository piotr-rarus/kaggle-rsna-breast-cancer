import torch

from src.nn.data import (
    LightningDataModule,
    RSNABreastCancerTestDataset,
    RSNABreastCancerTrainDataset,
)


def test_dicom_test_dataset(
    mock_test_dataset: RSNABreastCancerTestDataset,
) -> None:
    # reading dicoms
    # no labels case (no "cancer" column in .csv)
    assert len(mock_test_dataset) == 1
    image = mock_test_dataset[0]
    assert image.shape == (1, 2776, 2082)
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
    assert image.shape == (1, 64, 58)
    assert image.dtype == torch.float
    assert torch.allclose(image.mean(), torch.tensor(16.3004), atol=1e-3)
    assert torch.allclose(image.std(), torch.tensor(45.7319), atol=1e-3)


def test_dataloader(mock_lightning_data_module: LightningDataModule) -> None:
    # reading dicoms
    # no labels case (no "cancer" column in .csv)
    train_loader = mock_lightning_data_module.train_dataloader()
    val_loader = mock_lightning_data_module.val_dataloader()
    assert len(train_loader) == 8
    assert len(val_loader) == 5
    first_batch = next(iter(train_loader))
    assert torch.allclose(first_batch[0][0].mean(), torch.tensor(16.3004), atol=1e-3)


def test_dataloader_with_oversampling(
    mock_lightning_data_module_with_oversampling: LightningDataModule,
) -> None:
    # with oversampling
    train_loader = mock_lightning_data_module_with_oversampling.train_dataloader()
    val_loader = mock_lightning_data_module_with_oversampling.val_dataloader()
    assert len(train_loader) == 14
    assert len(val_loader) == 5
    first_batch = next(iter(train_loader))
    assert torch.allclose(first_batch[0][0].mean(), torch.tensor(16.3004), atol=1e-3)
