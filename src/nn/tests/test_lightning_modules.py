import pytorch_lightning as pl
import torch

from src.nn.data import LightningDataModule
from src.nn.lightning_model import LightningClassifier
from src.nn.models.dummy import DummyModel


def test_dataloader(mock_lightning_data_module: LightningDataModule) -> None:
    # reading dicoms
    # no labels case (no "cancer" column in .csv)
    train_loader = mock_lightning_data_module.train_dataloader()
    val_loader = mock_lightning_data_module.val_dataloader()
    assert len(train_loader) == 5
    assert len(val_loader) == 8
    first_batch = next(iter(train_loader))
    assert torch.allclose(first_batch[0][0].mean(), torch.tensor(16.3004), atol=1e-3)


def test_dataloader_with_oversampling(
    mock_lightning_data_module_with_oversampling: LightningDataModule,
) -> None:
    # with oversampling
    train_loader = mock_lightning_data_module_with_oversampling.train_dataloader()
    val_loader = mock_lightning_data_module_with_oversampling.val_dataloader()
    assert len(train_loader) == 6
    assert len(val_loader) == 8
    first_batch = next(iter(train_loader))
    assert torch.allclose(first_batch[0][0].mean(), torch.tensor(16.3004), atol=1e-3)


def test_pytorch_lightning_model(
    mock_lightning_data_module_with_oversampling: LightningDataModule,
) -> None:
    trainer = pl.Trainer(
        max_epochs=3,
        logger=False,
        enable_checkpointing=False,
    )
    model = LightningClassifier(backbone=DummyModel(), num_classes=2)
    trainer.fit(model, datamodule=mock_lightning_data_module_with_oversampling)
