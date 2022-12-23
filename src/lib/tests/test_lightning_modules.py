from pathlib import Path

import pytorch_lightning as pl
import torch

from src.backbone_models.dummy import DummyModel
from src.lib.data import LightningDataModule
from src.lib.lightning_model import LightningCLF


def test_dataloader(mock_dicoms_folderpath: Path, mock_train_csv_path: Path) -> None:
    # reading dicoms
    # no labels case (no "cancer" column in .csv)
    dm = LightningDataModule(
        batch_size=1,
        random_state=0,
        val_dataset_factor=2,
        labels_csv_path=mock_train_csv_path,
        images_folder=mock_dicoms_folderpath,
        oversample_train_dataset=False,
    )
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    assert len(train_loader) == 5
    assert len(val_loader) == 8
    first_batch = next(iter(train_loader))
    assert torch.allclose(first_batch[0][0].mean(), torch.tensor(16.3004), atol=1e-3)

    # with oversampling
    dm = LightningDataModule(
        batch_size=1,
        random_state=0,
        val_dataset_factor=2,
        labels_csv_path=mock_train_csv_path,
        images_folder=mock_dicoms_folderpath,
        oversample_train_dataset=True,
    )
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    assert len(train_loader) == 6
    assert len(val_loader) == 8
    first_batch = next(iter(train_loader))
    assert torch.allclose(first_batch[0][0].mean(), torch.tensor(16.3004), atol=1e-3)


def test_pl_model(mock_dicoms_folderpath: Path, mock_train_csv_path: Path) -> None:
    dm = LightningDataModule(
        batch_size=2,
        random_state=0,
        val_dataset_factor=2,
        labels_csv_path=mock_train_csv_path,
        images_folder=mock_dicoms_folderpath,
        oversample_train_dataset=True,
    )
    trainer = pl.Trainer(
        max_epochs=3,
        logger=False,
        enable_checkpointing=False,
    )
    model = LightningCLF(backbone=DummyModel(), num_classes=2)
    trainer.fit(model, datamodule=dm)
