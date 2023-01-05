import pytorch_lightning as pl

from src.nn.classifier import LightningClassifier
from src.nn.data import LightningDataModule
from src.nn.models.dummy import DummyModel


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
