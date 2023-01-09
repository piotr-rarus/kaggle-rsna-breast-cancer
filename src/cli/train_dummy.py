from pathlib import Path

import click
from pytorch_lightning import Trainer, seed_everything

from src.nn.classifier import LightningClassifier
from src.nn.data import LightningDataModule
from src.nn.models.dummy import DummyModel


@click.command()
@click.option(
    "--images-dir",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        writable=False,
        readable=True,
        path_type=Path,
    ),
    required=True,
)
@click.option(
    "--metadata-csv-path",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        writable=False,
        readable=True,
        path_type=Path,
    ),
    required=True,
)
@click.option(
    "--oversample-train-dataset", is_flag=True, show_default=True, default=False
)
@click.option("--random-state", required=True, type=int, default=42, show_default=True)
@click.option(
    "--split-random-state", required=True, type=int, default=0, show_default=True
)
@click.option(
    "--val-split-size-factor", required=True, type=float, default=0.1, show_default=True
)
@click.option("--max-epochs", required=True, type=int, default=100, show_default=True)
def main(
    images_dir: Path,
    metadata_csv_path: Path,
    oversample_train_dataset: bool = False,
    random_state: int = 42,
    split_random_state: int = 0,
    val_split_size_factor: float = 0.1,
    max_epochs: int = 100,
) -> None:
    seed_everything(random_state)

    data_module = LightningDataModule(
        images_dir=images_dir,
        metadata_csv_path=metadata_csv_path,
        batch_size=128,
        oversample_train_dataset=oversample_train_dataset,
        val_split_size_factor=val_split_size_factor,
        split_random_state=split_random_state,
        num_workers=4,
    )

    trainer = Trainer(
        max_epochs=max_epochs,
        logger=False,
        enable_checkpointing=False,
    )

    model = LightningClassifier(backbone=DummyModel(), num_classes=2)
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
