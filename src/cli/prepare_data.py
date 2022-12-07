import shutil
from pathlib import Path

import click
import cv2
import numpy as np
from joblib import Parallel, delayed
from pydicom import dcmread
from tqdm.auto import tqdm

from src.util import get_logger

logger = get_logger(__name__)


@click.command()
@click.option(
    "--data-dir",
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
    "--output-dir",
    type=click.Path(
        exists=False,
        file_okay=False,
        dir_okay=True,
        writable=True,
        readable=True,
        path_type=Path,
    ),
    required=True,
)
@click.option("--resolution", type=click.INT)
@click.option("--n-jobs", type=click.INT)
def main(data_dir: Path, output_dir: Path, resolution: int, n_jobs: int) -> None:
    logger.debug(f"Initializing output data structure: {output_dir}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
        logger.debug("Found existing dir. Removing existing files.")
    output_dir.mkdir(exist_ok=True, parents=True)

    with Parallel(n_jobs=n_jobs) as parallel:
        filepaths = list(data_dir.rglob("*.dcm"))

        parallel(
            delayed(__prepare_dicom)(dicom_path, output_dir, resolution)
            for dicom_path in tqdm(filepaths)
        )


def __prepare_dicom(dicom_path: Path, output_dir: Path, resolution: int) -> None:
    dicom = dcmread(dicom_path)
    image = dicom.pixel_array.astype(float)
    image = image / image.max() * 255
    image = image.astype(np.uint8)
    image_dir = output_dir / dicom_path.parent.name
    image_dir.mkdir(exist_ok=True, parents=True)
    image_path = image_dir / f"{dicom_path.stem}.png"

    height, width = image.shape[0], image.shape[1]
    width = int(width * resolution / height)
    image = cv2.resize(image, dsize=(width, resolution), interpolation=cv2.INTER_AREA)

    cv2.imwrite(str(image_path), image)


if __name__ == "__main__":
    main()
