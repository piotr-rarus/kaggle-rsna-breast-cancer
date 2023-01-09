from pathlib import Path

import cv2
import numpy as np
from joblib import Parallel, delayed
from pydicom import dcmread
from tqdm.auto import tqdm

from src.lib.cv import normalize_dicom


def prepare_data(
    data_dir: Path, output_dir: Path, resolution: int, n_jobs: int
) -> None:
    with Parallel(n_jobs=n_jobs) as parallel:
        filepaths = list(data_dir.rglob("*.dcm"))

        parallel(
            delayed(__prepare_dicom)(dicom_path, output_dir, resolution)
            for dicom_path in tqdm(filepaths)
        )


def __prepare_dicom(dicom_path: Path, output_dir: Path, resolution: int) -> None:
    dicom = dcmread(dicom_path)
    image = normalize_dicom(dicom) * 255
    image = image.astype(np.uint8)

    height, width = image.shape[0], image.shape[1]
    width = int(width * resolution / height)
    image = cv2.resize(image, dsize=(width, resolution), interpolation=cv2.INTER_AREA)

    patient_id = dicom_path.parent.name
    image_path = output_dir / patient_id / f"{dicom_path.stem}.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(image_path), image)
