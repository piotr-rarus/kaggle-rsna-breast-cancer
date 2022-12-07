from pydicom import FileDataset

from src.lib.cv import normalize_dicom


def test_normalize_dicom(mock_dicom: FileDataset) -> None:
    image = normalize_dicom(mock_dicom)
    assert image.min() >= 0
    assert image.max() <= 1
