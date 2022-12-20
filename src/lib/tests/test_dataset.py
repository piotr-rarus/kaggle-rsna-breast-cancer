import torch

from src.lib.data import RSNABreastCancerDataset


def test_dataset() -> None:
    # reading dicoms
    # no labels case (no "cancer" column in .csv)
    dicom_dataset = RSNABreastCancerDataset(
        labels_csv_path="data/test.csv", images_folder="data/test"
    )
    assert len(dicom_dataset) == 4
    image, dummy_label_for_test_dataset = dicom_dataset[0]
    assert dummy_label_for_test_dataset == 0
    assert image.shape == (2776, 2082)
    assert image.dtype == torch.float
    assert torch.allclose(image.mean(), torch.tensor(52.5057), atol=1e-3)
    assert torch.allclose(image.std(), torch.tensor(55.5450), atol=1e-3)

    # reading pngs
    # labels present ("cancer" column in .csv)
    png64_train_dataset = RSNABreastCancerDataset(
        labels_csv_path="data/train.csv", images_folder="data/train_64"
    )
    assert len(png64_train_dataset) == 54706
    image, target = png64_train_dataset[0]
    assert target == 0
    assert image.shape == (64, 58)
    assert image.dtype == torch.float
    assert torch.allclose(image.mean(), torch.tensor(16.3004), atol=1e-3)
    assert torch.allclose(image.std(), torch.tensor(45.7319), atol=1e-3)
