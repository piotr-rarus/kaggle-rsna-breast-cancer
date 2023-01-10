import numpy as np
from numpy.typing import NDArray
from pydicom import FileDataset


def normalize_dicom(
    dicom: FileDataset, lower_percentile: float = 0.01, upper_percentile: float = 0.99
) -> NDArray[np.float64]:
    image = dicom.pixel_array

    if dicom.PhotometricInterpretation == "MONOCHROME1":
        image = 1 - image

    histogram, bins = np.histogram(image, bins=512)
    bins = bins[1:]
    # we do histogram[1:] as all the lowest-valued pixels are background,
    # the background should have as little influence on normalization as possible
    pdf = histogram[1:] / histogram[1:].sum()
    cdf = pdf.cumsum()

    new_lowest = bins[np.abs(cdf - lower_percentile).argmin()]
    median = bins[np.abs(cdf - 0.5).argmin()]
    new_highest = bins[np.abs(cdf - upper_percentile).argmin()]

    image[image == image.min()] = median  # move background to median
    image = image.clip(min=new_lowest, max=new_highest)

    image = image - image.min()
    image = image / image.max()

    return image
