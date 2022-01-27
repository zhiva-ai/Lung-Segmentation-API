from pydicom import FileDataset
from typing import Tuple


def get_pixel_spacing_and_slice_thickness_in_centimeters(
    instance: FileDataset,
) -> Tuple[float, float, float]:
    """
    :param instance: example pydicom instance, one from the
    :return:
    """
    pixel_spacing_x, pixel_spacing_y = instance.PixelSpacing
    pixel_spacing_x_cm, pixel_spacing_y_cm = pixel_spacing_x / 10, pixel_spacing_y / 10

    slice_thickness = instance.SliceThickness
    slice_thickness_cm = slice_thickness / 10

    return pixel_spacing_x_cm, pixel_spacing_y_cm, slice_thickness_cm
