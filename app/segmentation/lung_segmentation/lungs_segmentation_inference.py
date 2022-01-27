import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Union, List
from pydicom import FileDataset

from app.segmentation.lung_segmentation.model import (
    lungs_transforms,
    lungs_model,
)
from app.segmentation.utils import get_pixel_spacing_and_slice_thickness_in_centimeters


def lung_segmentation_inference(instances: List[FileDataset]):
    instances.sort(key=lambda instance: instance.InstanceNumber)
    (
        pixel_spacing_x_cm,
        pixel_spacing_y_cm,
        slice_thickness_cm,
    ) = get_pixel_spacing_and_slice_thickness_in_centimeters(instances[0])

    # (width, height, frames)
    series_array = np.stack([i.pixel_array for i in instances], axis=-1)
    # (frames, width, height)
    inference_result = get_lungs_masks(
        series_array, pixel_spacing_x_cm, pixel_spacing_y_cm, slice_thickness_cm
    )

    lung_masks = inference_result["lung_masks"]
    lung_volume_cm = inference_result["lung_volume_cm"]

    lung_segmentation_metadata = {
        "dataType": "text",
        "value": f"{lung_volume_cm} cm³",
        "title": "Lung volume",
        "description": "Lung volume in cm³ calculated from lung_segmentation masks",
    }

    return {
        "lung_masks": lung_masks,
        "lung_segmentation_metadata": lung_segmentation_metadata,
    }


@torch.no_grad()
def get_lungs_masks(
    ct_scan: np.ndarray,
    pixel_spacing_x_cm: float,
    pixel_spacing_y_cm: float,
    slice_thickness_cm: float,
) -> Dict[str, Union[np.darray, float]]:
    """
    Given an np.array of frames from a CT scan, the method returns the lung lung_segmentation for each frame.
    Nvidia clara model available here: https://ngc.nvidia.com/catalog/models/nvidia:med:clara_pt_covid19_ct_lung_segmentation
    is used for inference. The 32 frames are sampled from the input DICOM and used as model input.
    The masks for the remaining frames are then interpolated. Based on the segmented masks, and the
    physical parameters from the DICOM tags we calculate the lungs volume

    :param ct_scan: # (width, height, frames) array from a DICOM image, the pixel values ranges from 0 up to 4096
    :param pixel_spacing_x_cm: Physical distance in the patient between the center of each pixel in cm
    :param pixel_spacing_y_cm: Physical distance in the patient between the center of each pixel in cm
    :param slice_thickness_cm:  Nominal slice thickness, in cm
    :return: (frames, width, height) array of 0s and 1s with the segmented lungs
    """
    width, height, number_of_frames = ct_scan.shape

    input_ = lungs_transforms(ct_scan)[0].unsqueeze(0)

    output_lungs = lungs_model(input_)
    # (batch_size, num_classes, width, height, frames) -> (batch_size, num_classes, frames, width, height)
    output_lungs = output_lungs.permute(0, 1, 4, 2, 3)

    output_lungs_interpolated = F.interpolate(
        output_lungs,
        size=(number_of_frames, width, height),
        mode="trilinear",
        align_corners=True,
    )

    lung_masks = (
        torch.argmax(output_lungs_interpolated[0], 0).cpu().numpy().astype(np.uint8)
    )

    # volume of lungs is
    # a (number of pixels classified as "lung") times
    # (pixel spacing in x axis) times (pixel spacing in y axis)
    # times slice thickness
    lung_volume_cm = (
        np.sum(lung_masks)
        * pixel_spacing_x_cm
        * pixel_spacing_y_cm
        * slice_thickness_cm
    )

    return {"lung_masks": lung_masks, "lung_volume_cm": lung_volume_cm}
