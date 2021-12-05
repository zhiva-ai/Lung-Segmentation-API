from pathlib import Path

from monai.transforms import (
    Compose,
    AddChannel,
    Resize,
    Spacing,
    ScaleIntensityRange,
    CropForeground,
    ToTensor,
)
from monai.networks.nets.ahnet import Ahnet
import torch
import math
import numpy as np
import cv2

NUM_SLICES = 32

tranform_one = Compose(
    [
        AddChannel(),
    ]
)

tranform_two = Compose(
    [
        AddChannel(),
        Spacing(pixdim=[0.8, 0.8, 5], mode="bilinear", align_corners=True),
    ]
)

transforms = Compose(
    [
        AddChannel(),
        Spacing(pixdim=[0.8, 0.8, 5], mode="bilinear", align_corners=True),
        ScaleIntensityRange(a_min=-1000, a_max=500, b_min=0, b_max=1),
        CropForeground(source_key="image"),
        Resize((224, 224, NUM_SLICES)),
        ToTensor(),
    ]
)

back_to_original_size_transforms = Compose(
    [
        AddChannel(),
        Spacing(pixdim=[1.0, 1.0, 1.0], mode="bilinear", align_corners=True),
        ScaleIntensityRange(a_min=-1000, a_max=500, b_min=0, b_max=255, clip=True),
        CropForeground(source_key="image"),
    ]
)

MODEL_PATH = Path(__file__).parent.joinpath("weights").joinpath("lungs_model.pt")

model_lungs = Ahnet(
    spatial_dims=3, psp_block_num=0, upsample_mode="trilinear", out_channels=2
)
model_lungs.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))


def get_interpolated_masks_array(
    ct_scan: np.ndarray, prediction_output: np.ndarray
) -> np.ndarray:
    original_image = np.expand_dims(ct_scan, 0)
    original_image = back_to_original_size_transforms(original_image[0])[0]

    masks = []

    for idx in range(original_image.shape[-1]):
        # mask interpolation
        masks_index_fr = (idx / (original_image.shape[-1] - 1)) * (
            output_lungs.shape[-1] - 1
        )
        mi_a, mi_b = int(math.floor(masks_index_fr)), int(math.ceil(masks_index_fr))
        d = masks_index_fr - mi_a

        if not output_lungs.dtype == np.uint8:
            output_lungs = np.array(output_lungs, dtype=np.uint8)

        output_a = prediction_output[0, ..., mi_a]
        output_b = prediction_output[0, ..., mi_b]
        output_ = output_a.astype(float) * (1.0 - d) + output_b.astype(float) * d
        output_ = np.round(cv2.resize(output_, original_image.shape[1:3]))

        masks.append(output_)

    masks = np.stack(masks, axis=0)
    masks = np.array(masks, dtype=np.uint8)

    return masks


def get_lungs_masks(ct_scan: np.ndarray):
    """

    :param ct_scan:
    :return:
    """
#    print(f"Input shape: {ct_scan.shape}")
#    print(f"First transform shape: {tranform_one(ct_scan).shape}")
#    print(f"Second transform shape: {tranform_two(ct_scan).shape}")
#    print(f"Final transform shape: {transforms(ct_scan).shape}")

    input_ = transforms(ct_scan)[0].unsqueeze(0)

    output_lungs = model_lungs(input_)
    output_lungs = output_lungs.argmax(dim=1).detach().cpu().numpy()

    lung_masks = get_interpolated_masks_array(ct_scan, output_lungs)

    return lung_masks
