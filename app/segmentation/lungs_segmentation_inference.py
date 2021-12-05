import numpy as np
from pathlib import Path
from monai.networks.nets.ahnet import Ahnet
import torch
from monai.transforms import (
    Compose,
    AddChannel,
    Resize,
    Spacing,
    ScaleIntensityRange,
    CropForeground,
    ToTensor,
)

from app.segmentation.utils import get_interpolated_and_resized_masks

NUM_SLICES = 32

lungs_transforms = Compose(
    [
        AddChannel(),
        Spacing(pixdim=[0.8, 0.8, 5], mode="bilinear", align_corners=True),
        ScaleIntensityRange(a_min=-1000, a_max=500, b_min=0, b_max=1),
        CropForeground(source_key="image"),
        Resize((224, 224, NUM_SLICES)),
        ToTensor(),
    ]
)

LUNGS_MODEL_PATH = Path(__file__).parent.joinpath("weights").joinpath("lungs_model.pt")

lungs_model = Ahnet(
    spatial_dims=3, psp_block_num=0, upsample_mode="trilinear", out_channels=2
)
lungs_model.load_state_dict(torch.load(LUNGS_MODEL_PATH, map_location="cpu"))


def get_lungs_masks(ct_scan: np.ndarray):
    """

    :param ct_scan:
    :return:
    """

    input_ = lungs_transforms(ct_scan)[0].unsqueeze(0)

    output_lungs = lungs_model(input_)
    output_lungs = output_lungs.argmax(dim=1).detach().cpu().numpy()

    width, height, number_of_frames = ct_scan.shape
    lung_masks = get_interpolated_and_resized_masks(
        output_lungs, width, height, number_of_frames
    )

    return lung_masks
