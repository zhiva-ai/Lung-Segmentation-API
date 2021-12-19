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
import torch.nn.functional as F

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
lungs_model.eval()


@torch.no_grad()
def get_lungs_masks(ct_scan: np.ndarray) -> np.ndarray:
    """
    Given an np.array of frames from a CT scan, the method returns the lung segmentation for each frame.
    Nvidia clara model available here: https://ngc.nvidia.com/catalog/models/nvidia:med:clara_pt_covid19_ct_lung_segmentation
    is used for inference. The 32 frames are sampled from the input DICOM and used as model input.
    The masks for the remaining frames are then interpolated.
    :param ct_scan: # (width, height, frames) array from a DICOM image, the pixel values ranges from 0 up to 4096
    :return: (frames, width, height) array of 0s and 1s with the segmented lungs
    """
    width, height, number_of_frames = ct_scan.shape

    input_ = lungs_transforms(ct_scan)[0].unsqueeze(0)

    output_lungs = lungs_model(input_)
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

    return lung_masks
