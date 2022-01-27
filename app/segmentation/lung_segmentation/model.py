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
from pydicom import FileDataset
from typing import Tuple

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
