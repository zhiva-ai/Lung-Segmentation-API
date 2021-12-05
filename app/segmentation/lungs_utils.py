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

NUM_SLICES = 32

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

back_to_original_size_transforms = Compose([
        AddChannel(),
        Spacing(pixdim=[1., 1., 1.], mode="bilinear", align_corners=True),
        ScaleIntensityRange(a_min=-1000, a_max=500, b_min=0, b_max=255, clip=True),
        CropForeground(source_key="image"),
    ])

MODEL_PATH = Path(__file__).parent.joinpath("weights").joinpath("lungs_model.pt")

model_lungs = Ahnet(
    spatial_dims=3, psp_block_num=0, upsample_mode="trilinear", out_channels=2
)
model_lungs.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))