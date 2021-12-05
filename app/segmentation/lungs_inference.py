import numpy as np
from app.segmentation.lungs_utils import transforms, back_to_original_size_transforms, model_lungs
from app.segmentation.utils import get_interpolated_masks_array


def get_lungs_masks(ct_scan: np.ndarray):
    """

    :param ct_scan:
    :return:
    """
    input_ = transforms(ct_scan)[0].unsqueeze(0)

    output_lungs = model_lungs(input_)
    output_lungs = output_lungs.argmax(dim=1).detach().cpu().numpy()

    lung_masks = get_interpolated_masks_array(ct_scan, output_lungs)

    return lung_masks
