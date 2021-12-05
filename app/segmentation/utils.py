import math
import numpy as np
import cv2
from app.segmentation.lungs_utils import back_to_original_size_transforms


def get_interpolated_masks_array(
        ct_scan: np.ndarray, prediction_output: np.ndarray
) -> np.ndarray:
    original_image = np.expand_dims(ct_scan, 0)
    original_image = back_to_original_size_transforms(original_image[0])[0]

    masks = []

    print(f"Original image: ", original_image.shape)

    for idx in range(original_image.shape[-1]):
        # mask interpolation
        masks_index_fr = (idx / (original_image.shape[-1] - 1)) * (
                prediction_output.shape[-1] - 1
        )
        mi_a, mi_b = int(math.floor(masks_index_fr)), int(math.ceil(masks_index_fr))
        d = masks_index_fr - mi_a

        if not prediction_output.dtype == np.uint8:
            prediction_output = np.array(prediction_output, dtype=np.uint8)

        output_a = prediction_output[0, ..., mi_a]
        output_b = prediction_output[0, ..., mi_b]
        output_ = output_a.astype(float) * (1.0 - d) + output_b.astype(float) * d
        output_ = np.round(cv2.resize(output_, original_image.shape[1:3]))

        masks.append(output_)

    masks = np.stack(masks, axis=0)
    masks = np.array(masks, dtype=np.uint8)

    return masks
