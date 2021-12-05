import math
import numpy as np
import cv2


def get_interpolated_and_resized_masks(
    prediction_output: np.ndarray,
    expected_width: int,
    expected_height: int,
    expected_number_of_frames: int,
) -> np.ndarray:
    masks = []

    for idx in range(expected_number_of_frames):
        # mask interpolation
        masks_index_fr = (idx / (expected_number_of_frames - 1)) * (
            prediction_output.shape[-1] - 1
        )
        mi_a, mi_b = int(math.floor(masks_index_fr)), int(math.ceil(masks_index_fr))
        d = masks_index_fr - mi_a

        if not prediction_output.dtype == np.uint8:
            prediction_output = np.array(prediction_output, dtype=np.uint8)

        output_a = prediction_output[0, ..., mi_a]
        output_b = prediction_output[0, ..., mi_b]
        output_ = output_a.astype(float) * (1.0 - d) + output_b.astype(float) * d
        output_ = np.round(cv2.resize(output_, (expected_width, expected_height)))

        masks.append(output_)

    masks = np.stack(masks, axis=0)
    masks = np.array(masks, dtype=np.uint8)

    return masks
