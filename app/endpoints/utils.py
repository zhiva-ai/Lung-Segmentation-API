import numpy as np


def convert_single_class_mask_to_response_json(
    study_instance_uid: str,
    series_instance_uid: str,
    mapping: dict,
    masks: np.ndarray,
    class_name="Lung",
    description="Lung",
    color="lightskyblue",
    class_color="lightskyblue",
    active_color="aquamarine",
) -> dict:
    rois_in_series = {}

    print(f"Maks shape in convert: {masks.shape}")

    for i in range(masks.shape[0]):
        single_mask = {
            "points": masks[i].tolist(),
            "color": color,
            "classColor": class_color,
            "activeColor": active_color,
            "className": class_name,
            "description": description,
        }

        segments = [single_mask]

        frame_number = str(i + 1)

        rois_in_series[mapping[frame_number]] = {"segments": segments}

    final_dict = {study_instance_uid: {series_instance_uid: rois_in_series}}

    return final_dict
