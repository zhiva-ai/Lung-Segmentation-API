import numpy as np
from time import time
from app.docker_logs import get_logger
import orjson

logger = get_logger("serialization-logger")

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
) -> bytes:
    json_start = time()

    rois_in_series = {}

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

    json_end = time()
    logger.info(f"Jsonization duration: {json_end - json_start} s.")

    return orjson.dumps(final_dict)
