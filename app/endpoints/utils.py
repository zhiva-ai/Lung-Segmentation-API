import numpy as np
from typing import Dict, Any
from app.docker_logs import get_logger
import orjson

logger = get_logger("serialization-logger")


def convert_lungs_prediction_to_json_response(
    study_instance_uid: str,
    series_instance_uid: str,
    mapping: dict,
    masks: np.ndarray,
    series_metadata: Dict[str, Any],
    class_name="Lung",
    description="Lung",
    color="lightskyblue",
    class_color="lightskyblue",
    active_color="aquamarine",
) -> bytes:
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

    rois_in_series["metadata"] = [series_metadata]

    final_dict = {
        study_instance_uid: {
            series_instance_uid: rois_in_series,
        }
    }
    return orjson.dumps(final_dict)
