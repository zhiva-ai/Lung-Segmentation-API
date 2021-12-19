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

    final_dict = recursive_dict(final_dict)

    return orjson.dumps(final_dict)


def recursive_dict(d):
    for k, v in d.items():
        if isinstance(v, dict):
            recursive_dict(v)
        else:
            d = replace_str(d)
    return d

def replace_str(final_dict):
    for key in final_dict.keys():
        if type(key) is not str:
            try:
                final_dict[str(key)] = final_dict[key]
            except:
                try:
                    final_dict[repr(key)] = final_dict[key]
                except:
                    pass
            del final_dict[key]

    return final_dict
