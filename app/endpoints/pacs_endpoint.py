from fastapi import APIRouter
from app.segmentation.lungs_segmentation_inference import get_lungs_masks
from app.endpoints.utils import convert_single_class_mask_to_response_json
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel
from app.docker_logs import get_logger
from time import time
from typing import List
import numpy as np
import io
import pydicom
import json



router = APIRouter(
    prefix="/pacs-endpoint",
)

logger = get_logger("pacs-endpoint-logger")


class Item(BaseModel):
    instances: List[str]

def get_instance_number(instance):
    return instance.InstanceNumber

@router.post("/predict", response_class=ORJSONResponse)
async def predict(
    item: Item
):
    """
    Lung segmentation endpoint. Takes the PACS server address, study and series UIDs as input,
    retrieves the DICOM image and provides it as input for the lung segmentation model. The mask predictions in the
    json is returned
    :param item:
    :return: json in the specified format
    """

    download_start = time()
    instances = [
        pydicom.dcmread(io.BytesIO(bytes(json.loads(instance))))
        for instance in item.instances
    ]
    instances.sort(key=get_instance_number)
    logger.info(f"{len(instances)} instances in series")
    download_end = time()
    logger.info(f"Parse duration: {download_end - download_start} s.")

    inference_start = time()

    mapping_dict = {str(i.InstanceNumber): str(i.SOPInstanceUID) for i in instances}

    study_instance_uid = str(instances[0].StudyInstanceUID)
    series_instance_uid = str(instances[0].SeriesInstanceUID)

    # (width, height, frames)
    series_array = np.stack([i.pixel_array for i in instances], axis=-1)
    # (frames, width, height)
    masks = get_lungs_masks(series_array)

    inference_end = time()
    logger.info(f"Inference duration: {inference_end-inference_start} s.")

    # return {"Status": "success"}
    return convert_single_class_mask_to_response_json(
        study_instance_uid,
        series_instance_uid,
        mapping_dict,
        masks,
        "Lungs",
        "Lungs",
    )
