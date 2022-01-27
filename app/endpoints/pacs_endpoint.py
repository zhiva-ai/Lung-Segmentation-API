from fastapi import APIRouter
from app.segmentation.lung_segmentation.lungs_segmentation_inference import get_lungs_masks, lung_segmentation_inference
from app.endpoints.utils import (
    convert_single_class_mask_to_json_response,
)
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel
from app.docker_logs import get_logger
from time import time
from typing import List
import io
import pydicom
import json

router = APIRouter(
    prefix="/pacs-endpoint",
)

logger = get_logger("pacs-endpoint-logger")


class Item(BaseModel):
    instances: List[str]


@router.post("/predict", response_class=ORJSONResponse)
async def predict(item: Item):
    """
    Lung lung_segmentation endpoint. Takes the PACS server address, study and series UIDs as input,
    retrieves the DICOM image and provides it as input for the lung lung_segmentation model. The mask predictions in the
    json is returned
    :param item:
    :return: json in the specified format
    """
    instances = [
        pydicom.dcmread(io.BytesIO(bytes(json.loads(instance))))
        for instance in item.instances
    ]
    logger.info(f"{len(instances)} instances in series")

    study_instance_uid = str(instances[0].StudyInstanceUID)
    series_instance_uid = str(instances[0].SeriesInstanceUID)
    mapping_dict = {str(i.InstanceNumber): str(i.SOPInstanceUID) for i in instances}

    inference_start = time()
    lung_masks, lung_segmentation_metadata = lung_segmentation_inference(instances)
    inference_end = time()

    logger.info(f"Inference duration: {inference_end - inference_start} s.")

    # return {"Status": "success"}
    return convert_single_class_mask_to_json_response(
        study_instance_uid,
        series_instance_uid,
        mapping_dict,
        lung_masks,
        lung_segmentation_metadata,
        "Lungs",
        "Lungs",
    )
