from fastapi import APIRouter
from dicomweb_client.api import DICOMwebClient
from app.segmentation.lungs_segmentation_inference import get_lungs_masks
from app.endpoints.utils import convert_single_class_mask_to_response_json
import numpy as np
from pydantic import BaseModel
from app.docker_logs import get_logger
from time import time
from fastapi.responses import ORJSONResponse


class PACSStudy(BaseModel):
    server_address: str
    study_instance_uid: str
    series_instance_uid: str


router = APIRouter(
    prefix="/pacs-endpoint",
)

logger = get_logger("pacs-endpoint-logger")


@router.get("/predict", response_class=ORJSONResponse)
async def predict(
    server_address: str, study_instance_uid: str, series_instance_uid: str
):
    """
    Lung segmentation endpoint. Takes the PACS server address, study and series UIDs as input,
    retrieves the DICOM image and provides it as input for the lung segmentation model. The mask predictions in the
    json is returned
    :param server_address:
    :param study_instance_uid:
    :param series_instance_uid:
    :return: json in the specified format
    """
    download_start = time()
    client = DICOMwebClient(url=server_address)
    instances = client.retrieve_series(
        study_instance_uid=study_instance_uid,
        series_instance_uid=series_instance_uid,
    )
    logger.info(f"{len(instances)} instances in series")
    download_end = time()
    logger.info(f"Download duration: {download_end - download_start} s.")

    inference_start = time()
    # (width, height, frames)
    series_array = np.stack([i.pixel_array for i in instances], axis=-1)

    mapping_dict = {str(i.InstanceNumber): i.SOPInstanceUID for i in instances}

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
