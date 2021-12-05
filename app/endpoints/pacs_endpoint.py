from fastapi import APIRouter
from dicomweb_client.api import DICOMwebClient
from app.segmentation.lungs_inference import get_lungs_masks
from app.endpoints.utils import convert_single_class_mask_to_response_json
import numpy as np
from pydantic import BaseModel
from app.docker_logs import get_logger


class PACSStudy(BaseModel):
    server_address: str
    study_instance_uid: str
    series_instance_uid: str


router = APIRouter(
    prefix="/pacs-endpoint",
)

logger = get_logger("pacs-endpoint-logger")


@router.post("/predict")
async def predict(pacs_study: PACSStudy):
    client = DICOMwebClient(url=pacs_study.server_address)
    instances = client.retrieve_series(
        study_instance_uid=pacs_study.study_instance_uid,
        series_instance_uid=pacs_study.series_instance_uid,
    )
    logger.info(f"{len(instances)} instances in series")

    # (width, height, frames)
    series_array = np.stack([i.pixel_array for i in instances], axis=-1)

    mapping_dict = {str(i.InstanceNumber): i.SOPInstanceUID for i in instances}

    masks = get_lungs_masks(series_array)

    return {"Status": "success"}

    # return convert_single_class_mask_to_response_json(
    #     pacs_study.study_instance_uid,
    #     pacs_study.series_instance_uid,
    #     mapping_dict,
    #     masks.transpose(2, 0, 1),
    #     "Lungs",
    #     "Lungs",
    # )
