from fastapi import APIRouter
from dicomweb_client.api import DICOMwebClient
from app.segmentation.inference import get_lungs_masks
import numpy as np

from app.docker_logs import get_logger

router = APIRouter(
    prefix="/pacs-endpoint",
)

logger = get_logger("pacs-endpoint-logger")


@router.post("/predict")
async def predict(
    server_address: str, study_instance_uid: str, series_instance_uid: str
):
    client = DICOMwebClient(url=server_address)
    instances = client.retrieve_series(
        study_instance_uid=study_instance_uid, series_instance_uid=series_instance_uid
    )
    logger.info(f"{len(instances)} instances in series")

    # (frames, width, height)
    series_array = np.stack([i.pixel_array for i in instances], axis=0)

    masks = get_lungs_masks(series_array)

    return {"message": masks.shape}
