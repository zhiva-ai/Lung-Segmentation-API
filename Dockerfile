FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONUNBUFFERED=1

RUN apt-get update ##[edited]
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY ./app ./app

#Lungs weights .pt file
RUN apt update
RUN apt install wget
RUN wget -O ./app/segmentation/weights/lungs_model.pt https://api.ngc.nvidia.com/v2/models/nvidia/med/clara_pt_covid19_ct_lung_segmentation/versions/1/files/models/model.pt

EXPOSE 80
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
