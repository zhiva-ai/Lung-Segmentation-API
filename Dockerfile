FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONUNBUFFERED=1

RUN apt-get update ##[edited]
RUN apt-get install ffmpeg libsm6 libxext6 iputils-ping  -y

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY ./app ./app

#Lungs weights .pt file
RUN tar xf ./app/segmentation/lung_segmentation/weights/lungs_model.tar.xz -C ./app/segmentation/lung_segmentation/weights

EXPOSE 8011
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8011"]
