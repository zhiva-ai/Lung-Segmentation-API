[![](https://images.microbadger.com/badges/license/nbrown/revealjs.svg)](LICENSE)
# Lung segmentation API
API serving the lungs predictions for chest CT DICOM images. 

We use the Nvidia Clara lungs 3D semanticsegmentation model, avalible [here](https://ngc.nvidia.com/catalog/models/nvidia:med:clara_pt_covid19_ct_lung_segmentation).


API takes a CT scan, samples 32 samples from it and conducts 3D semantic segmentation. 
After that ot interpolate the predictions between the samples to achieve a segmentation 
mask for each frame and this is returned as a `.json` file.  

<img src="assets/lung-segmentation-visualisation.webp" width="700px"/>

# How to run the API 

```
docker-compose up
```

API avalible at:

```
0.0.0.0:8011
```

# FAQ
- I'm getting an error on MacBook `docker rpc error code = unknown desc = executor failed running [...]`.

Your docker settings are limiting the size of the image and cannot install all the `requirements.txt`. Go to `Preferences > Resources > Advanced` in your `Docker Desktop` application and increase the memory limit.

# How to test it? 
Follow the official [tutorial](https://docs.zhiva.ai/latest). You need to configure a proxy server that will route requests 
from the [viewer](https://alpha.zhiva.ai/login) to the model API. 

# Authors
- [Piotr Mazurek](https://github.com/tugot17)
