import json
import requests
import time

url = "0.0.0.0:8011/pacs-endpoint/predict"

server_address = "https://server.dcmjs.org/dcm4chee-arc/aets/DCM4CHEE/rs"
study_instance_uid = "1.3.6.1.4.1.25403.345050719074.3824.20170125095438.5"
series_instance_uid = "1.3.6.1.4.1.25403.345050719074.3824.20170125095449.8"

params = {
    "server_address": server_address,
    "study_instance_uid": study_instance_uid,
    "series_instance_uid": series_instance_uid,
}

if __name__ == '__main__':
    start = time.time()

    result = requests.get(url, params=params)
    params = result.json()
    with open("data.json", "w") as f:
        json.dump(params, f)

    end = time.time()
    print(f"Duration: {end-start} s.")
    # print(params)
