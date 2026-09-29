import requests

with open(
    "/home/mahmoud/projects/RegionAwareVisionAssistant/data/MVTecAD/wood/test/color/001.png",
    "rb",
) as f:
    response = requests.post(
        "http://127.0.0.1:8000/predict/?category=wood",
        files={"file": f},
    )

print(response.json().keys())
print(response.json()["regions"])
