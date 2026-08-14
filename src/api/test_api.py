import requests
files = {"file": open("/home/mahmoud/projects/RegionAwareVisionAssistant/data/MVTecAD/wood/test/color/001.png", "rb")}
response = requests.post("http://127.0.0.1:8000/predict/?category=wood", files=files)
print(response.json().keys())
print(response.json()["regions"])
