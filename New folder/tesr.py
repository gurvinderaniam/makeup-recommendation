import requests

url = "http://127.0.0.1:5000/predict"
data = {
    "Skin_Tone_R": 120,
    "Skin_Tone_G": 85,
    "Skin_Tone_B": 60,
    "Face_Shape": "Oval"
}

response = requests.post(url, json=data)
print("Status Code:", response.status_code)
print("Response:", response.json())
