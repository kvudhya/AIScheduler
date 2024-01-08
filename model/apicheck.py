import requests


response = requests.post(url="http://127.0.0.1:8000/prediction",json={"event": "Math homework"})

print(response)