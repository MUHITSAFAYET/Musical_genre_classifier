import requests
import json

resp = requests.post("http://localhost:5000", files={'file': open('test/hiphop.0000.wav', 'rb')})

print(resp.json())