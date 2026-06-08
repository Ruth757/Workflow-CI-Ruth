import requests
import json

url = "http://127.0.0.1:5001/invocations"

data = {
    "inputs": [
        [1, 35, 1, 100, 5000]
    ]
}

headers = {
    "Content-Type": "application/json"
}

response = requests.post(
    url,
    data=json.dumps(data),
    headers=headers
)

print(response.status_code)
print(response.text)