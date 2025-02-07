#%%
import requests

url = "https://localhost:8000"
data = {"key": "value"}

response = requests.get(url, json=data,verify=False)


response.text
#%%
import requests

url = "https://localhost:8000/submit"
data = {"key": "value"}

response = requests.post(url, json=data,verify=False)
response.text