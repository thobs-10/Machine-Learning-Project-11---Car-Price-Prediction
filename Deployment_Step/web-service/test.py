import requests

car = {
    "Mileage": [100000],
    "name": ['Honda'],
    "company": ['Honda Amaze 1.5 SX i DTEC'],
    "year": [2010]
}

url = 'http://localhost:9696/predict'
response = requests.post(url, json=car)
print(response.json())