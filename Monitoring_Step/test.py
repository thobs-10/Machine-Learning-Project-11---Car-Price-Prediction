import requests

url = 'http://127.0.0.1:9696/predict'

ride = {
    #'lpep_pickup_datetime': '2021-01-01 00:15:56',
    'Mileage': 100000,
    'name':"Honda",
    'company': "Honda Amaze 1.5 SX i DTEC",
    'year': 2010
}

response = requests.post(url, json=ride).json()
print(response)
