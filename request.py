import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url, json = {'Country' : 'South Korea', 'Education' : 'Bachelor’s degree', 'Experience' : 4, 'Age' : '25-34 years old' })

print(r.json())