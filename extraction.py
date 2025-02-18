import requests
import json

# API endpoint
url = 'https://admin.apilproperties.com/api/properties'

# Send a GET request to the API
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    data = response.json()  # Convert the response to a Python dictionary
    
    # Save the data to a JSON file
    with open('properties.json', 'w') as json_file:
        json.dump(data, json_file)
    
    print("Data saved successfully!")
else:
    print("Failed to fetch data:", response.status_code)
