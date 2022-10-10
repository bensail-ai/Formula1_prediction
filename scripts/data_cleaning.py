import requests
import pandas as pd

def get_elevation(lat, long):
    query = ('https://api.open-elevation.com/api/v1/lookup'
             f'?locations={lat},{long}')
    r = requests.get(query).json()  # json object
    elevation = r['results'][0]['elevation']

    return  elevation


