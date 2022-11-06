#%%
import json
import pandas as pd
import numpy as np
from datetime import datetime
from weather_funcs import *
from bs4 import BeautifulSoup
import selenium
import requests


#%%
races=pd.read_csv("../data/raw/races.csv") # read files races
circuits=pd.read_csv("../data/raw/circuits.csv")# read file


# %%
races_weather = races.merge(circuits[['circuitId','lat','lng']],left_on='circuitId',right_on='circuitId').copy() #combine
races_wc = races_weather[['raceId', 'year', 'circuitId', 'name', 'date', 'time', 'url','quali_date', 'quali_time', 'sprint_date', 'sprint_time', 'lat', 'lng']] #select useful columns
races_wc = races_wc[races_wc['date'] < str(pd.to_datetime('today').normalize())] # select races less than today
races_wc.replace("\\N",np.NaN,inplace=True) # clean the \N
# %%
#race weather
races_wc[['race_temp','race_precip','race_humidity','race_condition']]=races_wc.apply(lambda x: get_weather_data(x.lat,x.lng,x.date,x.time),axis=1) # get weather data from api for races 


#%%
races_wc.to_csv('../data/raw/races_weather_api.csv')
#%%
races_wc[['quali_temp','quali_precip','quali_humidity','quali_condition']]=races_wc.apply(lambda x: get_weather_data_quali(x.lat,x.lng,x.quali_date,x.quali_time,x.date,x.time),axis=1) #get weather data from api for qualifying

#%%
races_wc.to_csv('../data/raw/races_weather_api_quali.csv')

#%%


races_wc['race_weather_wiki']=races_wc.apply(lambda x: get_url_weather(x.url),axis=1) # scrap the wikipedia websites for weather data
# %%
races_wc.to_csv('../data/raw/races_weather_all.csv')
#races_wc.to_csv('../data/raw/races_weather_wiki.csv')
# %%
