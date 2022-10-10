#%%
import json
import pandas as pd

import matplotlib.pyplot as plt
import re
from datetime import datetime
import numpy as np
#%%
with open('test.json') as file:
    data = json.load(file)

race_time="14:00:00"

for day in data['days']:
    day_date = day['datetime']
    time= []
    temp =[]
    preciptype =[]
    precip=[]
    humidity=[]
    conditions=[]
    timestamp=[]
    for hour in day['hours']:
        time.append(hour['datetime'])
        timestamp.append(hour['datetimeEpoch'])
        temp.append(hour['temp'])
        precip.append(hour['precip'])
        preciptype.append(hour['preciptype'])
        humidity.append(hour['humidity'])
        conditions.append(hour['conditions'])
# %%
dfweather=pd.DataFrame({'time':time,'temp':temp,
'precip':precip, 'preciptype': preciptype,
'humidity':humidity,'conditions':conditions,
'timestamp':timestamp})
dfweather['day']=day_date

dfweather['datetime']=dfweather['day']+'-'+dfweather['time']
# %%

dfweather


data=pd.read_csv('combined_data.csv',index_col=[0])

#%%

dfgpx = data.copy()
dfgpx['date'] = dfgpx.time.str.split(' ').str[0]
dftest = dfgpx[dfgpx['date'] == '2022-08-19'].copy()
# %%
#todo
#timestamp gpx
def gettimestamp(string):
    temp = re.split('-| |:',string.split('+')[0])
    datetime_tmp=[int(item) for item in temp]
    Y,M,D,H,m,S=datetime_tmp
    timestamp=datetime(Y,M,D,H,m,S).timestamp()
    return timestamp

def gettimestamp2(string):
    temp = re.split('-| |:',string)
    datetime_tmp=[int(item) for item in temp]
    Y,M,D,H,m,S=datetime_tmp
    timestamp=datetime(Y,M,D,H,m,S).timestamp()
    return timestamp

dftest['timestamp'] = dftest['time'].apply(lambda x: gettimestamp(x))

#timestamp weather
#dfweather['timestamp'] = dfweather['datetime'].apply(lambda x: gettimestamp2(x))

#assign wind direction to time in gpx
dftest['wdir']=dftest['timestamp'].apply(lambda x: dfweather['windir'][(dfweather['timestamp'] - x).abs().argsort()[0]])

# find twa and assign angles
dftest['TWA']=abs(dftest['course']-dftest['wdir']).apply(lambda x: 360-x if x > 180 else x)
#%%
posaildict={range(0,60):'upwind',
range(60,110): 'reaching',
 range(110,180): 'downwind'}
newdict={}
for k,v in posaildict.items():
    #print(k)
    for item in k:
        #print(item)
        newdict[int(item)] = v

dftest['TWA_int']=dftest['TWA']
dftest = dftest[dftest['course'].notna()]
dftest = dftest.astype({'TWA_int':'int'})
dftest['posail']=dftest['TWA_int'].map(newdict)