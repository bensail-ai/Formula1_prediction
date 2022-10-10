import pandas as pd
import numpy as np
import json
import os
import weather_funcs
from datetime import datetime
import re
import sys
df_data=pd.read_csv('../garmin_data/raw_data/combined_data.csv',index_col=[0])

def gettimestamp(string):
    temp = re.split('-| |:',string.split('+')[0])
    datetime_tmp=[int(item) for item in temp]
    Y,M,D,H,m,S=datetime_tmp
    timestamp=datetime(Y,M,D,H,m,S).timestamp()
    return timestamp

posaildict={range(0,60):'upwind',
range(60,110): 'reaching',
 range(110,180): 'downwind'}
newdict={}
for k,v in posaildict.items():
    #print(k)
    for item in k:
        #print(item)
        newdict[int(item)] = v

dfgpx=df_data.copy()
dfgpx['date'] = dfgpx.time.str.split(' ').str[0]
dfgpx['timestamp'] = dfgpx['time'].apply(lambda x: gettimestamp(x))
dataset_lat_long = dfgpx.groupby(['dataset_id'])['latitude','longitude'].mean()
weathers=[]
dfgpx['wdir']=0
dfgpx['TWA']=0
dfgpx['TWA_int']=0
dfgpx['possail']=' '
for dataset in dataset_lat_long.index:
    print(dataset)
    lat = round(dataset_lat_long[dataset_lat_long.index ==dataset].values[0][0],6)
    long = round(dataset_lat_long[dataset_lat_long.index ==dataset].values[0][1],6)
    location=str(lat)+','+str(long)
    Startdate = dfgpx[(dfgpx['dataset_id']==dataset)].date.iloc[0]
    Enddate = dfgpx[(dfgpx['dataset_id']==dataset)].date.iloc[-1]
    print(location,' ', Startdate, ' ', Enddate)
    try:
        temp =weather_funcs.get_weather_data(location,Startdate,Enddate)
        weathers.append(temp)
        dfgpx.loc[dfgpx['dataset_id']==dataset,'wdir']=dfgpx.loc[dfgpx['dataset_id']==dataset,'timestamp'].apply(lambda x: temp['windir'][(temp['timestamp'] - x).abs().argsort()[0]])
        dfgpx.loc[dfgpx['dataset_id']==dataset,'TWA']=abs(dfgpx.loc[dfgpx['dataset_id']==dataset,'course']-dfgpx.loc[dfgpx['dataset_id']==dataset,'wdir']).apply(lambda x: 360-x if x > 180 else x)
      

    except KeyboardInterrupt:
        print('Interrupted')
        sys.exit(0)
       
dfweather=pd.concat(weathers).reset_index(drop=True)
dfweather.to_csv('../garmin_data/weather_data/weather_data.csv')

dfgpx.to_csv('../garmin_data/auto_labelled/gps_data_TWA.csv')

try:
    dfgpx['TWA_int']=round(dfgpx['TWA'].fillna(0.0),0).astype('int')
    dfgpx['possail']=dfgpx['TWA_int'].map(newdict)
    dfgpx.to_csv('../garmin_data/auto_labelled/gps_data_TWA_labelled_1.csvcsv')
except:
    pass