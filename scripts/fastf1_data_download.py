#%%
import fastf1
import pandas as pd
import time
import os
#%%
def f1_telemetry_data(year,round,session=['qualifying']):
    if os.path.exists('./f1_cache'):
        pass
    else:
        os.mkdir('./f1_cache')
    fastf1.Cache.enable_cache('./f1_cache')

    event = fastf1.get_event(year,round)
    #qualifying and practice3 is when the drivers go flat out for the maximum performance in 1 lap
    qualifying=event.get_qualifying()
    practice3=event.get_practice(3)
    event_dict={}
    for ses in session:
        if ses =='qualifying':
            event_dict[ses]=qualifying
        elif ses == 'practice3':
            event_dict[ses]=practice3
    #event_dict={'qualifying':qualifying, 'practice3':practice3 }
    #session = fastf1.get_session(2018, 'Australia Grand Prix', 'Q')


    data_dict={}
    for session in event_dict:
        print(event_dict[session],'session loading')
        event_dict[session].load(laps=True,telemetry=True,weather=True)
        time.sleep(0.5)
        print(f'{session}_loaded')
        laps = event_dict[session].laps.pick_wo_box()# pick the laps not in or outlaps as not representative

        laps.reset_index(drop=True, inplace=True)
        laps.reset_index(names=f'lapId',inplace=True)
        weather_data = laps.get_weather_data()
        weather_data.reset_index(drop=True,inplace=True)
        #weather data has a data point per lap so can just join and drop weather data time column
        laps_weather = pd.concat([laps, weather_data.loc[:, ~(weather_data.columns == 'Time')]], axis=1)
        frames=[]                   
        for i, lap in enumerate(laps_weather.index):
            print(i)
            try:
                df= laps_weather.iloc[i].get_telemetry()
                df['lapId'] = i
                frames.append(df)
                print('got telemetry data')
            except:
                try:
                    df= laps_weather.iloc[i].get_car_data()
                    df['lapId'] = i
                    df[f'{session}_DistanceToDriverAhead']=' '
                    df[f'{session}__DriverAhead']=' '
                    df[f'{session}__Distance']=' '
                    df[f'{session}__RelativeDistance']=' '
                    df[f'{session}__Status']=' '
                    df[f'{session}__X']=' '
                    df[f'{session}__Y']=' '
                    df[f'{session}__Z']=' '
                    frames.append(df)
                    print('got car data')
                except:
                    print('no car or telemetry data for that lap')
                    pass
        if len(frames) >1:    
            df_tele= pd.concat(frames)
            laps_combined = laps_weather.merge(df_tele,on='lapId',how='left')
        else:
            laps_combined = laps_weather.copy()
      
        laps_combined= laps_combined.add_prefix(f'{session}_')
        laps_combined[f'{session}_DriverNumber']=laps_combined[f'{session}_DriverNumber'].astype('float')
        laps_combined.drop(columns=[f'{session}_PitOutTime',f'{session}_PitInTime'],inplace=True)
        data_dict[session]=laps_combined

    #combined_data = events[0].merge(events[1],left_on='qualifying_DriverNumber',right_on='practice3_DriverNumber',how='left')

    return data_dict

# %%

def combine_telemetry(event_data,ergast_df,session='qualifying',year=2018):
    frames=[]
    for event in event_data:
        ergast_df_event = ergast_df[(ergast_df['year']==year) & (ergast_df['name']==event)]
        for ses in event_data[event]:
            if ses == session:
                combined = ergast_df_event.merge(event_data[event][ses],left_on='number',right_on='qualifying_DriverNumber',how='left')
                frames.append(combined)
    data=pd.concat(frames)
    data.reset_index(drop=True,inplace=True)
    return data