#!/usr/bin/env python3
#%%
import fastf1
import pandas as pd
import time
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore') # lots of warnings for fastf1
#%%
from scripts.f1_ultils import *
#from f1_ultils import *
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
        if len(frames) >=1:    
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

def get_fastf1_session(year,round,session='qualifying'):

    if os.path.exists('./f1_cache'):
        pass
    else:
        os.mkdir('./f1_cache')
    fastf1.Cache.enable_cache('./f1_cache')

    event = fastf1.get_event(year,round)
    if session == 'qualifying':
        data = event.get_qualifying()
    elif session == 'practice3':
        data =event.get_practice(3)
    else:
        print('bad input')
    print(data,'session loading')
    data.load(laps=True,telemetry=True,weather=True)
    return data



def session_lap_driver(data,driver,session='qualifying'):
    

    lap_df= data.laps.pick_wo_box().pick_driver(driver)
    lap_df.reset_index(drop=True, inplace=True)
    lap_df.reset_index(names=f'lapId',inplace=True)
    weather_data = lap_df.get_weather_data()
    weather_data.reset_index(drop=True,inplace=True)
    #weather data has a data point per lap so can just join and drop weather data time column
    laps_weather = pd.concat([lap_df, weather_data.loc[:, ~(weather_data.columns == 'Time')]], axis=1)
    frames=[]                   
    for i, lap in enumerate(laps_weather.index):
        #print(i)
        try:
            df= laps_weather.iloc[i].get_telemetry()
            df['lapId'] = i
            frames.append(df)
            #print('got telemetry data')
        except:            
            print('no telemetry data for that lap')
            pass
    if len(frames) >=1:    
        df_tele= pd.concat(frames)
        laps_combined = laps_weather.merge(df_tele,on='lapId',how='right')
        laps_combined= laps_combined.add_prefix(f'{session}_')
        laps_combined[f'{session}_DriverNumber']=laps_combined[f'{session}_DriverNumber'].astype('float')
        laps_combined.drop(columns=[f'{session}_PitOutTime',f'{session}_PitInTime'],inplace=True)
    else:
        laps_combined = pd.DataFrame(columns=laps_weather.index)
    
    
    return laps_combined


def prepare_results_dataframe(file):

    ergast_combined_df=pd.read_csv(file)
    qualify_df= ergast_combined_df[['raceId', 'driverRef', 'number','name','circuitRef', 'country', 'nationality_drivers',
       'constructorRef', 'nationality_constructors', 'year', 'lat_x', 'lng_x',
       'alt', 'quali_position', 'dob','q1','q2','q3']]
    qualify_df= qualify_df[qualify_df['year'] >=2018]
    qualify_df.reset_index(drop=True, inplace=True)
    return qualify_df

def clean_quali_times(df):
    df['q1'].fillna('nan',inplace=True)
    df['q2'].fillna('nan',inplace=True)
    df['q3'].fillna('nan',inplace=True)
    df['q1_milliseconds']=df['q1'].apply(convert_time_miliseconds)
    df['q2_milliseconds']=df['q2'].apply(convert_time_miliseconds)
    df['q3_milliseconds']=df['q3'].apply(convert_time_miliseconds)
    df=clean_q3_times(df,'q1_milliseconds','q2_milliseconds','q3_milliseconds','fastest_lap_milliseconds')
    df['fastest_all_sessions_milliseconds'] = df.apply(lambda x: find_fastest_lap(x['q1_milliseconds'],x['q2_milliseconds'],x['q3_milliseconds']),axis=1)
    df.drop( columns=['q1_milliseconds',
    'q2_milliseconds',
    'q3_milliseconds',
    'q1',
    'q2',
    'q3'
    ], axis=1,inplace=True)
    
    return df

def add_col(df,col,value):

    df[col] = value

    return df


def clean_time(df, columns =['qualifying_end_lap_sessiontime','qualifying_LapTime'
,'qualifying_Sector1Time','qualifying_Sector2Time','qualifying_Sector3Time',
'qualifying_lap_timedelta']):

    df.rename(columns ={'qualifying_Time_x':'qualifying_end_lap_sessiontime','qualifying_Time_y':'qualifying_lap_timedelta'},inplace=True)
    
    
    for col in columns:
        if df[col].dtype == '<m8[ns]':            
            df[col] = df[col].dt.total_seconds()
        else:
            df[col] = df[col].astype('timedelta64')
            df[col] = df[col].dt.total_seconds()

    return df


def clean_laps(lap_df):
    lapIds = lap_df['qualifying_lapId'].unique()
    laps=[]
    for lap in lapIds:
        if lap_df.loc[lap_df['qualifying_lapId']==lap,'qualifying_Distance'].max() < 100:
            laps.append(lap)
        elif lap_df.loc[lap_df['qualifying_lapId']==lap,'qualifying_Distance'].min() <= 0:
            laps.append(lap)
    lap_df=lap_df[~(lap_df['qualifying_lapId'].isin(laps))]
    lap_df= lap_df[lap_df['qualifying_Distance'].notnull()]
    return lap_df



def clean_lap_df(lap_df):
    lap_df['qualifying_lapId'].dropna(inplace=True)
    lap_df = lap_df[lap_df['qualifying_IsAccurate']]
    lap_df.drop(columns=['qualifying_DriverAhead',
    'qualifying_DistanceToDriverAhead',
    'qualifying_Sector1SessionTime',
    'qualifying_Sector2SessionTime',
    'qualifying_Sector3SessionTime',
    'qualifying_LapStartTime',
    'qualifying_SessionTime'], axis=1, inplace=True)     
    lap_df = fill_mean_parameter_fastf1(lap_df,'qualifying_SpeedI1')
    lap_df = fill_mean_parameter_fastf1(lap_df,'qualifying_SpeedI2')
    lap_df = fill_mean_parameter_fastf1(lap_df,'qualifying_SpeedFL')
    lap_df = fill_mean_parameter_fastf1(lap_df,'qualifying_SpeedST')
    lap_df=lap_df[lap_df.index.isin(lap_df['qualifying_Compound'].dropna(axis=0).index)]
    lap_df=clean_time(lap_df)
    lap_df=clean_laps(lap_df)
    
    return lap_df

def pick_fastest_lap(lap_df):

    fast_lap = lap_df['qualifying_LapTime'].min()*1000

    return fast_lap

def replace_fast_nan(lap_df):
    
    lap_df.loc[lap_df['fastest_all_sessions_milliseconds'].isna(),'fastest_all_sessions_milliseconds'] = pick_fastest_lap(lap_df)

    return lap_df

def laps_corners(lap_df):
    lapIds = lap_df['qualifying_lapId'].unique()
    for lap in lapIds:
        
        lap_df.loc[lap_df['qualifying_lapId']==lap,'corner']= (flag_corners(lap_df[lap_df['qualifying_lapId']==lap].copy(),smoothing=10))['corner']
    return lap_df


def circuit_length(lap_df):
    lapIds = lap_df['qualifying_lapId'].unique()
    lengths=[]
    for lap in lapIds:
        lap_min = (lap_df.loc[(lap_df['qualifying_lapId'] ==lap),'qualifying_Distance'].min())
        lap_max = (lap_df.loc[(lap_df['qualifying_lapId'] ==lap),'qualifying_Distance'].max())
        lengths.append(lap_max-lap_min)
    return int(np.mean(lengths))


def circuit_straight(lap_df):
    lapIds = lap_df['qualifying_lapId'].unique()
    total_=[]
    mean_=[]
    max_=[]
    for lap in lapIds:
        lap_query=(lap_df['qualifying_lapId'] == lap)
        if lap_df.loc[lap_query,'qualifying_TrackStatus'].all() == 1:
            total_straight, mean_straight, max_straight = straight_lengths(lap_df[lap_query].copy())
            total_.append(total_straight)
            mean_.append(mean_straight)
            max_.append(max_straight)
    
    return int(np.mean(total_)), int(np.mean(mean_)), int(np.mean(max_))
    

def circuit_corner(lap_df):
    lapIds = lap_df['qualifying_lapId'].unique()
    total_=[]
    mean_=[]
    max_=[]
    total_curv_=[]
    mean_curv_=[]
    max_curv_=[]
    std_curv_=[]
    num_=[]
    for lap in lapIds:
        lap_query=(lap_df['qualifying_lapId'] == lap)
        if lap_df.loc[lap_query,'qualifying_TrackStatus'].all() == 1:
            total_straight, mean_straight, max_straight,total_curv,mean_curv,max_curv,std_curv,num_corners = corners(lap_df[lap_query].copy())
            total_.append(total_straight)
            mean_.append(mean_straight)
            max_.append(max_straight)
            total_curv_.append(total_curv)
            mean_curv_.append(mean_curv)
            max_curv_.append(max_curv)
            num_.append(num_corners)
            std_curv_.append(std_curv)

    return int(np.mean(total_)), int(np.mean(mean_)), int(np.mean(max_)), (np.mean(total_curv_)), (np.mean(mean_curv_)), (np.mean(max_curv_)), (np.mean(std_curv_)),int(np.mean(num_))


def circuit_aggregations(lap_df_aggr,lap_df):
    lap_df_aggr['circuit_length'] = circuit_length(lap_df)
    circuit_total_straight, circuit_mean_straight, circuit_max_straight= circuit_straight(lap_df)
    lap_df_aggr['circuit_total_straight'] = circuit_total_straight
    lap_df_aggr['circuit_mean_straight'] = circuit_mean_straight
    lap_df_aggr['circuit_max_straight'] = circuit_max_straight
    circuit_total_corner_length, circuit_mean_corner_length,circuit_max_corner_length,circuit_total_corner_curvature,circuit_mean_corner_curvature,circuit_max_corner_curvature,circuit_std_corner_curvature,circuit_number_of_corners = circuit_corner(lap_df)
    lap_df_aggr['circuit_total_corner_length'] =circuit_total_corner_length
    lap_df_aggr['circuit_mean_corner_length'] =circuit_mean_corner_length
    lap_df_aggr['circuit_max_corner_length'] =circuit_max_corner_length
    lap_df_aggr['circuit_total_corner_curvature'] = circuit_total_corner_curvature
    lap_df_aggr['circuit_mean_corner_curvature'] =circuit_mean_corner_curvature
    lap_df_aggr['circuit_max_corner_curvature'] =circuit_max_corner_curvature
    lap_df_aggr['circuit_std_corner_curvature'] =circuit_std_corner_curvature
    lap_df_aggr['circuit_number_of_corners'] =circuit_number_of_corners

    return lap_df_aggr


def car_avg_speed(lap_df):
    lapIds = lap_df['qualifying_lapId'].unique()
    top_speed=[]
    var_speed=[]
    bottom_speed_lap=[]
    straight_speed=[]
    var_straight_=[]
    
    for lap in lapIds:
        lap_query=(lap_df['qualifying_lapId'] == lap)
        if lap_df.loc[lap_query,'qualifying_TrackStatus'].all() == 1:
            max_speed,std_speed,min_speed,mean_straight_speed,var_straight_speed = speed_data(lap_df[lap_query].copy())
            top_speed.append(max_speed)
            var_speed.append(std_speed)
            bottom_speed_lap.append(min_speed)
            straight_speed.append(mean_straight_speed)
            var_straight_.append(var_straight_speed)
    return int(np.mean(top_speed)),int(np.mean(var_speed)),int(np.mean(bottom_speed_lap)),int(np.mean(mean_straight_speed)),int(np.mean(var_straight_speed))


def car_fast_lap_speed(lap_df):
    similarity = (lap_df['qualifying_LapTime']*1000-lap_df['fastest_all_sessions_milliseconds']).unique().min()
    query=(lap_df['qualifying_LapTime']*1000-lap_df['fastest_all_sessions_milliseconds']) == similarity
    max_fastest_lap_speed= int(lap_df.loc[query,'qualifying_Speed'].max())
    var_fastest_lap_speed = int(lap_df.loc[query,'qualifying_Speed'].std())
    min_fastest_lap_speed = int(lap_df.loc[query,'qualifying_Speed'].min())

    return max_fastest_lap_speed, var_fastest_lap_speed, min_fastest_lap_speed

def car_accleration(lap_df):
    lapIds = lap_df['qualifying_lapId'].unique()
    top_accleration=[]
    var_accleration=[]
    min_accleration=[]        
    for lap in lapIds:
        lap_query=(lap_df['qualifying_lapId'] == lap)
        if int(lap_df.loc[lap_query,'qualifying_TrackStatus'].all()) == 1:  
            acc_= acceleration(lap_df[lap_query].copy())
            if len(acc_) !=0:                
                top_accleration.append(np.max(acc_))
                var_accleration.append(np.std(acc_))
                min_accleration.append(np.min(acc_))
    return int(np.mean(top_accleration)),int(np.mean(var_accleration)),int(np.mean(min_accleration))

def car_fast_lap_accleration(lap_df):
    similarity = (lap_df['qualifying_LapTime']*1000-lap_df['fastest_all_sessions_milliseconds']).unique().min()
    query=(lap_df['qualifying_LapTime']*1000-lap_df['fastest_all_sessions_milliseconds']) == similarity
    max_fastest_accleration= int(np.max(acceleration(lap_df.loc[query].copy())))
    var_fastest_accleration= int(np.std(acceleration(lap_df.loc[query].copy())))
    min_fastest_accleration= int(np.min(acceleration(lap_df.loc[query].copy())))
    return max_fastest_accleration,var_fastest_accleration,min_fastest_accleration


def car_rpm(lap_df):
    lapIds = lap_df['qualifying_lapId'].unique()
    max_max_rpm=[]     
    straight_var_rpm=[]
    straight_mean_rpm =[]  
    for lap in lapIds:
        lap_query=(lap_df['qualifying_lapId'] == lap)
        if lap_df.loc[lap_query,'qualifying_TrackStatus'].all() == 1:  
            max_rpm,mean_straight_rpm, var_straight_rpm= rpm(lap_df[lap_query].copy())
            max_max_rpm.append(np.mean(max_rpm))            
            straight_mean_rpm.append(np.mean(mean_straight_rpm))
            straight_var_rpm.append(np.mean(var_straight_rpm))
    return int(np.max(max_max_rpm)), int(np.mean(straight_var_rpm)), int(np.mean(straight_mean_rpm))    
 

def car_fast_lap_rpm(lap_df):
    similarity = (lap_df['qualifying_LapTime']*1000-lap_df['fastest_all_sessions_milliseconds']).unique().min()
    query=(lap_df['qualifying_LapTime']*1000-lap_df['fastest_all_sessions_milliseconds']) == similarity
    max_rpm_fl,mean_straight_rpm_fl, var_straight_rpm_fl= rpm(lap_df.loc[query].copy())
    return int(max_rpm_fl), int(np.mean(var_straight_rpm_fl)), int(np.mean(mean_straight_rpm_fl))




def car_aggregations(lap_df_aggr,lap_df):

    lap_df_aggr['max_max_speed'] = int(lap_df['qualifying_Speed'].max())

    max_fastest_lap_speed, var_fastest_lap_speed, min_fastest_lap_speed = car_fast_lap_speed(lap_df)
    max_lap_speed,var_lap_speed,bottom_lap_speed,mean_straight_speed,var_straight_speed=car_avg_speed(lap_df)
    max_fastest_accleration,var_fastest_accleration,min_fastest_accleration = car_fast_lap_accleration(lap_df)
    mean_max_lap_accleration,mean_var_lap_accleration,mean_min_lap_accleration = car_accleration(lap_df)
    max_max_rpm, mean_var_straight_rpm, mean_straight_rpm= car_rpm(lap_df)
    max_fastest_lap_rpm, var_fastest_lap_straight_rpm, mean_fastest_lap_straight_rpm = car_fast_lap_rpm(lap_df)
    lap_df_aggr['max_fastest_lap_speed'] =max_fastest_lap_speed
    lap_df_aggr['var_fastest_lap_speed'] =var_fastest_lap_speed
    lap_df_aggr['min_fastest_lap_speed'] =min_fastest_lap_speed
    lap_df_aggr['max_lap_speed'] =max_lap_speed
    lap_df_aggr['var_lap_speed'] =var_lap_speed
    lap_df_aggr['bottom_lap_speed'] =bottom_lap_speed
    lap_df_aggr['mean_straight_speed'] =mean_straight_speed
    lap_df_aggr['var_straight_speed'] =var_straight_speed
    lap_df_aggr['max_fastest_accleration'] =max_fastest_accleration
    lap_df_aggr['var_fastest_accleration'] = var_fastest_accleration
    lap_df_aggr['min_fastest_accleration'] =min_fastest_accleration
    lap_df_aggr['mean_max_lap_accleration'] =mean_max_lap_accleration
    lap_df_aggr['mean_var_lap_accleration'] = mean_var_lap_accleration
    lap_df_aggr['mean_min_lap_accleration'] = mean_min_lap_accleration
    lap_df_aggr['max_fastest_lap_rpm'] =max_fastest_lap_rpm
    lap_df_aggr['var_fastest_lap_straight_rpm'] =var_fastest_lap_straight_rpm
    lap_df_aggr['mean_fastest_lap_straight_rpm'] = mean_fastest_lap_straight_rpm
    lap_df_aggr['max_max_rpm'] = max_max_rpm
    lap_df_aggr['mean_var_straight_rpm'] = mean_var_straight_rpm
    lap_df_aggr['mean_straight_rpm'] = mean_straight_rpm

    return lap_df_aggr

def gear_laps(lap_df):
    lapIds = lap_df['qualifying_lapId'].unique()
    gear_1_times=[]
    gear_2_times=[]  
    gear_3_times=[]  
    gear_4_times=[]  
    gear_5_times=[]  
    gear_6_times=[]  
    gear_7_times=[]  
    gear_8_times=[]     
    for lap in lapIds:
        lap_query=(lap_df['qualifying_lapId'] == lap)
        if lap_df.loc[lap_query,'qualifying_TrackStatus'].all() == 1:                
            gear_dict= gear_data(lap_df[lap_query].copy())
            gear_1_times.append(np.mean(gear_dict['gear_1']))            
            gear_2_times.append(np.mean(gear_dict['gear_2']))
            gear_3_times.append(np.mean(gear_dict['gear_3']))
            gear_4_times.append(np.mean(gear_dict['gear_4']))            
            gear_5_times.append(np.mean(gear_dict['gear_5']))
            gear_6_times.append(np.mean(gear_dict['gear_6']))
            gear_7_times.append(np.mean(gear_dict['gear_7']))
            gear_8_times.append(np.mean(gear_dict['gear_8']))
    return int(np.mean(gear_1_times)),int(np.mean(gear_2_times)),int(np.mean(gear_3_times)),int(np.mean(gear_4_times)),int(np.mean(gear_5_times)),int(np.mean(gear_6_times)),int(np.mean(gear_7_times)),int(np.mean(gear_8_times))


def gear_fast_lap(lap_df):
    similarity = (lap_df['qualifying_LapTime']*1000-lap_df['fastest_all_sessions_milliseconds']).unique().min()
    query=(lap_df['qualifying_LapTime']*1000-lap_df['fastest_all_sessions_milliseconds']) == similarity
    gear_dict_fl= gear_data(lap_df.loc[query].copy())
    return int(np.mean(gear_dict_fl['gear_1'])), int(np.mean(gear_dict_fl['gear_2'])),int(np.mean(gear_dict_fl['gear_3'])),int(np.mean(gear_dict_fl['gear_4'])),int(np.mean(gear_dict_fl['gear_5'])),int(np.mean(gear_dict_fl['gear_6'])),int(np.mean(gear_dict_fl['gear_7'])),int(np.mean(gear_dict_fl['gear_8']))

  
def gear_aggregations(lap_df_aggr,lap_df):
    avg_gear1_time,avg_gear2_time,avg_gear3_time,avg_gear4_time,avg_gear5_time,avg_gear6_time,avg_gear7_time,avg_gear8_time = gear_laps(lap_df)
    avg_gear1_time_fl,avg_gear2_time_fl,avg_gear3_time_fl,avg_gear4_time_fl,avg_gear5_time_fl,avg_gear6_time_fl,avg_gear7_time_fl,avg_gear8_time_fl = gear_fast_lap(lap_df)
    lap_df_aggr['avg_gear1_time'] =avg_gear1_time
    lap_df_aggr['avg_gear2_time'] = avg_gear2_time
    lap_df_aggr['avg_gear3_time'] = avg_gear3_time
    lap_df_aggr['avg_gear4_time'] = avg_gear4_time
    lap_df_aggr['avg_gear5_time'] = avg_gear5_time
    lap_df_aggr['avg_gear6_time'] = avg_gear6_time
    lap_df_aggr['avg_gear7_time'] = avg_gear7_time
    lap_df_aggr['avg_gear8_time'] = avg_gear8_time
    lap_df_aggr['avg_gear1_time_fl'] =avg_gear1_time_fl
    lap_df_aggr['avg_gear2_time_fl'] = avg_gear2_time_fl
    lap_df_aggr['avg_gear3_time_fl'] = avg_gear3_time_fl
    lap_df_aggr['avg_gear4_time_fl'] =avg_gear4_time_fl
    lap_df_aggr['avg_gear5_time_fl'] = avg_gear5_time_fl
    lap_df_aggr['avg_gear6_time_fl'] = avg_gear6_time_fl
    lap_df_aggr['avg_gear7_time_fl'] =avg_gear7_time_fl
    lap_df_aggr['avg_gear8_time_fl'] = avg_gear8_time_fl

    return lap_df_aggr

def driver_lap_aggregations(lap_df):
    lapIds = lap_df['qualifying_lapId'].unique()
    lap_time_on_brakes=[]
    lap_distance_on_brakes=[]
    lap_bottom_speed =[]
    lap_max_corner_speed =[]
    lap_bottom_speed_tightness_corner =[]
    drs_open_time=[]
    drs_open_distance=[]
    for lap in lapIds:
        lap_query=lap_df['qualifying_lapId'] == lap
        if lap_df.loc[lap_query,'qualifying_TrackStatus'].all() == 1:
            drs_time,drs_distance=DRS_open(lap_df[lap_query].copy())
            drs_open_distance.append(drs_distance)
            drs_open_time.append(drs_time)
            bottom_speed,max_corner_speed,bottom_speed_tightness_corner=driver_corners(lap_df[lap_query].copy())
            lap_bottom_speed_tightness_corner.append(bottom_speed_tightness_corner)
            lap_bottom_speed.append(np.mean(bottom_speed))
            lap_max_corner_speed.append(np.max(max_corner_speed))
            time_on_brakes, distance_on_brakes=driver_brake(lap_df[lap_query].copy())
            lap_time_on_brakes.append(time_on_brakes)
            lap_distance_on_brakes.append(distance_on_brakes)
    return int(np.mean(lap_time_on_brakes)),int(np.mean(lap_distance_on_brakes)),int(np.mean(drs_open_time)),int(np.mean(drs_open_distance)),int(np.mean(lap_bottom_speed)),int(np.mean(lap_max_corner_speed)),int(np.mean(lap_bottom_speed_tightness_corner))

def driver_fast_lap_aggregations(lap_df):
    similarity = (lap_df['qualifying_LapTime']*1000-lap_df['fastest_all_sessions_milliseconds']).unique().min()
    query=(lap_df['qualifying_LapTime']*1000-lap_df['fastest_all_sessions_milliseconds']) == similarity 
    fl_time_on_brakes, fl_distance_on_brakes=driver_brake(lap_df.loc[query].copy())
    fl_drs_time,fl_drs_distance=DRS_open(lap_df.loc[query].copy())
    fl_bottom_speed,fl_max_corner_speed,bottom_speed_tightness_corner=driver_corners(lap_df.loc[query].copy())
    return int(fl_time_on_brakes),int(fl_distance_on_brakes),int(fl_drs_time),int(fl_drs_distance),int(np.mean(fl_bottom_speed)),int(np.max(fl_max_corner_speed)),int(bottom_speed_tightness_corner)

def driver_aggregations(lap_df_aggr,lap_df):
    avg_lap_time_on_brake,avg_lap_distance_on_brake, avg_lap_time_on_DRS, avg_lap_distance_on_DRS,avg_lap_bottom_speed_corner, avg_lap_max_speed_corner,  avg_lap_bottom_speed_tightest_corner  =driver_lap_aggregations(lap_df)
    lap_df_aggr['avg_lap_time_on_brake'] =avg_lap_time_on_brake
    lap_df_aggr['avg_lap_distance_on_brake'] = avg_lap_distance_on_brake
    lap_df_aggr['avg_lap_time_on_DRS'] = avg_lap_time_on_DRS
    lap_df_aggr['avg_lap_distance_on_DRS'] = avg_lap_distance_on_DRS
    lap_df_aggr['avg_lap_bottom_speed_corner'] = avg_lap_bottom_speed_corner
    lap_df_aggr['avg_lap_max_speed_corner'] = avg_lap_max_speed_corner
    lap_df_aggr['avg_lap_bottom_speed_tightest_corner'] = avg_lap_bottom_speed_tightest_corner
    fl_lap_time_on_brake, fl_lap_distance_on_brake, fl_lap_time_on_DRS, fl_lap_distance_on_DRS, fl_lap_bottom_speed_corner, fl_lap_max_speed_corner, fl_lap_bottom_speed_tightest_corner = driver_fast_lap_aggregations(lap_df)
    lap_df_aggr['fl_lap_time_on_brake'] =fl_lap_time_on_brake
    lap_df_aggr['fl_lap_distance_on_brake'] = fl_lap_distance_on_brake
    lap_df_aggr['fl_lap_time_on_DRS'] = fl_lap_time_on_DRS
    lap_df_aggr['fl_lap_distance_on_DRS'] = fl_lap_distance_on_DRS
    lap_df_aggr['fl_lap_bottom_speed_corner'] = fl_lap_bottom_speed_corner
    lap_df_aggr['fl_lap_max_speed_corner'] = fl_lap_max_speed_corner
    lap_df_aggr['fl_lap_bottom_speed_tightest_corner'] = fl_lap_bottom_speed_tightest_corner
    
    return lap_df_aggr



def tyre_aggregations(lap_df_aggr,lap_df):
    similarity = (lap_df['qualifying_LapTime']*1000-lap_df['fastest_all_sessions_milliseconds']).unique().min()
    query=(lap_df['qualifying_LapTime']*1000-lap_df['fastest_all_sessions_milliseconds']) == similarity    
    fastestlap_tyre = (lap_df.loc[query,'qualifying_Compound'].mode().values[0])
    fastestlap_tyre_life=(lap_df.loc[query,'qualifying_TyreLife'].mean())
    lap_df_aggr['fl_tyre'] =fastestlap_tyre
    lap_df_aggr['fl_tyre_life'] = fastestlap_tyre_life

    return lap_df_aggr

def sector_laps(lap_df):
    lapIds = lap_df['qualifying_lapId'].unique()
    avg_Sector1=[]
    avg_Sector2=[]
    avg_Sector3=[]
    for lap in lapIds:
        lap_query=(lap_df['qualifying_lapId'] == lap)
        if lap_df.loc[lap_query,'qualifying_TrackStatus'].all() == 1:
            avg_Sector1.append(lap_df.loc[lap_query,'qualifying_Sector1Time'].mean())
            avg_Sector2.append(lap_df.loc[lap_query,'qualifying_Sector2Time'].mean())
            avg_Sector3.append(lap_df.loc[lap_query,'qualifying_Sector3Time'].mean())
    return np.mean(avg_Sector1),np.mean(avg_Sector2),np.mean(avg_Sector3)

def sector_fast_lap(lap_df):
    similarity = (lap_df['qualifying_LapTime']*1000-lap_df['fastest_all_sessions_milliseconds']).unique().min()
    query=(lap_df['qualifying_LapTime']*1000-lap_df['fastest_all_sessions_milliseconds']) == similarity
    
    fastestlap_Sector1 = (lap_df.loc[query,'qualifying_Sector1Time'].mean())
    fastestlap_Sector2=(lap_df.loc[query,'qualifying_Sector2Time'].mean())
    fastestlap_Sector3=(lap_df.loc[query,'qualifying_Sector3Time'].mean())

    return fastestlap_Sector1,fastestlap_Sector2,fastestlap_Sector3
 
    

def sector_aggregations(lap_df_aggr,lap_df):
    fastestlap_Sector1,fastestlap_Sector2, fastestlap_Sector3= sector_fast_lap(lap_df)
    lap_df_aggr['fastestlap_Sector1'] =fastestlap_Sector1
    lap_df_aggr['fastestlap_Sector2'] = fastestlap_Sector2
    lap_df_aggr['fastestlap_Sector3'] = fastestlap_Sector3
    avglap_Sector1,avglap_Sector2,avglap_Sector3 =sector_laps(lap_df)
    lap_df_aggr['avglap_Sector1'] =avglap_Sector1
    lap_df_aggr['avglap_Sector2'] = avglap_Sector2
    lap_df_aggr['avglap_Sector3'] = avglap_Sector3


    return lap_df_aggr


def weather_laps(lap_df):
    lapIds = lap_df['qualifying_lapId'].unique()
    avg_lap_percentagerainfall=[]
    avg_lap_track_temperature=[]
    avg_lap_humidty=[]
    for lap in lapIds:
        lap_query=(lap_df['qualifying_lapId'] == lap)
        if lap_df.loc[lap_query,'qualifying_TrackStatus'].all() == 1:
            avg_lap_percentagerainfall.append(lap_df.loc[lap_query,'qualifying_Rainfall'].sum()/len(lap_df.loc[lap_query,'qualifying_Rainfall'])*100)
            avg_lap_track_temperature.append(lap_df.loc[lap_query,'qualifying_TrackTemp'].mean())
            avg_lap_humidty.append(lap_df.loc[lap_query,'qualifying_Humidity'].mean())
    return np.mean(avg_lap_percentagerainfall),np.mean(avg_lap_track_temperature),np.mean(avg_lap_humidty)
    
def weather_fast_lap(lap_df):
    similarity = (lap_df['qualifying_LapTime']*1000-lap_df['fastest_all_sessions_milliseconds']).unique().min()
    query=(lap_df['qualifying_LapTime']*1000-lap_df['fastest_all_sessions_milliseconds']) == similarity
    fastestlap_percentagerainfall = (lap_df.loc[query,'qualifying_Rainfall'].sum()/len(lap_df.loc[query,'qualifying_Rainfall'])*100)
    fastestlap_track_temperature=(lap_df.loc[query,'qualifying_TrackTemp'].mean())
    fastestlap_humidity=(lap_df.loc[query,'qualifying_Humidity'].mean())
    return fastestlap_percentagerainfall, fastestlap_track_temperature, fastestlap_humidity





def weather_aggregations(lap_df_aggr,lap_df):    
    fastestlap_percentagerainfall, fastestlap_track_temperature, fastestlap_humidity = weather_fast_lap(lap_df)
    lap_df_aggr['fastestlap_percentagerainfall'] =fastestlap_percentagerainfall
    lap_df_aggr['fastestlap_track_temperature'] = fastestlap_track_temperature
    lap_df_aggr['fastestlap_humidity'] = fastestlap_humidity
    avg_lap_percentagerainfall, avg_lap_track_temperature, avg_lap_humidty = weather_laps(lap_df)
    lap_df_aggr['avg_lap_percentagerainfall'] =avg_lap_percentagerainfall
    lap_df_aggr['avg_lap_track_temperature'] = avg_lap_track_temperature
    lap_df_aggr['avg_lap_humidty'] = avg_lap_humidty

    return lap_df_aggr

def get_age(lap_df_aggr,lap_df):
    age = (lap_df['qualifying_Date'] - pd.to_datetime(lap_df_aggr['dob'])).astype('<m8[Y]').mean()
    return age
   
def all_aggregations(lap_df_aggr, lap_df):
    lap_df_aggr= circuit_aggregations(lap_df_aggr,lap_df)
    lap_df_aggr = car_aggregations(lap_df_aggr,lap_df)
    lap_df_aggr= gear_aggregations(lap_df_aggr,lap_df)
    lap_df_aggr= driver_aggregations(lap_df_aggr,lap_df)
    lap_df_aggr= tyre_aggregations(lap_df_aggr,lap_df)
    lap_df_aggr= sector_aggregations(lap_df_aggr,lap_df)
    lap_df_aggr= weather_aggregations(lap_df_aggr,lap_df)
    lap_df_aggr['age']= get_age(lap_df_aggr,lap_df)
    return lap_df_aggr

def clean_aggregations(lap_df_aggr, columns=['name','dob']):
    lap_df_aggr.drop(columns =columns,axis=1, inplace=True)
    return lap_df_aggr


def pull_clean_aggregate_telemetry(file,fast_time='ergast'):

    df = prepare_results_dataframe(file)
    df = clean_quali_times(df)
    
    lap_dfs=[]
    indexes = df.groupby(['year','name'])['raceId'].unique().index
    for year,name in indexes:        
        data = get_fastf1_session(year,name)
        print(f'Loaded data for {year} {name} ')
        df_race=df[(df['year'] == year) & (df['name'] == name)].copy()
        for index, row in df_race.iterrows():           
            lap_df= session_lap_driver(data,int(row['number']))
            
            if len(lap_df) >0 :# no lap data available for the driver pass
                try:
                    lap_df = clean_lap_df(lap_df)

                    if len(lap_df) >0 : # bad data means cleaning removes all data therefore pass
                        lap_df_aggr= row.copy()
                        if (fast_time == 'ergast') & (pd.notnull(row['fastest_lap_milliseconds']) == True):
                            lap_df = add_col(lap_df,'fastest_lap_milliseconds',row['fastest_lap_milliseconds'])
                            lap_df = add_col(lap_df,'fastest_all_sessions_milliseconds',row['fastest_all_sessions_milliseconds'])
                            lap_df['fastest_lap_milliseconds'] =lap_df['fastest_lap_milliseconds'].astype('float')
                            lap_df['fastest_lap_milliseconds'] =lap_df['fastest_lap_milliseconds'].astype('float')
                        else:
                            lap_df['fastest_lap_milliseconds'] = pick_fastest_lap(lap_df)
                            lap_df['fastest_all_sessions_milliseconds'] = pick_fastest_lap(lap_df)
                            lap_df_aggr['fastest_lap_milliseconds'] = pick_fastest_lap(lap_df)
                            lap_df_aggr['fastest_all_sessions_milliseconds'] = pick_fastest_lap(lap_df)
                        lap_df = replace_fast_nan(lap_df)
                        lap_df = laps_corners(lap_df)
                        
                        lap_df_aggr=all_aggregations(lap_df_aggr,lap_df)
                        lap_df_aggr=clean_aggregations(lap_df_aggr,lap_df)
                        lap_dfs.append(lap_df_aggr.to_frame().T)
                except:
                    pass
            else:
                pass
            print(f"completed driver {row['number']} ", end='\r')
    print(f'completed {year} {name}')
    Lap_aggr_combined=pd.concat(lap_dfs)
    
    return Lap_aggr_combined


#%%


def get_year_quali():
    season_fast1_df = fastf1.get_event_schedule(year=datetime.today().year,include_testing=False)
    circuit_mapper={
    'Sakhir': 'bahrain',
    'Jeddah': 'jeddah',
    'Melbourne':'albert_park',
    'Imola':'imola',
    'Miami':'miami',
    'Barcelona': 'catalunya',
    'Monaco': 'monaco',
    'Baku': 'baku',
    'Montréal' : 'villeneuve',
    'Silverstone':'silverstone',
    'Spielberg': 'red_bull_ring',
    'Le Castellet':'ricard',
    'Budapest':'hungaroring',
    'Spa-Francorchamps':'spa',
    'Zandvoort':'zandvoort',
    'Monza':'monza',
    'Marina Bay': 'marina_bay',
    'Suzuka': 'suzuka',
    'Austin': 'americas',
    'Mexico City': 'rodriguez',
    'São Paulo': 'interlagos',
    'Yas Island': 'yas_marina'
    }
    season_fast1_df['Location'] = season_fast1_df['Location'].map(circuit_mapper)
    season_fast1_df['quali_date']=''
    for i,row in season_fast1_df.iterrows():
        if row['Session1']=='Qualifying':
            season_fast1_df.loc[i,'quali_date']= row['Session1Date']
        elif row['Session2']=='Qualifying':
            season_fast1_df.loc[i,'quali_date']= row['Session2Date']
        elif row['Session3']=='Qualifying':
            season_fast1_df.loc[i,'quali_date']= row['Session3Date']
        elif row['Session4']=='Qualifying':
            season_fast1_df.loc[i,'quali_date']= row['Session4Date']
        elif row['Session5']=='Qualifying':
            season_fast1_df.loc[i,'quali_date']= row['Session5Date']
    event_df= season_fast1_df[['RoundNumber','Location','quali_date']].copy()
    return event_df

def new_sessions(data,season_df):

    existingquery = season_df['Location'].isin(list(data.loc[data['year']==datetime.today().year,'circuitRef'].unique()))
    datequery = season_df['quali_date']< datetime.today()

    return event_df[(~existingquery) & (datequery)]





# %%
