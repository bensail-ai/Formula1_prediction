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
def f1_telemetry_data(year:float,round:float,session=['qualifying']):
    """
    This function returns the pandas dataframe Fastf1 telemetry for all the drivers
    in the specified session in  GrandPrix 

    Function loads the session, returns the laps, weather and telemetry.

    Combined the lap results, weather and telemtry data into one dataframe. 
    Args:
        year (float): year of the GrandPrix
        round (float): round number of the GrandPrix
        session (list, optional): options are "qualifying" or "practice3". Defaults to ['qualifying'].

    Returns:
        dict: key is session and value is the combined dataframe
    """
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
                try: # sometimes telemetry data doesn't exist so try to download just car data for that driver
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



def combine_telemetry(event_data:dict,ergast_df:pd.DataFrame,session='qualifying',year=2018):
    """
    This Function combines the lap telemetry data with Ergast result Dataframe

    Inputs are nested dictionary of circuits and sessions

    Args:
        event_data (dict): this is a dataframe of lap telemetry data, keys are the circuits and values is another dictionary where key
        is session name and value pandas DataFrame of lap telemetry data
        ergast_df (pd.DataFrame): Ergast cleaned results DataFrame
        session (str, optional): options are "qualifying" or "practice3". Defaults to 'qualifying'.
        year (int, optional): year of the races. Defaults to 2018.

    Returns:
        pd.DataFrame: DataFrame of merged Telemetry Data with Ergast result DataFrame
    """
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

def get_fastf1_session(year:int,round:int,session='qualifying'):
    """
    Function returns the FastF1 session object loaded for the inputs

    Args:
        year (int): year of the GrandPrix
        round (int): round number of the GrandPrix
        session (str, optional): session name options are "qualifying" or "practice3". Defaults to 'qualifying'.

    Returns:
        session: FastF1 session object
    """

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
    """
    Creates the pandas dataframe of lap telemetry data for a driver from a specified FastF1 session object

    Args:
        data (FastF1 session): Fastf1 session object already loaded
        driver (int): driver number
        session (str, optional): session name. Defaults to 'qualifying'.

    Returns:
        pd.DataFrame: DataFrame of Fastf1 lap telemetry data merged with weather data
    """

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


def prepare_results_dataframe(file:str):
    """
    This function reads the combined and clean Ergast results dataframe and retains the columns required to merge with
    FastF1 telemetry data

    The file should be located ./data/clean

    Args:
        file (str): file path to the Ergast cleaned csv file

    Returns:
        pd.DataFrame: cleaned Ergast results DataFrame
    """
    ergast_combined_df=pd.read_csv(file)
    qualify_df= ergast_combined_df[['raceId', 'driverRef', 'number','name','circuitRef', 'country', 'nationality_drivers',
       'constructorRef', 'nationality_constructors', 'year', 'lat_x', 'lng_x',
       'alt', 'quali_position', 'dob','q1','q2','q3']]
    qualify_df= qualify_df[qualify_df['year'] >=2018]
    qualify_df.reset_index(drop=True, inplace=True)
    return qualify_df

def clean_quali_times(df:pd.DataFrame):
    """
    Function that cleans the Qualifying results in the Ergast DataFrame
    Converts objects to milliseconds
    Combines Q1,Q2,Q3 into one column and finds the fastest times of all sessions

    Args:
        df (pd.DataFrame): Ergast results DataFrame

    Returns:
        pd.DataFrame: cleaned qualifying times of Ergast results DataFrame
    """
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

def add_col(df:pd.DataFrame,col:str,value):
    """
    Adding a column and a value to pandas DataFrame

    Args:
        df (pd.DataFrame): DataFrame
        col (str): column name
        value (float): value of column

    Returns:
        pd.DataFrame: pandas DataFrame
    """
    df[col] = value

    return df


def clean_time(df:pd.DataFrame, columns =['qualifying_end_lap_sessiontime','qualifying_LapTime'
,'qualifying_Sector1Time','qualifying_Sector2Time','qualifying_Sector3Time',
'qualifying_lap_timedelta']):
    """
    Function cleans the time columns of lap telemetry data from FastF1

    returns the columns as total seconds not pandas time delta objects

    Args:
        df (pd.DataFrame): dataframe lap telemetry data merged with Ergast results DataFrame
        columns (list, optional): list of columns to clean. Defaults to ['qualifying_end_lap_sessiontime','qualifying_LapTime' ,
        'qualifying_Sector1Time','qualifying_Sector2Time','qualifying_Sector3Time', 'qualifying_lap_timedelta'].

    Returns:
        pd.DataFrame: cleaned DataFrame
    """

    df.rename(columns ={'qualifying_Time_x':'qualifying_end_lap_sessiontime','qualifying_Time_y':'qualifying_lap_timedelta'},inplace=True)
    
    
    for col in columns:
        if df[col].dtype == '<m8[ns]':            
            df[col] = df[col].dt.total_seconds()
        else:
            df[col] = df[col].astype('timedelta64')
            df[col] = df[col].dt.total_seconds()

    return df


def clean_laps(lap_df:pd.DataFrame):
    """
    Function cleans the lap telemetry data before feature aggregation

    It checks the distance columns and removes laps with poor distance data
    by looking for laps that are too short or
    the distance data is an exponetial curve and not linear as expected
    it does this by checking that the max is less than 100
    and checking that the difference between quarter of the maximum and 25% quartile is small (exponential check)

    returns the clean laps

    Args:
        lap_df (pd.DataFrame): FastF1 lap telemetry data merged with Ergast results DataFrame

    Returns:
        pd.DataFrame: cleaned laps
    """
    lapIds = lap_df['qualifying_lapId'].unique()
    laps=[]
    for lap in lapIds:
        distance = lap_df.loc[lap_df['qualifying_lapId']==lap,'qualifying_Distance']
        if distance.max() < 100:
            laps.append(lap)
        elif (distance.min() <= 0) & (((distance.max()/4) - distance.describe()['25%'])>250):
            laps.append(lap)
    lap_df=lap_df[~(lap_df['qualifying_lapId'].isin(laps))]
    lap_df= lap_df[lap_df['qualifying_Distance'].notnull()]
    return lap_df



def clean_lap_df(lap_df:pd.DataFrame):
    """
    Cleans the FastF1 telemetry laps, fills nans by the mean of that qualifying stint
    removes not necessary columns 
    cleans the time and the distance column

    Args:
        lap_df (pd.DataFrame): FastF1 lap telemetry data merged with Ergast results DataFrame

    Returns:
        pd.DataFrame: cleaned laps
    """
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

def pick_fastest_lap(lap_df:pd.DataFrame):
    """
    Finds the fastest lap in the FastF1 lap telemetry data

    Args:
        lap_df (pd.DataFrame): FastF1 lap telemetry data

    Returns:
        float: the fast lap in milliseconds
    """
    fast_lap = lap_df['qualifying_LapTime'].min()*1000

    return fast_lap

def replace_fast_nan(lap_df:pd.DataFrame):
    """
    Fills in NaN values of the fastest lap from the Ergast Results dataframe with
    the fastest lap from the FastF1 lap telemtry data for a driver

    Args:
        lap_df (pd.DataFrame): FastF1 lap telemetry data merged with Ergast results DataFrame

    Returns:
        pd.DataFrame: combined DataFrame with NaNs filled
    """
    lap_df.loc[lap_df['fastest_all_sessions_milliseconds'].isna(),'fastest_all_sessions_milliseconds'] = pick_fastest_lap(lap_df)

    return lap_df

def laps_corners(lap_df:pd.DataFrame):
    """
    Flags the corners for all laps in the FastF1 lap telemtry data

    Args:
        lap_df (pd.DataFrame): FastF1 lap telemetry data merged with Ergast results DataFrame

    Returns:
        pd.DataFrame: lap telemetry DataFrame with corner flag added
    """
    lapIds = lap_df['qualifying_lapId'].unique()
    for lap in lapIds:
        
        lap_df.loc[lap_df['qualifying_lapId']==lap,'corner']= (flag_corners(lap_df[lap_df['qualifying_lapId']==lap].copy(),smoothing=10))['corner']
    return lap_df


def circuit_length(lap_df:pd.DataFrame):
    """
    Calculates the circuit length for a lap in FastF1 lap telemetry data

    Args:
        lap_df (pd.DataFrame): FastF1 lap telemetry data merged with Ergast results DataFrame

    Returns:
        int: mean circuit length (m)
    """
    lapIds = lap_df['qualifying_lapId'].unique()
    lengths=[]
    for lap in lapIds:
        lap_min = (lap_df.loc[(lap_df['qualifying_lapId'] ==lap),'qualifying_Distance'].min())
        lap_max = (lap_df.loc[(lap_df['qualifying_lapId'] ==lap),'qualifying_Distance'].max())
        lengths.append(lap_max-lap_min)
    return int(np.mean(lengths))


def circuit_straight(lap_df:pd.DataFrame):
    """
    Runs Feature aggregations on FastF1 lap telemetry data
    Calculates features on the circuit straight lengths

    Args:
        lap_df (pd.DataFrame): FastF1 lap telemetry data merged with Ergast results DataFrame
    Returns:
        int: mean total length of straights in the lap
        int: mean length of straights in the lap
        int: max length of straight in the lap
    """
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
    

def circuit_corner(lap_df:pd.DataFrame):
    """
    Calculates the corner feature aggregations for every lap in FastF1 combined DataFrame

    Args:
        lap_df (pd.DataFrame): FastF1 lap telemetry data merged with Ergast results DataFrame

    Returns:
        int: total corner length
        int: mean corner length
        int: max corner length
        int: sum of all corner curvatures
        int: mean corner curvature
        int: max corner curvature
        int: variance in corner curvature
        int: number of corners in the lap
    """
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
        if lap_df.loc[lap_query,'qualifying_TrackStatus'].all() == 1: #ensures it is a clear lap for driver
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


def circuit_aggregations(lap_df_aggr:pd.DataFrame,lap_df:pd.DataFrame):
    """
    Function runs the Feature Aggregations for Circuit characteristics and saves it to the aggregated
    Dataframe

    Args:
        lap_df_aggr (pd.DataFrame): Driver race record DataFrame with aggregated features
        lap_df (pd.DataFrame): FastF1 lap telemetry data merged with Ergast results DataFrame

    Returns:
        pd.DataFrame: DataFrame of aggregated lap features at a driver record level
    """
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


def car_avg_speed(lap_df:pd.DataFrame):
    """
    Function calculates car speed features

    Args:
        lap_df (pd.DataFrame): FastF1 lap telemetry data merged with Ergast results DataFrame

    Returns:
        int: mean top speed of the laps
        int: mean variance in speed of the laps
        int: mean minimum speed of the laps
        int: mean speed on the straights
        int: mean variance in speed on the straights
    """
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


def car_fast_lap_speed(lap_df:pd.DataFrame):
    """
    Finds the feature aggregations of speed of the fastest lap per driver in FastF1 lap telemetry data

    Args:
        lap_df (pd.DataFrame): FastF1 lap telemetry data merged with Ergast results DataFrame

    Returns:
        max_fastest_lap_speed (int): maximum speed on the fastest lap
        var_fastest_lap_speed (int): variance of speed on the fastest lap
        min_fastest_lap_speed (int): minimum speed on the fastest lap
    """
    similarity = (lap_df['qualifying_LapTime']*1000-lap_df['fastest_all_sessions_milliseconds']).unique().min()
    query=(lap_df['qualifying_LapTime']*1000-lap_df['fastest_all_sessions_milliseconds']) == similarity
    max_fastest_lap_speed= int(lap_df.loc[query,'qualifying_Speed'].max())
    var_fastest_lap_speed = int(lap_df.loc[query,'qualifying_Speed'].std())
    min_fastest_lap_speed = int(lap_df.loc[query,'qualifying_Speed'].min())

    return max_fastest_lap_speed, var_fastest_lap_speed, min_fastest_lap_speed

def car_accleration(lap_df:pd.DataFrame):
    """
    Calculates the cars acceleration for all the laps in the FastF1 lap telemetry dataframe

    Args:
        lap_df (pd.DataFrame): FastF1 lap telemetry data merged with Ergast results DataFrame

    Returns:
        int: mean of fastest acceleration of the laps
        int: mean of variance of accleration of the laps
        int: minimum accleration of the laps
    """
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

def car_fast_lap_accleration(lap_df:pd.DataFrame):
    """
    Calculates the accleration features for the fastest lap of the combined FastF1 lap telemetry DataFrame

    Args:
        lap_df (pd.DataFrame): FastF1 lap telemetry data merged with Ergast results DataFrame

    Returns:
        max_fastest_accleration (int): maximum acceleration on the fastest lap
        var_fastest_accleration (int): variance in acceleration on the fastest lap
        min_fastest_accleration (int): minimum acceleration on the fastest lap
    """
    similarity = (lap_df['qualifying_LapTime']*1000-lap_df['fastest_all_sessions_milliseconds']).unique().min()
    query=(lap_df['qualifying_LapTime']*1000-lap_df['fastest_all_sessions_milliseconds']) == similarity
    max_fastest_accleration= int(np.max(acceleration(lap_df.loc[query].copy())))
    var_fastest_accleration= int(np.std(acceleration(lap_df.loc[query].copy())))
    min_fastest_accleration= int(np.min(acceleration(lap_df.loc[query].copy())))
    return max_fastest_accleration,var_fastest_accleration,min_fastest_accleration


def car_rpm(lap_df: pd.DataFrame):
    """
    Calculates the features of RPM on the combined FastF1 lap telemetry DataFrame

    Args:
        lap_df (pd.DataFrame): FastF1 lap telemetry data merged with Ergast results DataFrame

    Returns:
        int: the maximum of the maximum RPM on all the laps
        int: the mean variance in RPM on the straights
        int: the mean straight RPM on all the laps
    """
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
 

def car_fast_lap_rpm(lap_df:pd.DataFrame):
    """
    Calculates the features of RPM on the combined FastF1 lap telemetry DataFrame for the fastest laps only

    Args:
        lap_df (pd.DataFrame): FastF1 lap telemetry data merged with Ergast results DataFrame

    Returns:
        int: the maximum of the maximum RPM on the fastest lap
        int: the mean variance in RPM on the straights on the fastest lap
        int: the mean straight RPM on the fastest lap
    """
    similarity = (lap_df['qualifying_LapTime']*1000-lap_df['fastest_all_sessions_milliseconds']).unique().min()
    query=(lap_df['qualifying_LapTime']*1000-lap_df['fastest_all_sessions_milliseconds']) == similarity
    max_rpm_fl,mean_straight_rpm_fl, var_straight_rpm_fl= rpm(lap_df.loc[query].copy())
    return int(max_rpm_fl), int(np.mean(var_straight_rpm_fl)), int(np.mean(mean_straight_rpm_fl))




def car_aggregations(lap_df_aggr:pd.DataFrame,lap_df:pd.DataFrame):
    """
    Calculates all the car charactersitics features and saves the features to the aggregate DataFrame

    Args:
        lap_df_aggr (pd.DataFrame): Driver race record DataFrame with aggregated features
        lap_df (pd.DataFrame): FastF1 lap telemetry data merged with Ergast results DataFrame

    Returns:
        pd.DataFrame: Driver race record DataFrame with aggregated features
    """
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

def gear_laps(lap_df:pd.DataFrame):
    """
    Calculates the features of gear times for every lap in the FastF1 lap telemetry data merged with Ergast results DataFrame


    Args:
        lap_df (pd.DataFrame): FastF1 lap telemetry data merged with Ergast results DataFrame

    Returns:
        int: mean of gear 1 time seconds
        int: mean of gear 2 time seconds
        int: mean of gear 3 time seconds
        int: mean of gear 4 time seconds
        int: mean of gear 5 time seconds
        int: mean of gear 6 time seconds
        int: mean of gear 7 time seconds
        int: mean of gear 8 time seconds
 
    """
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


def gear_fast_lap(lap_df:pd.DataFrame):
    """
    Calculates the gear times for the fastest lap only in the FastF1 lap telemetry DataFrame

    Args:
        lap_df (pd.DataFrame): FastF1 lap telemetry data merged with Ergast results DataFrame

    Returns:
        int: mean of gear 1 time seconds
        int: mean of gear 2 time seconds
        int: mean of gear 3 time seconds
        int: mean of gear 4 time seconds
        int: mean of gear 5 time seconds
        int: mean of gear 6 time seconds
        int: mean of gear 7 time seconds
        int: mean of gear 8 time seconds
    """
    similarity = (lap_df['qualifying_LapTime']*1000-lap_df['fastest_all_sessions_milliseconds']).unique().min()
    query=(lap_df['qualifying_LapTime']*1000-lap_df['fastest_all_sessions_milliseconds']) == similarity
    gear_dict_fl= gear_data(lap_df.loc[query].copy())
    return int(np.mean(gear_dict_fl['gear_1'])), int(np.mean(gear_dict_fl['gear_2'])),int(np.mean(gear_dict_fl['gear_3'])),int(np.mean(gear_dict_fl['gear_4'])),int(np.mean(gear_dict_fl['gear_5'])),int(np.mean(gear_dict_fl['gear_6'])),int(np.mean(gear_dict_fl['gear_7'])),int(np.mean(gear_dict_fl['gear_8']))

  
def gear_aggregations(lap_df_aggr:pd.DataFrame,lap_df:pd.DataFrame):
    """
    Calculates all the gear charactersitics features and saves the features to the aggregate DataFrame

    Args:
        lap_df_aggr (pd.DataFrame): Driver race record DataFrame with aggregated features
        lap_df (pd.DataFrame): FastF1 lap telemetry data merged with Ergast results DataFrame

    Returns:
        pd.DataFrame: Driver race record DataFrame with aggregated features
    """
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

def driver_lap_aggregations(lap_df:pd.DataFrame):
    """
    Calculates the driver features of DRS time & Distance, Braking time & distance, Speed in the corners

    Args:
        lap_df (pd.DataFrame): FastF1 lap telemetry data merged with Ergast results DataFrame

    Returns:
        int: mean lap time on the bakes of all the laps
        int: mean lap distance on the brakes of all the laps
        int: mean drs open time of all the laps
        int: mean drs open distance of all the laps
        int: mean minimum speed of all the laps
        int: mean maximum speed in the corners of all the laps
        int: mean speed in the tightest corner of all the laps
    """
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

def driver_fast_lap_aggregations(lap_df:pd.DataFrame):
    """
    Calculate the driver features of DRS time & Distance, Braking time & distance, Speed in the corners for the fastest lap only

    Args:
        lap_df (pd.DataFrame): FastF1 lap telemetry data merged with Ergast results DataFrame

    Returns:
        int: mean lap time on the bakes of the fastest lap
        int: mean lap distance on the brakes of the fastest lap
        int: mean drs open time of the fastest lap
        int: mean drs open distance of the fastest lap
        int: mean minimum speed of the fastest lap
        int: mean maximum speed in the corners of the fastest lap
        int: mean speed in the tightest corner of the fastest lap
    """
    similarity = (lap_df['qualifying_LapTime']*1000-lap_df['fastest_all_sessions_milliseconds']).unique().min()
    query=(lap_df['qualifying_LapTime']*1000-lap_df['fastest_all_sessions_milliseconds']) == similarity 
    fl_time_on_brakes, fl_distance_on_brakes=driver_brake(lap_df.loc[query].copy())
    fl_drs_time,fl_drs_distance=DRS_open(lap_df.loc[query].copy())
    fl_bottom_speed,fl_max_corner_speed,bottom_speed_tightness_corner=driver_corners(lap_df.loc[query].copy())
    return int(fl_time_on_brakes),int(fl_distance_on_brakes),int(fl_drs_time),int(fl_drs_distance),int(np.mean(fl_bottom_speed)),int(np.max(fl_max_corner_speed)),int(bottom_speed_tightness_corner)

def driver_aggregations(lap_df_aggr:pd.DataFrame,lap_df:pd.DataFrame):
    """
    Calculates all the driver charactersitics features and saves the features to the aggregate DataFrame

    Args:
        lap_df_aggr (pd.DataFrame): Driver race record DataFrame with aggregated features
        lap_df (pd.DataFrame): FastF1 lap telemetry data merged with Ergast results DataFrame

    Returns:
        pd.DataFrame: Driver race record DataFrame with aggregated features
    """
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



def tyre_aggregations(lap_df_aggr:pd.DataFrame,lap_df:pd.DataFrame):
    """
    Calculates the tyre feature aggregations on the fastest lap and saves it to the aggregate DataFrame

    Args:
        lap_df_aggr (pd.DataFrame): Driver race record DataFrame with aggregated features
        lap_df (pd.DataFrame): FastF1 lap telemetry data merged with Ergast results DataFrame

    Returns:
        pd.DataFrame: Driver race record DataFrame with aggregated features
    """
    similarity = (lap_df['qualifying_LapTime']*1000-lap_df['fastest_all_sessions_milliseconds']).unique().min()
    query=(lap_df['qualifying_LapTime']*1000-lap_df['fastest_all_sessions_milliseconds']) == similarity    
    fastestlap_tyre = (lap_df.loc[query,'qualifying_Compound'].mode().values[0])
    fastestlap_tyre_life=(lap_df.loc[query,'qualifying_TyreLife'].mean())
    lap_df_aggr['fl_tyre'] =fastestlap_tyre
    lap_df_aggr['fl_tyre_life'] = fastestlap_tyre_life

    return lap_df_aggr

def sector_laps(lap_df:pd.DataFrame):
    """
    Returns the average lap sector times from the FastF1 lap telemetry combined DataFrame

    Args:
        lap_df (pd.DataFrame): FastF1 lap telemetry data merged with Ergast results DataFrame

    Returns:
        float: avgerage time for sector 1
        float: average time for sector 2
        float: average time for sector 3
    """
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

def sector_fast_lap(lap_df:pd.DataFrame):
    """
    Returns the fastest lap sector times from the FastF1 lap telemetry combined DataFrame

    Args:
        lap_df (pd.DataFrame): FastF1 lap telemetry data merged with Ergast results DataFrame

    Returns:
        float: fastest lap time for sector 1
        float: fastest lap time for sector 2
        float: fastest lap time for sector 3
    """
    similarity = (lap_df['qualifying_LapTime']*1000-lap_df['fastest_all_sessions_milliseconds']).unique().min()
    query=(lap_df['qualifying_LapTime']*1000-lap_df['fastest_all_sessions_milliseconds']) == similarity
    
    fastestlap_Sector1 = (lap_df.loc[query,'qualifying_Sector1Time'].mean())
    fastestlap_Sector2=(lap_df.loc[query,'qualifying_Sector2Time'].mean())
    fastestlap_Sector3=(lap_df.loc[query,'qualifying_Sector3Time'].mean())

    return fastestlap_Sector1,fastestlap_Sector2,fastestlap_Sector3
 
    

def sector_aggregations(lap_df_aggr:pd.DataFrame,lap_df:pd.DataFrame):
    """
    Calculates the sector feature times and takes the average for all the laps. 
    Adds the features to the aggregate DataFrame

    Args:
        lap_df_aggr (pd.DataFrame): Driver race record DataFrame with aggregated features
        lap_df (pd.DataFrame): FastF1 lap telemetry data merged with Ergast results DataFrame

    Returns:
        pd.DataFrame: Driver race record DataFrame with aggregated features
    """
    fastestlap_Sector1,fastestlap_Sector2, fastestlap_Sector3= sector_fast_lap(lap_df)
    lap_df_aggr['fastestlap_Sector1'] =fastestlap_Sector1
    lap_df_aggr['fastestlap_Sector2'] = fastestlap_Sector2
    lap_df_aggr['fastestlap_Sector3'] = fastestlap_Sector3
    avglap_Sector1,avglap_Sector2,avglap_Sector3 =sector_laps(lap_df)
    lap_df_aggr['avglap_Sector1'] =avglap_Sector1
    lap_df_aggr['avglap_Sector2'] = avglap_Sector2
    lap_df_aggr['avglap_Sector3'] = avglap_Sector3


    return lap_df_aggr


def weather_laps(lap_df:pd.DataFrame):
    """
    Calculates the average rainfall, track temperature and humidty for all the laps in the combined FastF1 lap telemetry DataFrame

    Args:
        lap_df (pd.DataFrame): FastF1 lap telemetry data merged with Ergast results DataFrame

    Returns:
        float: average fraction of rainfall for all the laps
        float: average track temperature for all the laps
        float: average humidity for all the laps
    """
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
    
def weather_fast_lap(lap_df: pd.DataFrame):
    """
    Calculates the average rainfall, track temperature and humidty for the fastest lap per driver in the combined FastF1 lap telemetry DataFrame

    Args:
        lap_df (pd.DataFrame): FastF1 lap telemetry data merged with Ergast results DataFrame

    Returns:
        float: fraction of rainfall for the fastest lap
        float: track temperature for all the fastest lap
        float: humidity for all the fastest lap
    """
    similarity = (lap_df['qualifying_LapTime']*1000-lap_df['fastest_all_sessions_milliseconds']).unique().min()
    query=(lap_df['qualifying_LapTime']*1000-lap_df['fastest_all_sessions_milliseconds']) == similarity
    fastestlap_percentagerainfall = (lap_df.loc[query,'qualifying_Rainfall'].sum()/len(lap_df.loc[query,'qualifying_Rainfall'])*100)
    fastestlap_track_temperature=(lap_df.loc[query,'qualifying_TrackTemp'].mean())
    fastestlap_humidity=(lap_df.loc[query,'qualifying_Humidity'].mean())
    return fastestlap_percentagerainfall, fastestlap_track_temperature, fastestlap_humidity





def weather_aggregations(lap_df_aggr:pd.DataFrame,lap_df:pd.DataFrame):
    """
    Calculates the weather features for average laps and fastest laps. 
    Adds the features to the aggregate DataFrame

    Args:
        lap_df_aggr (pd.DataFrame): Driver race record DataFrame with aggregated features
        lap_df (pd.DataFrame): FastF1 lap telemetry data merged with Ergast results DataFrame

    Returns:
        pd.DataFrame: Driver race record DataFrame with aggregated features
    """    
    fastestlap_percentagerainfall, fastestlap_track_temperature, fastestlap_humidity = weather_fast_lap(lap_df)
    lap_df_aggr['fastestlap_percentagerainfall'] =fastestlap_percentagerainfall
    lap_df_aggr['fastestlap_track_temperature'] = fastestlap_track_temperature
    lap_df_aggr['fastestlap_humidity'] = fastestlap_humidity
    avg_lap_percentagerainfall, avg_lap_track_temperature, avg_lap_humidty = weather_laps(lap_df)
    lap_df_aggr['avg_lap_percentagerainfall'] =avg_lap_percentagerainfall
    lap_df_aggr['avg_lap_track_temperature'] = avg_lap_track_temperature
    lap_df_aggr['avg_lap_humidty'] = avg_lap_humidty

    return lap_df_aggr

def get_age(lap_df_aggr:pd.DataFrame,lap_df:pd.DataFrame):
    """
    calculates the age of the driver for every record in the aggregate DataFrame

    Args:
        lap_df_aggr (pd.DataFrame): Driver race record DataFrame with aggregated features
        lap_df (pd.DataFrame): FastF1 lap telemetry data merged with Ergast results DataFrame

    Returns:
        pd.DataFrame: Driver race record DataFrame with aggregated features
    """
    age = (lap_df['qualifying_Date'] - pd.to_datetime(lap_df_aggr['dob'])).astype('<m8[Y]').mean()
    return age
   
def all_aggregations(lap_df_aggr:pd.DataFrame, lap_df:pd.DataFrame):
    """
    Runs all the aggregate feature functions for:
    Circuit
    Car
    Gear
    Driver
    Tyre
    Weather
    and Age

    Args:
        lap_df_aggr (pd.DataFrame): Driver race record DataFrame with aggregated features
        lap_df (pd.DataFrame): FastF1 lap telemetry data merged with Ergast results DataFrame

    Returns:
        pd.DataFrame: Driver race record DataFrame with aggregated features
    """
    lap_df_aggr= circuit_aggregations(lap_df_aggr,lap_df)
    lap_df_aggr = car_aggregations(lap_df_aggr,lap_df)
    lap_df_aggr= gear_aggregations(lap_df_aggr,lap_df)
    lap_df_aggr= driver_aggregations(lap_df_aggr,lap_df)
    lap_df_aggr= tyre_aggregations(lap_df_aggr,lap_df)
    lap_df_aggr= sector_aggregations(lap_df_aggr,lap_df)
    lap_df_aggr= weather_aggregations(lap_df_aggr,lap_df)
    lap_df_aggr['age']= get_age(lap_df_aggr,lap_df)
    return lap_df_aggr

def clean_aggregations(lap_df_aggr:pd.DataFrame, columns=['name','dob']):
    """
    Function removes the columns from the input in the aggregate DataFrame

    Args:
        lap_df_aggr (pd.DataFrame): Driver race record DataFrame with aggregated features
        columns (list, optional): list of columns to remove. Defaults to ['name','dob'].

    Returns:
        pd.DataFrame: Driver race record DataFrame with aggregated features
    """
    lap_df_aggr.drop(columns =columns,axis=1, inplace=True)
    return lap_df_aggr


def pull_clean_aggregate_telemetry(file:str,fast_time='ergast'):
    """
    Function to download and create aggregate features for all the races in the Ergast results dataframe from 2018.

    The function loads the Ergast DataFrame and for each driver race record, pulls the telemetry from FastF1 and completes
    the feature aggregations 

    The file should be located ./data/clean for the ergast clean combined .csv file

    Args:
        file (str): file path to the Ergast cleaned csv file    
        fast_time (str, optional): Which fast time to use if "Ergast" uses the fastest lap time from the Ergast DataFrame. Defaults to 'ergast'.

    Returns:
        pd.DataFrame: The complete driver race record DataFrame with aggregated features
    """

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
    """

    Gets all the qualifying sessions as a DataFrame for this years season as a DataFrame, of circuit, Round and date from FastF1 API

    The the circuit names are mapped to match the Ergast DataFrame format

    Returns:
        pd.DataFrame: Summary DataFrame of this years Qualifying sessions
    """
    if os.path.exists('./f1_cache'):
        pass
    else:
        os.mkdir('./f1_cache')
    fastf1.Cache.enable_cache('./f1_cache')
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

def new_sessions(file: str,event_df: pd.DataFrame):
    """
    For the qualifying sessions for this year it returns the dataframe of the ones which are not in the Ergast cleaned file

    Args:
        file (str): file path to the Ergast cleaned csv file    
        event_df (pd.DataFrame): event DataFrame of qualifying events this year

    Returns:
        pd.DataFrame: The event DataFrame of the events which are not in the Ergast DataFrame
    """
   
    df = prepare_results_dataframe(file)
    existingquery = event_df['Location'].isin(list(df.loc[df['year']==datetime.today().year,'circuitRef'].unique()))
    datequery = event_df['quali_date']< datetime.today()

    return event_df[(~existingquery) & (datequery)]



def get_new_results_dataframe(file:str,event_df:pd.DataFrame):
    """
    Pull the Qualifying Results for the events in the event_df DataFrame. It returns a DataFrame matching the Ergast Results DataFrame for those
    events.

    Args:
        file (str): file path to the Ergast cleaned csv file    
        event_df (pd.DataFrame): The event DataFrame of the events which are not in the Ergast DataFrame

    Returns:
        pd.DataFrame: The qualifying results for the races in the event DataFrame
    """
    df = prepare_results_dataframe(file)
    df = clean_quali_times(df)
    driver_mapper={
    'Charles Leclerc': 'leclerc',
    'Sergio Perez': 'perez',
    'Lewis Hamilton': 'hamilton',
    'Carlos Sainz': 'sainz',
    'Fernando Alonso': 'alonso',
    'Lando Norris': 'norris',
    'Pierre Gasly': 'gasly',
    'Max Verstappen': 'max_verstappen',
    'Kevin Magnussen':'kevin_magnussen',
    'Yuki Tsunoda': 'tsunoda',
    'George Russell': 'russell',
    'Lance Stroll': 'stroll',
    'Mick Schumacher': 'mick_schumacher',
    'Sebastian Vettel': 'vettel',
    'Guanyu Zhou': 'zhou',
    'Valtteri Bottas':'bottas',
    'Daniel Ricciardo': 'ricciardo',
    'Esteban Ocon': 'ocon',
    'Alexander Albon': 'albon',
    'Nicholas Latifi': 'latifi'
    }
    constructor_mapper={
    'Ferrari':'ferrari',
    'Red Bull Racing':'red_bull',
    'AlphaTauri':'alphatauri',
    'Mercedes':'mercedes',
    'Haas F1 Team':'haas',
    'Aston Martin':'aston_martin',
    'Alfa Romeo':'alfa',
    'McLaren':'mclaren',
    'Alpine':'alpine',
    'Williams':'williams'

    }
    raceid=df['raceId'].max()+1
    results_dataframes=[]
    for i, row in event_df.iterrows():
        event=fastf1.get_event(2022,row['RoundNumber'])
        session = event.get_qualifying()
        session.load()
        results_df = session.results
        results_df['driverRef']= results_df['FullName'].map(driver_mapper)
        results_df['circuitRef']=row['Location']
        results_df['q1_milliseconds'] = results_df['Q1'].dt.total_seconds()*1000
        results_df['q2_milliseconds'] = results_df['Q2'].dt.total_seconds()*1000
        results_df['q3_milliseconds'] = results_df['Q3'].dt.total_seconds()*1000
        results_df=clean_q3_times(results_df,'q1_milliseconds','q2_milliseconds','q3_milliseconds','fastest_lap_milliseconds')
        results_df['fastest_all_sessions_milliseconds'] = results_df.apply(lambda x: find_fastest_lap(x['q1_milliseconds'],x['q2_milliseconds'],x['q3_milliseconds']),axis=1)
        results_df['constructorRef'] = results_df['TeamName'].map(constructor_mapper)
        new_races_df=results_df[['driverRef','constructorRef','circuitRef','fastest_all_sessions_milliseconds','fastest_lap_milliseconds']].copy()
        new_races_df['quali_position'] = results_df['Position'].copy()
        new_races_df['number']=results_df['DriverNumber'].astype('float').copy()
        query = df['circuitRef']==new_races_df['circuitRef'].unique()[0]
        new_races_df['country']=df.loc[query,'country'].unique()[0]
        new_races_df['name']=df.loc[query,'name'].unique()[0]
        new_races_df['lat_x']=df.loc[query,'lat_x'].unique()[0]
        new_races_df['lng_x']=df.loc[query,'lng_x'].unique()[0]
        new_races_df['alt']=df.loc[query,'alt'].unique()[0]
        new_races_df['year']=event_df['quali_date'].astype('datetime64').dt.year.unique()[0]
        drivers = new_races_df['driverRef'].unique()
        for driver in drivers:
            query2= df['driverRef']==driver
            driver_query=new_races_df['driverRef']==driver
            new_races_df.loc[driver_query,'nationality_drivers']=df.loc[query2,'nationality_drivers'].unique()[0]
            new_races_df.loc[driver_query,'dob']=df.loc[query2,'dob'].unique()[0]
        constructors= new_races_df['constructorRef'].unique()
        for cons in constructors:
            query3=df['constructorRef']==cons
            query4=new_races_df['constructorRef']==cons
            new_races_df.loc[query4,'nationality_constructors']=df.loc[query3,'nationality_constructors'].unique()[0]

        new_races_df['raceId']=raceid
        results_dataframes.append(new_races_df)
        raceid+=1

    new_results_df=pd.concat(results_dataframes)
    new_results_df.reset_index(drop=True,inplace=True)
    return new_results_df
#%%         

def pull_new_races_aggregate_telemetry(df:pd.DataFrame,fast_time='ergast'):
    """
    Function to download and create aggregate features for all the races which are not included for this year in the Ergast results dataframe.

    The function uses the qualifying results dataframe for the get_new_function_results_dataframe function and gets the telemetry data for each
    driver and runs thethe feature aggregations 

    Args:
        df (pd.DataFrame): The qualifying results for the races not included in Ergast DataFrame
        fast_time (str, optional): Which fast time to use if "Ergast" uses the fastest lap time from the Ergast DataFrame. Defaults to 'ergast'.

    Returns:
        pd.DataFrame: The complete driver race record DataFrame with aggregated features for the new races
    """
     
    lap_dfs=[]
    indexes = df.groupby(['year','name'])['raceId'].unique().index
    for year,name in indexes:        
        data = get_fastf1_session(year,name)
        print(f'Loaded data for {year} {name} ')
        df_race=df[(df['year'] == year) & (df['name'] == name)].copy()
        for index, row in df_race.iterrows():           
            lap_df= session_lap_driver(data,int(row['number']))
            
            if len(lap_df) >0 :# no lap data available for the driver pass
                
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
            
            else:
                pass
            print(f"completed driver {row['number']} ", end='\r')
    print(f'completed {year} {name}')
    Lap_aggr_combined=pd.concat(lap_dfs)
    
    return Lap_aggr_combined