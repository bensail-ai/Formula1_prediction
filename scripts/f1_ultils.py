#!/usr/bin/env python3
import numpy as np
import math
import pandas as pd
from scipy.interpolate import interp1d
import os

def read_ergast_files(directory:str):
    """Read all the .csv files from a particular folder save the results in a dictionary object

    Args:
        directory (str): directory of the folder of csv's to read

    Returns:
        dict: Dictionary of pandas DataFrames of all the csv files in a folder
    """
    f1_data={}
    for file in os.listdir(directory):
        if file.endswith('.csv'):            
            f1_data[f"df_{file.split('.')[0]}"]=pd.read_csv(os.path.join(directory,file))
            
    return f1_data



def convert_time_miliseconds(x):
    """
    Converts a string with time data formated 00:00 of minutes:seconds to total seconds

    Args:
        x (str): string of time data MM:SS

    Returns:
        float: total seconds of string
    """
    if x == 'nan':
        return None
    else:    
        if len(x.split(':')) > 2:
            secs = float(x.split(':')[0])*60*60*1000 + float(x.split(':')[1])*60*1000+ float(x.split(':')[2])*1000
        elif len(x.split(':')) > 1:
            secs = float(x.split(':')[0])*60*1000 + float(x.split(':')[1])*1000
        elif len(x.split(':')) == 1:
            secs= float(x.split(':')[0])*1000
        return round(secs,0)

def find_fastest_lap(x,y,z):
    """Find the fastest time from 3 individual floats

    Args:
        x (float): float time seconds
        y (float): float time seconds
        z (float): float time seconds

    Returns:
        float: minimum of x,y,z floats
    """
    speeds=[]
    speeds.append(x)
    speeds.append(y)
    speeds.append(z)

    
    return min(speeds)



def clean_q3_times(df:pd.DataFrame, q1:str,q2:str,q3:str,col_out:str):
    """
    Clean the qualifying 3 column of Ergast Database, 
    when there are null values in the q3 column and q2 column
    it copies the q1 results into q3 column

    when there are null values in the q3 column only it coppies the 
    q2 reults into q3 column

    Args:
        df (pd.DataFrame): Ergast combined result DataFrame
        q1 (str): column name of q1
        q2 (str): column name of q2
        q3 (str): column name of q3
        col_out (str): column name to save the combined qualifying times

    Returns:
        pd.DataFrane: Ergast Dataframe withe combined qualifying times
    """
    df[col_out]=np.NaN
    df.loc[(df[q2].isna()) & (df[q3].isna()),col_out]=df.loc[(df[q2].isna()) & (df[q3].isna()),q1]
    df.loc[(df[q3].isna()) & (~df[q2].isna()),col_out]=df.loc[(df[q3].isna())& (~df[q2].isna()),q2]
    df.loc[(~df[q3].isna()) & (~df[q2].isna()),col_out]=df.loc[(~df[q3].isna())& (~df[q2].isna()),q3]
    return df


def fill_mean_parameter(df:pd.DataFrame,parameter:str):
    """This helper function fills a parameter based on the mean from qualifying stint

    Requires a combined F1Fast telemetry and Ergast dataframe with lap data and qualifying stint columns

    Args:
        df (pd.DataFrame): combined Ergast and F1Fast telemetry lap dataframe
        parameter (str): Parameter column name to find mean of

    Returns:
        pd.DataFrame: combined dataframe withe parameter nans filled in with a mean value
    """
    indexes = df.loc[df[parameter].isna()].groupby(['qualifying_lapId',
    'raceId','qualifying_DriverNumber'])['qualifying_Stint'].unique().index
    values = df.loc[df[parameter].isna()].groupby(['qualifying_lapId',
    'raceId','qualifying_DriverNumber'])['qualifying_Stint'].unique().values
    for i,ind in enumerate(indexes):
        for val in values[i]:
            query = (df['qualifying_Stint']==val)&(df['raceId']==ind[1]) & (df['qualifying_DriverNumber'] ==ind[2])
            mean_para = df.loc[query,[parameter]].mean()
            query2= (df['qualifying_lapId']==ind[0])&(df['raceId']==ind[1]) & (df['qualifying_DriverNumber'] ==ind[2])
            df.loc[query2,[parameter]] = mean_para.values
    return df


def fill_mean_parameter_fastf1(df: pd.DataFrame,parameter:str):
    """This helper function fills a parameter based on the mean from qualifying stint

    Requires a F1Fast telemetry dataframe  with lap data and qualifying stint columns

    Args:
        df (pd.DataFrame): F1Fast telemetry lap dataframe
        parameter (str): Parameter column name to find mean of

    Returns:
        pd.DataFrame: combined dataframe withe parameter nans filled in with a mean value
    """
    indexes = df.loc[df[parameter].isna()].groupby(['qualifying_Stint'])['qualifying_lapId'].unique().index
    values = df.loc[df[parameter].isna()].groupby(['qualifying_Stint'])['qualifying_lapId'].unique().values
    for i,ind in enumerate(indexes):
        for val in values[i]:
            query = (df['qualifying_Stint']==ind)
            mean_para = df.loc[query,[parameter]].mean()
            query2= (df['qualifying_lapId']==val)& (df['qualifying_Stint'] == ind)
            df.loc[query2,[parameter]] = mean_para.values
    return df
   


def flag_corners(df:pd.DataFrame,curvature=0.0005,interval=10,smoothing=10):
    """
    This function flags the corners of FastF1 Telemetry lap data by creating a column 'corner' with 
    flags of 'corner' or 'straight'

    It does it by finding the curvature of the second order derivative 
    abs(dx * d2y - d2x * dy) / (dx * dx + dy * dy) ** 1.5

    where curvature is greater than thershold curvature paramater flags this as a corner
    Args:
        df (pd.DataFrame): FastF1 lap telemetry data
        curvature (float, optional): curvature threshold . Defaults to 0.0005.
        interval (int, optional): interval to resample x and y data by. Defaults to 10.
        smoothing (int, optional): rolling average to apply to curvature measurements. Defaults to 10.

    Returns:
        pd.DataFrame: FastF1 telemetry data with corners flagged 
    """
    dist_new = np.arange(df['qualifying_Distance'].min(),df['qualifying_Distance'].max(),interval)
    f = interp1d(df['qualifying_Distance'], df['qualifying_X'])
    x_new = f(dist_new)
    f = interp1d(df['qualifying_Distance'], df['qualifying_Y'])
    y_new=f(dist_new)
    #interpolated x and y dataframe
    temp=pd.DataFrame({'distance':dist_new,'x':x_new,'y':y_new})
    
    # first derivatives
    temp['dx'] = np.gradient(temp['x'])
    temp['dy'] = np.gradient(temp['y'])

    # second derivatives
    temp['d2x'] = np.gradient(temp.dx)
    temp['d2y'] = np.gradient(temp.dy)

    # calculation of curvature from the typical formula - reference stackoverflow https://stackoverflow.com/questions/50562254/curvature-of-set-of-points
    temp['curvature'] = temp.eval('abs(dx * d2y - d2x * dy) / (dx * dx + dy * dy) ** 1.5')
    temp['curvature'] = temp.curvature.rolling(smoothing,min_periods=1, center=True).mean()
    temp.dropna(inplace=True)
    #find corners
    temp['corner']= np.where(temp['curvature']>curvature,'corner','straight')
    # find distance in original dataframe
    df['corner'] = df['qualifying_Distance'].apply(lambda x: temp.loc[temp['distance'].sub(x).abs().idxmin(),'corner'])

    return df


def corner_curvature(df: pd.DataFrame,interval=10,smoothing=10):
    """
    Calculates the curvature of the lap x and y data

    Args:
        df (pd.DataFrame): FastF1 lap telemetry data
        
        interval (int, optional): interval to resample x and y data by. Defaults to 10.
        smoothing (int, optional): rolling average to apply to curvature measurements. Defaults to 10.

    Returns:
        list: list of curvature values (floats)
    """
    dist_new = np.arange(df['qualifying_Distance'].min(),df['qualifying_Distance'].max(),interval)
    f = interp1d(df['qualifying_Distance'], df['qualifying_X'])
    x_new = f(dist_new)
    f = interp1d(df['qualifying_Distance'], df['qualifying_Y'])
    y_new=f(dist_new)
    #interpolated x and y dataframe
    temp=pd.DataFrame({'distance':dist_new,'x':x_new,'y':y_new})
    
    # first derivatives
    temp['dx'] = np.gradient(temp['x'])
    temp['dy'] = np.gradient(temp['y'])

    # second derivatives
    temp['d2x'] = np.gradient(temp.dx)
    temp['d2y'] = np.gradient(temp.dy)

    # calculation of curvature from the typical formula - reference stackoverflow https://stackoverflow.com/questions/50562254/curvature-of-set-of-points
    temp['curvature'] = temp.eval('abs(dx * d2y - d2x * dy) / (dx * dx + dy * dy) ** 1.5')
    temp['curvature'] = temp.curvature.rolling(smoothing,min_periods=1, center=True).mean()
    temp.dropna(inplace=True)

    return temp['curvature'].to_list()




def straight_lengths(df:pd.DataFrame):
    """
    Feature aggregation function 
    Returns the straight lengths from the FastF1 telemetry data

    Args:
        df (pd.DataFrame): Fastf1 Lap telemetry data for a single lap

    Returns:
        total_straight (float) : total summed straight lengths in (m) of a lap
        mean_straight (float) : mean straight lengths in (m) of a lap
        max_straight (float) : max straight lengths in (m) of a lap
    """
    df = df[df['corner']=='straight'].copy()
    indices = df.index.to_series()
    df['Group'] = ((indices - indices.shift(1)) != 1).cumsum()
    straights = df['Group'].unique()   
    St_lengths=[]
    for straight in straights:        
        min_straight = df.loc[df['Group'] == straight,'qualifying_Distance'].min()
        max_straight = df.loc[df['Group'] == straight,'qualifying_Distance'].max()
        St_lengths.append(max_straight-min_straight)
    
    total_straight= np.sum(St_lengths)
    mean_straight = np.mean(St_lengths)
    max_straight = np.max(St_lengths)

    return total_straight, mean_straight, max_straight


def corners(x:pd.DataFrame):  
    """
    Feature aggregation function 
    Returns the corner curvature values from the FastF1 telemetry data

    Args:
        df (pd.DataFrame): Fastf1 Lap telemetry data for a single lap

    Returns:
        total_corner (float) : total summed corner lengths in (m) of a lap
        mean_corner (float) : mean corner lengths in (m) of a lap
        max_corner (float) : max corner lengths in (m) of a lap
        total_curvature (float) : total summed corner curvature of a lap
        mean_curvature (float) : mean corner curvature of a lap
        max_curvature (float) : max corner curvature of a lap
        std_curvature (float) : std of corner curvature of a lap
        num_corners (float) : number of corners in a lap
    """  
    df = x[x['corner']=='corner'].copy()    
    indices = df.index.to_series()
    df['Group'] = ((indices - indices.shift(1)) != 1).cumsum()
    corners = df['Group'].unique() 
    num_corners=len(df['Group'].unique() )  
    corner_lengths=[]
    corner_tightness=[]
    for corner in corners:
        if len(df[df['Group'] == corner]) > 5:             
            min_corner = df.loc[df['Group'] == corner,'qualifying_Distance'].min()
            max_corner = df.loc[df['Group'] == corner,'qualifying_Distance'].max()        
            corner_lengths.append(max_corner-min_corner)
            corner_tightness.append(np.mean(corner_curvature(df.loc[df['Group']==corner,['qualifying_Distance','qualifying_X','qualifying_Y']].copy())))
        
    total_corner= np.sum(corner_lengths)
    mean_corner = np.mean(corner_lengths)
    max_corner = np.max(corner_lengths)
    total_curvature= np.sum(corner_tightness)
    mean_curvature = np.mean(corner_tightness)
    max_curvature = np.max(corner_tightness)
    std_curvature=np.std(corner_tightness)
    return total_corner,mean_corner,max_corner,total_curvature,mean_curvature,max_curvature,std_curvature,num_corners


def speed_data(df:pd.DataFrame):
    """

    Feature aggregation function for FastF1 lap telemetry data

    returns the speed features 

    Args:
        df (pd.DataFrame): FastF1 telemetry lap data for a single lap

    Returns:
        max_speed (float): max speed of that lap
        std_speed (float): variance in speed of that lap      
        min_speed (float): minimum speed of that lap
        mean_straight_speed (float): mean speed on the straight of that lap
        var_straight_speed (float): variance in speed on the straight of that lap
    """

    max_speed=df['qualifying_Speed'].max()
    std_speed = df['qualifying_Speed'].std()
    min_speed = df['qualifying_Speed'].min()
    straight_top_speed=[]

    straight = df[df['corner']=='straight'].copy()
    indices = straight.index.to_series()
    straight['Group'] = ((indices - indices.shift(1)) != 1).cumsum()
    straights = straight['Group'].unique()
    for stra in straights:
        straight_top_speed.append(straight.loc[straight['Group']==stra,'qualifying_Speed'].max())
    mean_straight_speed=np.mean(straight_top_speed)
    var_straight_speed=np.std(straight_top_speed) 

    return max_speed,std_speed,min_speed,mean_straight_speed,var_straight_speed

def throttle_bin(df:pd.DataFrame):
    """
    Feature aggregation function for FastF1 lap telemetry data

    Returns the Throrrle bin flag, 
    'off_full_thottle' when less than 98% # some laps had maximum 99 not 100
    ' full_throttle' 98% and greater

    Args:
        df (pd.DataFrame): FastF1 telemetry lap data
    Returns:
        pd.Dataframe: FastF1 telemetry lap data with Throttle bin
    """
    df['Throttle_bin'] =np.where(df['qualifying_Throttle'] < 98,'off_full_throttle','full_throttle')

    return df

def acceleration(x:pd.DataFrame,distance=100):
    """
    Returns the values of acceleration out of corners around the lap of a fastf1 dataframe

    Function takes each period the driver is on Full Throttle and returns the accleration 
    for the period Distance when the driver is on Full Throttle

    Args:
        x (pd.DataFrame): Fastf1 dataframe of lap telemetry data
        distance (int, optional): input integer of distance. Defaults to 100.

    Returns:
        list: List of accleration values (floats)
    """
    x= throttle_bin(x)
    df = x[x['Throttle_bin']=='full_throttle'].copy()    
    indices = df.index.to_series()
    df['Group'] = ((indices - indices.shift(1)) != 1).cumsum()
    throttle_group = df['Group'].unique()
    acc_=[]
    for throttle in throttle_group:
        start_speed = df.loc[df[df['Group']==throttle].first_valid_index(),'qualifying_Speed']
        start_distance = df.loc[df[df['Group']==throttle].first_valid_index(),'qualifying_Distance']
        start_time = df.loc[df[df['Group']==throttle].first_valid_index(),'qualifying_lap_timedelta']
        df.loc[df['Group']==throttle,'distance_cumsum'] = (df.loc[df['Group']==throttle,'qualifying_Distance']- start_distance).cumsum()
        if ((df['Group']==throttle) & (df['distance_cumsum'] >distance)).sum() != 0:
            end_time = df.loc[df[(df['Group']==throttle) & (df['distance_cumsum'] >distance)].first_valid_index() ,'qualifying_lap_timedelta']
            end_speed = df.loc[df[(df['Group']==throttle) & (df['distance_cumsum'] >distance)].first_valid_index() ,'qualifying_Speed']
            acc_.append((end_speed-start_speed)/(end_time-start_time))
    if len(acc_) > 1:
        acc_.pop(0)
    return acc_


def rpm(df:pd.DataFrame):
    """
    Feature aggregation function for FastF1 telemetry data

    Calculates the features of RPM

    Args:
        df (pd.DataFrame): FastF1 telemetry data for single lap

    Returns:
        max_rpm (float): maximum RPM on the whole lap
        mean_straight_rpm (float): mean straight RPM
        var_straight_rpm (float): variance of RPM on the straight
    """
    max_rpm=df['qualifying_RPM'].max()
  
    straight_rpm=[]
 

    straight = df[df['corner']=='straight'].copy()
    indices = straight.index.to_series()
    straight['Group'] = ((indices - indices.shift(1)) != 1).cumsum()
    straights = straight['Group'].unique()
    for stra in straights:
        straight_rpm.append(straight.loc[straight['Group']==stra,'qualifying_RPM'].values)
    straight_rpm = [j for i in straight_rpm for j in i]
    
    mean_straight_rpm=np.mean(straight_rpm)
    var_straight_rpm=np.std(straight_rpm) 

    return max_rpm,mean_straight_rpm, var_straight_rpm


def gear_data(df:pd.DataFrame):
    """
    Feature aggregation function for FastF1 telemetry data

    Calculates the time spent in each gear 

    returns this as a dictionary of times per gear
    Args:
        df (pd.DataFrame): FastF1 telemetry data for single lap

    Returns:
        dict: times as floats for each gear, if gear is not on a lap returns 0
    """
    gears = df['qualifying_nGear'].unique()
    gear_dict={
        'gear_0': 0,
        'gear_1': 0,
        'gear_2': 0,
        'gear_3': 0,
        'gear_4': 0,
        'gear_5': 0,
        'gear_6': 0,
        'gear_7': 0,
        'gear_8': 0,
    }
    for gear in gears:
        
        gear_df = df[df['qualifying_nGear']==gear].copy()    
        indices = gear_df.index.to_series()
        gear_df['Group'] = ((indices - indices.shift(1)) != 1).cumsum()
        gear_group = gear_df['Group'].unique()
        total_time=[]
        for gr in gear_group:
            start_time = gear_df.loc[gear_df[gear_df['Group']==gr].first_valid_index(),'qualifying_lap_timedelta']
            end_time = gear_df.loc[gear_df[gear_df['Group']==gr].last_valid_index(),'qualifying_lap_timedelta']
            total_time.append(end_time-start_time)
        
        gear_dict[f'gear_{int(gear)}'] = np.sum(total_time)            
   
    return gear_dict


def driver_brake(x: pd.DataFrame):
    """
    Feature aggregation function for FastF1 telemetry data

    Calculates the time and the distance on brakes for a single telemetry lap

    It works calculates the indices of groups of Brake =True

    For each group it calculates the time and dsitance and appends to a list 

    Calculates the sum of the lists for time and distance on brakes

    Args:
        x (pd.DataFrame): FastF1 telemetry lap

    Returns:
        float : total time on brakes for that lap
        float : total distance on brakes for that lap
    """
    df = x[x['qualifying_Brake']==True].copy()
    indices = df.index.to_series()
    df['Group'] = ((indices - indices.shift(1)) != 1).cumsum()
    brakes = df['Group'].unique() 
    time_on_brakes=[]
    distance_on_brakes=[]

    for brake in brakes:
        start_time = df.loc[df[df['Group']==brake].first_valid_index(),'qualifying_lap_timedelta']
        end_time = df.loc[df[df['Group']==brake].last_valid_index(),'qualifying_lap_timedelta']
        time_on_brakes.append(end_time-start_time)
        start_distance = df.loc[df[df['Group']==brake].first_valid_index(),'qualifying_Distance']
        end_distance = df.loc[df[df['Group']==brake].last_valid_index(),'qualifying_Distance']
        distance_on_brakes.append(end_distance-start_distance)
    
    return np.sum(time_on_brakes), np.sum(distance_on_brakes)


def driver_corners(x:pd.DataFrame):
    """
    Feature aggregation function for FastF1 telemetry data

    Calculates the speed through the corners for a fastF1 telemetry lap

    It does this by finding the individual groups of corners from the corner flag

    for each group it finds the speed, and speed for tightest corner using the corner curvature function

    Args:
        x (pd.DataFrame): Fastf1 telemetry lap

    Returns:
        bottom_speed (list): list of minimum speeds in the corners
        max_corner_speed (list): list of maximum speeds in the corners
        bottom_speed_tightness_corner (float): minimum speed in the laps tightest corner
    """
    df = x[x['corner']=='corner'].copy()    
    indices = df.index.to_series()
    df['Group'] = ((indices - indices.shift(1)) != 1).cumsum()
    corners = df['Group'].unique() 
 
    bottom_speed=[]
    max_corner_speed=[]
    max_corner_tightness = 0
    bottom_speed_tightness_corner =0
    for corner in corners:
        if len(df[df['Group'] == corner]) > 5:
            query = (df['Group']==corner)
            bottom_speed.append(np.min(df.loc[query,'qualifying_Speed']))
            max_corner_speed.append(np.max(df.loc[query,'qualifying_Speed']))                 
            corner_tightness = (np.mean(corner_curvature(df.loc[df['Group']==corner,['qualifying_Distance','qualifying_X','qualifying_Y']].copy())))
            if corner_tightness > max_corner_tightness:
                max_corner_tightness=corner_tightness
                bottom_speed_tightness_corner=(np.min(df.loc[query,'qualifying_Speed']))
    return bottom_speed,max_corner_speed,bottom_speed_tightness_corner  


def DRS_open(df:pd.DataFrame):
    """
    Feature aggregation function for FastF1 telemetry data

    Function calculates the total time and the total distance the driver 
    has the DRS open from a Fastf1 telemetry lap dataframe

    Args:
        df (pd.DataFrame):Fastf1 telemetry lap dataframe
    Returns:
        float: Total time DRS open for that lap
        float: Total distance DRS open for that lap
    """
    DRS_df = df[df['qualifying_DRS'].isin([10,12,14])].copy()
    indices = DRS_df.index.to_series()
    DRS_df['Group'] = ((indices - indices.shift(1)) != 1).cumsum()
    drs_sections = DRS_df['Group'].unique()
    total_time=[]
    distance_on_drs=[]
    for section in drs_sections:
        start_time = DRS_df.loc[DRS_df[DRS_df['Group']==section].first_valid_index(),'qualifying_lap_timedelta']
        end_time = DRS_df.loc[DRS_df[DRS_df['Group']==section].last_valid_index(),'qualifying_lap_timedelta']
        total_time.append(end_time-start_time)
        start_distance = DRS_df.loc[DRS_df[DRS_df['Group']==section].first_valid_index(),'qualifying_Distance']
        end_distance = DRS_df.loc[DRS_df[DRS_df['Group']==section].last_valid_index(),'qualifying_Distance']
        distance_on_drs.append(end_distance-start_distance)
    return np.sum(total_time), np.sum(distance_on_drs)  