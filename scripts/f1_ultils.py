import numpy as np
import math
import pandas as pd
from scipy.interpolate import interp1d
def convert_time_miliseconds(x):
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

    speeds=[]
    speeds.append(x)
    speeds.append(y)
    speeds.append(z)

    
    return min(speeds)


def fill_mean_parameter(df,parameter):

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


def flag_corners(df,curvature=0.0005,interval=10,smoothing=10):

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


def corner_curvature(df,interval=10,smoothing=10):

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




def straight_lengths(df):
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


def corners(x):    
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


def speed_data(df):


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


def acceleration(x,distance=100):
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


def rpm(df):
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


def gear_data(df):

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


def driver_brake(x):
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


def driver_corners(x):
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


def DRS_open(df):

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