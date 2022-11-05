import requests
import pandas as pd
import numpy as np
import re

def get_elevation(lat, long):
    query = ('https://api.open-elevation.com/api/v1/lookup'
             f'?locations={lat},{long}')
    r = requests.get(query).json()  # json object
    elevation = r['results'][0]['elevation']

    return  elevation



def convert_string_tonan(df):
    df.replace("\\N",np.NaN,inplace=True)
    return df


def fill_altitude_nans(df,col='alt'):
    df.loc[df[col].isna(),col]=df[df[col].isna()].apply(lambda x: get_elevation(x.lat,x.lng), axis=1)
    return df

def clean_wiki_weather(df,col='race_weather_wiki'):
    df[col] = df[col].str.split('\n| ').str[1:3].apply(lambda x: ' '.join(x))
    return df

def weather_flag(df,col_in,col_out):

    if col_in=='race_weather_wiki':
        df = clean_wiki_weather(df)
        conditions = df[col_in].unique()
        weather_dict={}
        for con in conditions:
            if re.search('wet|rain|damp|pioggia|showers|piovoso|drizzly|snow',con.lower()) != None:
                weather_dict[con] = 'wet'
            else:
                weather_dict[con]= 'dry'
        df[col_out] = df[col_in].map(weather_dict)
    elif col_in=='quali_condition':
        quali_conditions=df[col_in].unique()
        weather_dict={}
        for con in list(pd.Series(quali_conditions).dropna().values):
            if re.search('wet|rain|damp|pioggia|showers|piovoso|drizzly|snow',con.lower()) != None:
                weather_dict[con] = 'wet'
            else:
                weather_dict[con]= 'dry'
        df[col_out] = df[col_in].map(weather_dict)
    else:
        print('wrong column name')
  
    return df


def clean_ergast(f1_data,save = False):
    df_circuits_clean=fill_altitude_nans(f1_data['df_circuits'])
    df_constructors_clean =f1_data['df_constructors'].copy()
    df_constructors_clean.drop(columns='name',axis=0,inplace=True)
    df_constructor_results_clean=f1_data['df_constructor_results'].copy()
    df_constructor_results_clean.drop(columns='status',axis=0,inplace=True)
    df_constructor_standings_clean=f1_data['df_constructor_standings'].copy()
    df_constructor_standings_clean.drop(columns='positionText',axis=0, inplace = True)
    df_drivers_clean=f1_data['df_drivers'].copy()
    df_drivers_clean.drop(columns=['code','forename','surname'],axis=0,inplace=True)
    df_driver_standings_clean=f1_data['df_driver_standings'].copy()
    df_driver_standings_clean.drop(columns='positionText',axis=0,inplace=True)
    df_lap_times_clean=f1_data['df_lap_times'].copy()
    df_lap_times_clean.drop(columns='time',axis=0,inplace=True)
    df_pit_stops_clean=f1_data['df_pit_stops'].copy()
    df_pit_stops_clean.drop(columns='duration',axis=0,inplace=True)
    df_qualifying_clean=f1_data['df_qualifying'].copy()
    url= 'https://www.formula1.com/en/results.html/1995/races/637/australia/qualifying-0.html'
    page = requests.get(url)
    df_q3 = pd.read_html(page.content)[0]

    url= 'https://www.formula1.com/en/results.html/1995/races/637/australia/qualifying-1.html'
    page = requests.get(url)
    df_q1 = pd.read_html(page.content)[0]
    url= 'https://www.formula1.com/en/results.html/1995/races/637/australia/qualifying-2.html'
    page = requests.get(url)
    df_q2 = pd.read_html(page.content)[0]
    # %%
    df_q3.drop(columns=['Unnamed: 0','Unnamed: 7'],axis=0,inplace=True)
    df_q2.drop(columns=['Unnamed: 0','Unnamed: 7'],axis=0,inplace=True)
    df_q1.drop(columns=['Unnamed: 0','Unnamed: 7'],axis=0,inplace=True)
    q1_times = df_qualifying_clean.loc[df_qualifying_clean['raceId']==256,['q1','number']].merge(df_q1[['Time','No']],left_on='number',right_on='No')['Time']
    df_qualifying_clean.loc[df_qualifying_clean['raceId']==256,'q1'] = q1_times.values
    q2_times = df_qualifying_clean[df_qualifying_clean['raceId']==256].merge(df_q2[['Time','No']],left_on='number',right_on='No',how='left')['Time']
    df_qualifying_clean.loc[df_qualifying_clean['raceId']==256,'q2'] = q2_times.values
    q3_times = df_qualifying_clean[df_qualifying_clean['raceId']==256].merge(df_q3[['Time','No']],left_on='number',right_on='No',how='left')['Time']
    df_qualifying_clean.loc[df_qualifying_clean['raceId']==256,'q3'] = q3_times.values
    df_qualifying_clean.drop(columns=['number'],inplace=True)
    df_races_clean = f1_data['df_races'].copy()
    df_races_weather_clean = f1_data['df_races_weather_all'].copy()
    cols=['raceId','fp1_date', 'fp1_time', 'fp2_date', 'fp2_time', 'fp3_date', 'fp3_time']
    df_races_weather_clean= df_races_weather_clean.merge(df_races_clean[cols],left_on='raceId',right_on='raceId')
    df_races_weather_clean = weather_flag(df_races_weather_clean,col_in='race_weather_wiki',col_out='race_condition_wiki')
    df_races_weather_clean = weather_flag(df_races_weather_clean,col_in='quali_condition',col_out='quali_condition_clean')
    df_races_weather_clean.drop(columns=['race_weather_wiki','race_condition','quali_condition', 'Unnamed: 0'],axis=0,inplace=True)
    df_results_clean=f1_data['df_results'].copy()
    df_results_clean.drop(columns=['number','position','time'],axis=0,inplace=True)
    df_seasons_clean=f1_data['df_seasons'].copy()
    df_sprint_results_clean=f1_data['df_sprint_results'].copy()
    df_sprint_results_clean.drop(columns=['position','time','number'],inplace=True)
    df_status_clean=f1_data['df_status'].copy()
    df_combined_clean = df_races_weather_clean.merge(df_circuits_clean,left_on='circuitId',right_on='circuitId')
    df_combined_clean.drop(columns=['url_x','lat_y','lng_y','url_y'],axis=0,inplace=True)
    df_qualifying_clean.rename(columns={'position':'quali_position'},inplace=True)
    df_combined_clean= df_combined_clean.merge(df_qualifying_clean,left_on='raceId',right_on='raceId')
    df_sprint_results_clean= df_sprint_results_clean.merge(df_status_clean,left_on='statusId',right_on='statusId')
    df_sprint_results_clean.drop(columns='statusId',axis=0,inplace=True)
    mapper_cols = {'grid':'grid_sprint','positionText':'positionText_sprint', 'positionOrder': 'positionOrder_sprint', 'points':'points_sprint', 'laps': 'laps_sprint','milliseconds' :'milliseconds_sprint', 'fastestLap' :'fastestLap_sprint', 'fastestLapTime': 'fastestLapTime_sprint','status':'status_sprint'}
    df_sprint_results_clean.rename(columns= mapper_cols,inplace=True)
    df_combined_clean= df_combined_clean.merge(df_sprint_results_clean,on=['raceId','driverId','constructorId'], how='left')
    df_results_clean= df_results_clean.merge(df_status_clean,left_on='statusId',right_on='statusId')
    df_results_clean.drop(columns='statusId',axis=0,inplace=True)
    df_combined_clean= df_combined_clean.merge(df_results_clean,on=['raceId','driverId','constructorId'], how='left')
    mapper_drivers = {'points':'points_drivers', 'position':'position_drivers',
       'wins':'wins_drivers'}
    df_driver_standings_clean.rename(columns= mapper_drivers,inplace=True)
    df_combined_clean= df_combined_clean.merge(df_driver_standings_clean,on=['raceId','driverId'], how='left')
    mapper_constructors = {'points':'points_constructors', 'position':'position_constructors',
       'wins':'wins_constructors'}
    df_constructor_standings_clean.rename(columns= mapper_constructors,inplace=True)
    df_combined_clean=df_combined_clean.merge(df_constructor_standings_clean,on=['raceId','constructorId'], how='left')
    mapper_cons_results={'points':'points_constructor_ind_races'}
    df_constructor_results_clean.rename(columns= mapper_cons_results,inplace=True)
    df_combined_clean=df_combined_clean.merge(df_constructor_results_clean,on=['raceId','constructorId'], how='left')
    df_drivers_clean_tomerge=df_drivers_clean[['driverId', 'driverRef', 'number', 'dob', 'nationality']].copy()
    mapper_drivers_id={'nationality':'nationality_drivers'}
    df_drivers_clean_tomerge.rename(columns= mapper_drivers_id,inplace=True)
    df_combined_clean=df_combined_clean.merge(df_drivers_clean_tomerge,on=['driverId'], how='left')
    mapper_cons_id={'nationality':'nationality_constructors'}
    df_constructors_clean_tomerge=df_constructors_clean[['constructorId', 'constructorRef', 'nationality']].copy()
    df_constructors_clean_tomerge.rename(columns= mapper_cons_id,inplace=True)
    df_combined_clean=df_combined_clean.merge(df_constructors_clean_tomerge,on=['constructorId'], how='left')
    nationality_to_nation ={'British':'UK', 'Brazilian':'Brazil', 'German':'Germany', 
    'Polish' : 'Poland', 'Italian' : 'Italy', 'Finnish':'Finland',
    'Australian':'Australia', 'Spanish':'Spain', 'Japanese':'Japan', 'Swiss':'Switzerland',
    'French':'France', 'Austrian':'Austria', 'American':'USA', 'Dutch':'Netherlands',
    'Colombian':'Columbia', 'Canadian':'Canada', 'Portuguese':'Portugal',
    'Indian':'India', 'Hungarian':'Hungary', 'Irish':'Ireland', 'Argentine':'Argentina', 'Danish':'Denmark', 
    'Russian':'Russia','Mexican':'Mexico', 'Venezuelan':'Venezuela', 'Belgian':'Belgium', 'Swedish':'Sweden', 
    'Indonesian':'Indonesia','New Zealander': 'New Zealand', 'Monegasque':'Monaco', 'Thai':'Thailand',
    'Chinese': 'China', 'Malaysian':'Malaysia','Czech':'Czech Republic'}
    df_combined_clean['nationality_constructors']=df_combined_clean.nationality_constructors.map(nationality_to_nation)
    df_combined_clean['nationality_drivers']=df_combined_clean.nationality_drivers.map(nationality_to_nation)
    if save == True:
        df_combined_clean.to_csv('./data/clean/combined_ergast_clean.csv',index=False)
        #df_lap_times_clean.to_csv('../data/clean/df_lap_times_clean.csv', index=False) # do not need to keep race lap times as not used in further analysis 
        #df_pit_stops_clean.to_csv('../data/clean/df_pit_stops_clean.csv',index=False) # do not need  to keep pit stops as not used in further analysis
        return None
    else:
        return df_combined_clean









