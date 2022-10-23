#!/usr/bin/env python3
import numpy as np
import math
import pandas as pd
from scipy.interpolate import interp1d
import os


def engineer_target_bin(df):
    df['quali_position_binned'] = pd.cut(df['quali_position'],bins=4,labels=['0_to_5','5_to_10','eleven_to_15','fifteen_&_above'])
    races = df['raceId'].unique()
    for race in races:
        query = (df['raceId'] == race)
        fastesttime= df.loc[(query),'fastest_lap_milliseconds'].min()
        
        df.loc[query,'lap_timedelta_milliseconds']=(np.where(df.loc[query,'quali_position']==1,0,(fastesttime-df.loc[query,'fastest_lap_milliseconds'])))
    df['lap_timedelta_seconds_binned'] = pd.qcut(df['lap_timedelta_milliseconds'],q=4,labels=['>2.4s','2.4-1.6s','1.6-0.9s','0.9-0s'])
    return df

def convert_object_to_float(df):

    cat_list = list(df.select_dtypes("object").columns)
    true_cat=['driverRef', 'name', 'circuitRef', 'country',
        'nationality_drivers', 'constructorRef', 'nationality_constructors',
        'dob','fl_tyre']
    convert_list=[cat for cat in cat_list if cat not in true_cat]
    for col in convert_list:
        df[col]=df[col].astype(float)
    return df

def feature_engineer_country(df):

    df['driver_home_race']=np.where(df['country'] == df['nationality_drivers'],1,0)
    df['constructor_home_race']= np.where(df['country'] == df['nationality_constructors'],1,0)

    df.drop(columns=['country','nationality_drivers','nationality_constructors'],axis=1, inplace=True)

    return df


def clean_df(df,columns=['number','name','dob']):

    df.drop(columns=columns,axis=1, inplace=True)

    return df

def prepare_modelling_df(df):
    df = engineer_target_bin(df)
    df = convert_object_to_float(df)
    df = feature_engineer_country(df)
    df=clean_df(df)
    r = np.random.RandomState(13)
    rand_races = r.randint(0,16,10)
    years=[2018,2019,2020,2021,2022]
    test_r=[]
    b= 0
    for i,year in enumerate(years):
        races = df.loc[df['year']==year,'raceId'].unique()
        test_r.append(list(races[rand_races[b:b+2]]))    
        b=b+2
    test_r =list(np.array(test_r).flatten())
    query = (df['raceId'].isin(test_r))
    X_test = df[query].drop(columns=['fastest_lap_milliseconds', 'lap_timedelta_milliseconds','quali_position','raceId','quali_position_binned','lap_timedelta_seconds_binned']).copy()
    X_train = df[~query].drop(columns=['fastest_lap_milliseconds', 'lap_timedelta_milliseconds','quali_position','raceId','quali_position_binned','lap_timedelta_seconds_binned']).copy()
    y_test=df.loc[query,['lap_timedelta_seconds_binned','quali_position_binned','fastest_lap_milliseconds','quali_position']].copy()
    y_train=df.loc[~query,['lap_timedelta_seconds_binned','quali_position_binned','fastest_lap_milliseconds','quali_position']].copy()
    
    return X_test, X_train, y_test, y_train


def column_lists(x):
    df=x.copy()    
    hot_encode_col=list(df.select_dtypes("object").columns)
    scale_cols = [col for col in df.columns if col not in hot_encode_col]
    return scale_cols, hot_encode_col


def apply_manual_features(X_train,X_test,features=[]):
    features2=[feature.split('__')[1] for feature in features]


    if len(features2) != 0:
        X_train_manual= X_train[features2]
        X_test_manual = X_test[features2]
    
    return X_train_manual, X_test_manual

