#!/usr/bin/env python3
#%%
import math
import os
import re
from sklearn.base import BaseEstimator, TransformerMixin, _OneToOneFeatureMixin
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def engineer_target_bin(df,fast_lap='fastest_all_sessions_milliseconds'):    
    df['quali_position_binned'] = pd.cut(df['quali_position'],bins=4,labels=['0_to_5','5_to_10','eleven_to_15','fifteen_&_above'])
    races = df['raceId'].unique()
    for race in races:
        query = (df['raceId'] == race)
        fastesttime= df.loc[(query),fast_lap].min()
        
        df.loc[query,'lap_timedelta_milliseconds']=(np.where(df.loc[query,'quali_position']==1,0,(fastesttime-df.loc[query,fast_lap])))
    df['lap_timedelta_seconds_binned'] = pd.qcut(df['lap_timedelta_milliseconds'],q=4,labels=['>2.3s','2.3-1.6s','1.6-0.9s','0.9-0s'])
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

def prepare_modelling_df(df,fast_lap='all',test_raceids=None, test_split=True):
    df = convert_object_to_float(df)
    if fast_lap=='all':
        df = engineer_target_bin(df,fast_lap='fastest_all_sessions_milliseconds')
    else:
        df = engineer_target_bin(df,fast_lap='fastest_lap_milliseconds')
    
    df = feature_engineer_country(df)
    df=clean_df(df)
    df= fix_zero_DRS(df)
    if test_split == True:
        if test_raceids == None:
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
        else:
            test_r=test_raceids
        query = (df['raceId'].isin(test_r))
        X_test = df[query].drop(columns=['fastest_lap_milliseconds', 'lap_timedelta_milliseconds','quali_position','raceId','quali_position_binned','lap_timedelta_seconds_binned','fastest_all_sessions_milliseconds']).copy()
        X_train = df[~query].drop(columns=['fastest_lap_milliseconds', 'lap_timedelta_milliseconds','quali_position','raceId','quali_position_binned','lap_timedelta_seconds_binned','fastest_all_sessions_milliseconds']).copy()
        y_test=df.loc[query,['lap_timedelta_seconds_binned','quali_position_binned','lap_timedelta_milliseconds','quali_position']].copy()
        y_train=df.loc[~query,['lap_timedelta_seconds_binned','quali_position_binned','lap_timedelta_milliseconds','quali_position']].copy()
        
        return X_test, X_train, y_test, y_train
    else:
        X=df.drop(columns=['fastest_lap_milliseconds', 'lap_timedelta_milliseconds','quali_position','raceId','quali_position_binned','lap_timedelta_seconds_binned','fastest_all_sessions_milliseconds']).copy()
        y=df[['lap_timedelta_seconds_binned','quali_position_binned','lap_timedelta_milliseconds','quali_position']].copy()
        return X, y


def column_lists(x):
    df=x.copy()    
    hot_encode_col=list(df.select_dtypes("object").columns)
    scale_cols = [col for col in df.columns if col not in hot_encode_col]
    return scale_cols, hot_encode_col


def apply_manual_features(X_train,X_test,features=[]):
    features2=[]

    for feature in features:
        if len(feature.split('__')) >1:            
            m = re.search('driverRef*|constructorRef*|circuitRef*',feature)
            if m !=None:
                features2.append(m.group())
            else:
                features2.append(feature.split('__')[1])
        else:
            pass
    features2 = list(dict.fromkeys(features2))

    if len(features2) != 0:
        X_train_manual= X_train[features2]
        X_test_manual = X_test[features2]
    
    return X_train_manual, X_test_manual

def apply_manual_features_X(X,features=[]):
    features2=[]

    for feature in features:
        if len(feature.split('__')) >1:            
            m = re.search('driverRef*|constructorRef*|circuitRef*',feature)
            if m !=None:
                features2.append(m.group())
            else:
                features2.append(feature.split('__')[1])
        else:
            pass
    features2 = list(dict.fromkeys(features2))

    if len(features2) != 0:
        X_manual= X[features2]
        
    
    return X_manual

def fix_zero_DRS(df):
    df2 = df.copy()
    query = df['avg_lap_time_on_DRS'] == 0
    zero_drs= df.loc[query, ['circuitRef','driverRef','year']].copy()
    for ind,row in zero_drs.iterrows():
        circuit = df['circuitRef']== row['circuitRef']
        driver = df['driverRef'] == row['driverRef']
        year = df['year'] == row['year']
        not_zero = df['avg_lap_time_on_DRS'] != 0
        if len(df.loc[(circuit)&(driver)&(not_zero),'avg_lap_time_on_DRS'])>0:
            avg_drs_time = df.loc[(circuit)&(driver)&(not_zero),'avg_lap_time_on_DRS'].mean()
            avg_drs_dist = df.loc[(circuit)&(driver)&(not_zero),'avg_lap_distance_on_DRS'].mean()
            fl_drs_time = df.loc[(circuit)&(driver)&(not_zero),'fl_lap_time_on_DRS'].mean()
            fl_drs_distance = df.loc[(circuit)&(driver)&(not_zero),'fl_lap_distance_on_DRS'].mean()
            
        elif len(df.loc[(circuit)&(not_zero),'avg_lap_time_on_DRS'])>0:
            avg_drs_time = df.loc[(circuit)&(not_zero),'avg_lap_time_on_DRS'].mean()
            avg_drs_dist = df.loc[(circuit)&(not_zero),'avg_lap_distance_on_DRS'].mean()
            fl_drs_time = df.loc[(circuit)&(not_zero),'fl_lap_time_on_DRS'].mean()
            fl_drs_distance = df.loc[(circuit)&(not_zero),'fl_lap_distance_on_DRS'].mean()
        elif len(df.loc[(driver)&(not_zero),'avg_lap_time_on_DRS'])>0:
            avg_drs_time = df.loc[(driver)&(not_zero),'avg_lap_time_on_DRS'].mean()
            avg_drs_dist = df.loc[(driver)&(not_zero),'avg_lap_distance_on_DRS'].mean()
            fl_drs_time = df.loc[(driver)&(not_zero),'fl_lap_time_on_DRS'].mean()
            fl_drs_distance = df.loc[(driver)&(not_zero),'fl_lap_distance_on_DRS'].mean()
        else:
            avg_drs_time = df.loc[(not_zero),'avg_lap_time_on_DRS'].mean()
            avg_drs_dist = df.loc[(not_zero),'avg_lap_distance_on_DRS'].mean()
            fl_drs_time = df.loc[(not_zero),'fl_lap_time_on_DRS'].mean()
            fl_drs_distance = df.loc[(not_zero),'fl_lap_distance_on_DRS'].mean()
        df2.loc[(circuit)&(driver)&(year),'avg_lap_time_on_DRS'] = avg_drs_time
        df2.loc[(circuit)&(driver)&(year),'avg_lap_distance_on_DRS'] = avg_drs_dist
        df2.loc[(circuit)&(driver)&(year),'fl_lap_time_on_DRS'] = fl_drs_time
        df2.loc[(circuit)&(driver)&(year),'fl_lap_distance_on_DRS'] = fl_drs_distance

    return df2
    # %%



















