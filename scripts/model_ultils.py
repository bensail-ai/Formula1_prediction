#!/usr/bin/env python3
#%%
import math
import os
import re
from sklearn.base import BaseEstimator, TransformerMixin, _OneToOneFeatureMixin
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def engineer_target_bin(df:pd.DataFrame,fast_lap='fastest_all_sessions_milliseconds'):
    """
    Function creates the Target variables for classification

    It bins qualifying position into bins of 4 0-5 place, 5-10 place, 10-15 place and 15 to 20th
    It bins lap time delta by quartile typically 0-0.9s,0.9-1.6,1.6-2.3 and >2.3

    
    Args:
        df (pd.DataFrame): Formula1 aggregated DataFrame of Features 

    Returns:
        pd.DataFrame: Dataframe with addittional variables of lap_time_delta binned and quali_position_binned
    """   
    df['quali_position_binned'] = pd.cut(df['quali_position'],bins=4,labels=['0_to_5','5_to_10','eleven_to_15','fifteen_&_above'])
    races = df['raceId'].unique()
    for race in races:
        query = (df['raceId'] == race)
        fastesttime= df.loc[(query),fast_lap].min()
        
        df.loc[query,'lap_timedelta_milliseconds']=(np.where(df.loc[query,'quali_position']==1,0,(fastesttime-df.loc[query,fast_lap])))
    df['lap_timedelta_seconds_binned'] = pd.qcut(df['lap_timedelta_milliseconds'],q=4,labels=['>2.3s','2.3-1.6s','1.6-0.9s','0.9-0s'])
    return df

def convert_object_to_float(df:pd.DataFrame):
    """
    Converts all the object columns to numbers accept for the columns in true_cat

    Args:
        df (pd.DataFrame): Formula1 aggregated DataFrame of Features 

    Returns:
        pd.DataFrame: Formula aggregate DataFrame of features
    """
    cat_list = list(df.select_dtypes("object").columns)
    true_cat=['driverRef', 'name', 'circuitRef', 'country',
        'nationality_drivers', 'constructorRef', 'nationality_constructors',
        'dob','fl_tyre']
    convert_list=[cat for cat in cat_list if cat not in true_cat]
    for col in convert_list:
        df[col]=df[col].astype(float)
    return df

def feature_engineer_country(df:pd.DataFrame):
    """
    Calculates the feature of home country for drivers and teams

    Args:
        df (pd.DataFrame): Formula1 aggregated DataFrame of Features 

    Returns:
        pd.DataFrame: Formula1 aggregated DataFrame of Features 
    """
    df['driver_home_race']=np.where(df['country'] == df['nationality_drivers'],1,0)
    df['constructor_home_race']= np.where(df['country'] == df['nationality_constructors'],1,0)

    df.drop(columns=['country','nationality_drivers','nationality_constructors'],axis=1, inplace=True)

    return df


def clean_df(df:pd.DataFrame,columns=['number','name','dob']):
    """
    Function drops the columns from the column list in a DataFrame

    Args:
        df (pd.DataFrame): Pandas DataFrame
        columns (list, optional): columns to remove. Defaults to ['number','name','dob'].

    Returns:
        pd.DataFrame: Pandas DataFrame
    """
    df.drop(columns=columns,axis=1, inplace=True)

    return df

def prepare_modelling_df(df:pd.DataFrame,fast_lap='all',test_raceids=None, test_split=True):
    """
    This function runs all the necessary preprocessing to prepare the Data for modelling from the feature aggregated
    combined DataFrame

    

    Args:
        df (pd.DataFrame): combined formula1 dataframe with all aggregated features
        fast_lap (str, optional): which fastest lap to use either cleaned q3 times or fastest across all sessions. Defaults to 'all'.
        test_raceids (_type_, optional): RaceIds to be split into the Test set. If None selected at Random. Defaults to None.
        test_split (bool, optional): If True splits the prepared data into test and train dataframes, if False returns prepared X and y DataFrames. Defaults to True.

    Returns:
        if test_split is True:
            X_test (pd.DataFrame): Test X data
            X_train (pd.DataFrame): Train X data
            y_test (pd.DataFrame): Test y features, 'lap_timedelta_seconds_binned','quali_position_binned','lap_timedelta_milliseconds','quali_position'
            y_train (pd.DataFrame): Train y features, 'lap_timedelta_seconds_binned','quali_position_binned','lap_timedelta_milliseconds','quali_position'
        if test_split is False:
            X (pd.DataFrame): X data
            y (pd.DataFrame): y variables, 'lap_timedelta_seconds_binned','quali_position_binned','lap_timedelta_milliseconds','quali_position'
    """
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


def column_lists(x:pd.DataFrame):
    """
    Returns the list of categorical columns to hot encode and
    the list of columnsto scale

    Args:
        x (pd.DataFrame): X  feature data

    Returns:
        list: list of column names to scale
        list: list of column names to hot encode
    """
    df=x.copy()    
    hot_encode_col=list(df.select_dtypes("object").columns)
    scale_cols = [col for col in df.columns if col not in hot_encode_col]
    return scale_cols, hot_encode_col


def apply_manual_features(X_train,X_test,features=[]):
    """
    Selects the manual features from the list of features in X Train and X test 

    Since the feature names have prefix transformed_ 
    and for the categorical columns the encoded prefix and prefix of column name, e.g.:
    encoded__driverRef_ 

    these prefixs have to be removed to match X_train and X_test feature names 
    Args:
        X_train (pd.DataFrame): X train data
        X_test (pd.DataFrame): X test data
        features (list, optional): features to select. Defaults to [].

    Returns:
        pd.DataFrame: X train with subset of features
        pd.DataFrame: X test with subset of features
    """
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
    """
    Selects the manual features from the list of features in X 

    Since the feature names have prefix transformed_ 
    and for the categorical columns the encoded prefix and prefix of column name, e.g.:
    encoded__driverRef_ 

    these prefixs have to be removed to match X_train and X_test feature names 
    Args:
        X (pd.DataFrame): X data       
        features (list, optional): features to select. Defaults to [].

    Returns:
        pd.DataFrame: X with subset of features
        
    """
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

def fix_zero_DRS(df:pd.DataFrame):
    """
    In some cases the laps have a DRS code 9 and do not change for when it is opened or closed. From Feature aggregations this return
    DRS time and distance of 0.

    This function cleans the 0 by finding the average time and distance DRS is open from the same circuit from the previous year in the database.

    If that doesn't exist it just takes the average for that circuit

    If that doesn't exist it takes the drivers average

    Args:
        df (pd.DataFrame): Formula1 aggregated DataFrame of Features 

    Returns:
        pd.DataFrame: Formula1 aggregated DataFrame of Features with DRS cleaned
    """
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



















