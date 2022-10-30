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

def prepare_modelling_df(df,fast_lap='all'):
    df = convert_object_to_float(df)
    if fast_lap=='all':
        df = engineer_target_bin(df,fast_lap='fastest_all_sessions_milliseconds')
    else:
        df = engineer_target_bin(df,fast_lap='fastest_lap_milliseconds')
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
    X_test = df[query].drop(columns=['fastest_lap_milliseconds', 'lap_timedelta_milliseconds','quali_position','raceId','quali_position_binned','lap_timedelta_seconds_binned','fastest_all_sessions_milliseconds']).copy()
    X_train = df[~query].drop(columns=['fastest_lap_milliseconds', 'lap_timedelta_milliseconds','quali_position','raceId','quali_position_binned','lap_timedelta_seconds_binned','fastest_all_sessions_milliseconds']).copy()
    y_test=df.loc[query,['lap_timedelta_seconds_binned','quali_position_binned','lap_timedelta_milliseconds','quali_position']].copy()
    y_train=df.loc[~query,['lap_timedelta_seconds_binned','quali_position_binned','lap_timedelta_milliseconds','quali_position']].copy()
    
    return X_test, X_train, y_test, y_train


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


# %%



# inspired from https://www.andrewvillazon.com/custom-scikit-learn-transformers/
class ColumnSelector_BP(BaseEstimator, TransformerMixin,_OneToOneFeatureMixin):
    """Object for selecting specific columns from a data set.
    Parameters
    ----------
    cols : array-like (default: None)
        A list specifying the feature indices to be selected. For example,
        [1, 4, 5] to select the 2nd, 5th, and 6th feature columns, and
        ['A','C','D'] to select the name of feature columns A, C and D.
        If None, returns all columns in the array.
    drop_axis : bool (default=False)
        Drops last axis if True and the only one column is selected. This
        is useful, e.g., when the ColumnSelector is used for selecting
        only one column and the resulting array should be fed to e.g.,
        a scikit-learn column selector. E.g., instead of returning an
        array with shape (n_samples, 1), drop_axis=True will return an
        aray with shape (n_samples,).
    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/feature_selection/ColumnSelector/
    """

    def __init__(self, cols=None, drop_axis=False):
        self.cols = cols
        self.drop_axis = drop_axis

    def fit_transform(self, X, y=None):
        """Return a slice of the input array.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples] (default: None)
        Returns
        ---------
        X_slice : shape = [n_samples, k_features]
            Subset of the feature space where k_features <= n_features
        """
        return self.transform(X=X, y=y)

    def transform(self, X, y=None):
        """Return a slice of the input array.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples] (default: None)
        Returns
        ---------
        X_slice : shape = [n_samples, k_features]
            Subset of the feature space where k_features <= n_features
        """

        # We use the loc or iloc accessor if the input is a pandas dataframe
        if hasattr(X, "loc") or hasattr(X, "iloc"):
            if type(self.cols) == tuple:
                self.cols = list(self.cols)
            types = {type(i) for i in self.cols}
            if len(types) > 1:
                raise ValueError(
                    "Elements in `cols` should be all of the same data type."
                )
            if isinstance(self.cols[0], int):
                t = X.iloc[:, self.cols].values
            elif isinstance(self.cols[0], str):
                t = X.loc[:, self.cols].values
            else:
                raise ValueError("Elements in `cols` should be either `int` or `str`.")
        else:
            t = X[:, self.cols]

        if t.shape[-1] == 1 and self.drop_axis:
            t = t.reshape(-1)
        if len(t.shape) == 1 and not self.drop_axis:
            t = t[:, np.newaxis]
        return t

    def fit(self, X, y=None):
        """Mock method. Does nothing.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples] (default: None)
        Returns
        ---------
        self
        """
        return self.transform(X=X, y=y)
