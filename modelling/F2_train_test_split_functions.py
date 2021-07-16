import pandas as pd
import geopandas as gpd
import numpy as np

def func_test_train_split_xt(X,y,train_size_full,random_state):
    """ Function to split the spatial-temporal data into test and train.
    N.B. Splitting on both axes will result in loss of data."""
    
    np.random.seed(random_state)
    idx = pd.IndexSlice
    
    # need to sqrt the train_size to result in a true train_size reduction
    train_size = np.sqrt(train_size_full)
    
    # randomly sample msoa's
    msoa_index = np.unique(np.array(X.index.droplevel(1)))
    N_sample_msoa = np.floor(len(msoa_index)*train_size).astype(int)
    msoa_train = np.random.choice(msoa_index,N_sample_msoa,replace=False)
    msoa_test = np.array(list(set(msoa_index)-set(msoa_train)))
    
    # truncate random sample into test and train using year split
    N_dates = X.loc[np.array(X.index.droplevel(1))[0]].shape[0]
    
    ordered_dates = np.sort(np.array(X.index.droplevel(0)[:N_dates]))
    date_to_split = np.floor(len(ordered_dates)*train_size).astype(int)
    
    train_dates = ordered_dates[:date_to_split]
    test_dates = ordered_dates[date_to_split:]
    
    X_test = X.loc[idx[msoa_test,test_dates],:]
    X_train = X.loc[idx[msoa_train,train_dates],:]
    y_test = y.loc[idx[msoa_test,test_dates],:]
    y_train = y.loc[idx[msoa_train,train_dates],:]
    
    return X_train, y_train, X_test, y_test

def func_test_train_split_time(X,y,train_size,random_state):
    """ Function to split the data along the temporal axis only.
    
    Input: X features, y variable to predict, train_size % of time steps to keep."""
    
    np.random.seed(random_state)
    idx = pd.IndexSlice
    
    # count total number of time steps
    N_dates = X.loc[np.array(X.index.droplevel(1))[0]].shape[0]
    
    # store ordered dates from the axis
    ordered_dates = np.array(X.index.droplevel(0)[:N_dates])
    
    # find date to split on
    date_to_split = np.floor(len(ordered_dates)*train_size).astype(int)
    
    train_dates = ordered_dates[:date_to_split]
    test_dates = ordered_dates[date_to_split:]
    
    X_test = X.loc[idx[:,test_dates],:]
    X_train = X.loc[idx[:,train_dates],:]
    y_test = y.loc[idx[:,test_dates],:]
    y_train = y.loc[idx[:,train_dates],:]
    
    return X_train, y_train, X_test, y_test

def func_test_train_split_msoa(X,y,train_size,random_state):
    """ Function to split the spatial data into test and train
    where dataframe has a multiindex of (MSOA, time)."""
    
    np.random.seed(random_state)
    idx = pd.IndexSlice
    
    # randomly sample msoa's
    msoa_index = np.unique(np.array(X.index.droplevel(1)))
    N_sample_msoa = np.floor(len(msoa_index)*train_size).astype(int)
    msoa_train = np.random.choice(msoa_index,N_sample_msoa,replace=False)
    msoa_test = np.array(list(set(msoa_index)-set(msoa_train)))
     
    X_test = X.loc[idx[msoa_test,:],:]
    X_train = X.loc[idx[msoa_train,:],:]
    y_test = y.loc[idx[msoa_test,:],:]
    y_train = y.loc[idx[msoa_train,:],:]
    
    return X_train, y_train, X_test, y_test

def func_test_train_split_msoa_single(X,y,train_size,random_state):
    """ Function to split the spatial data into test and train
    where dataframe has a single index of MSOA."""
    
    np.random.seed(random_state)
    idx = pd.IndexSlice

    # randomly sample msoa's
    msoa_index = np.unique(np.array(X.index))
    N_sample_msoa = np.floor(len(msoa_index)*train_size).astype(int)
    msoa_train = np.random.choice(msoa_index,N_sample_msoa,replace=False)
    msoa_test = np.array(list(set(msoa_index)-set(msoa_train)))
     
    X_test = X.loc[msoa_test,:]
    X_train = X.loc[msoa_train,:]
    y_test = y.loc[msoa_test,:]
    y_train = y.loc[msoa_train,:]

    return X_train, y_train, X_test, y_test

def func_test_val_train_split_time(X,y, end_train_date, end_val_date):
    """ Function to split the data along the temporal axis only.

    
    Input: X features, y variable to predict, end_train_date, end_val_date
    Output: train, validation, test data."""
    
    idx = pd.IndexSlice
    
    # count total number of time steps
    N_dates = X.loc[np.array(X.index.droplevel(1))[0]].shape[0]
    
    # store ordered dates from the axis
    ordered_dates = np.array(X.index.droplevel(0)[:N_dates])
    
    index_end_train = np.where(ordered_dates == end_train_date)[0][0]
    index_end_validation = np.where(ordered_dates == end_val_date)[0][0]
    
    X_train =  X.loc[idx[:,ordered_dates[:index_end_train]],:]
    y_train = y.loc[idx[:,ordered_dates[:index_end_train]],:]
    
    X_val = X.loc[idx[:,ordered_dates[index_end_train:index_end_validation]],:]
    y_val = y.loc[idx[:,ordered_dates[index_end_train:index_end_validation]],:]
    
    X_test = X.loc[idx[:,ordered_dates[index_end_validation:]],:]
    y_test = y.loc[idx[:,ordered_dates[index_end_validation:]],:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def func_test_val_train_split(X,y, start_test_date, train_size):
    """ Function to split the data along the temporal axis for the test data
    and along the spatial axis (MSOA) for the validation data.
    
    Input:  X features, 
            y variable to predict, 
            start_test_date to split the temporal axis
            train_size to split the training set into train and validate
    Output: train, validation, test data."""
    
    idx = pd.IndexSlice
    
    ## ----- temporal split --------
    # count total number of time steps
    N_dates = X.loc[np.array(X.index.droplevel(1))[0]].shape[0]
    # store ordered dates from the axis
    ordered_dates = np.array(X.index.droplevel(0)[:N_dates])
    # find index of time split
    index_start_test = np.where(ordered_dates == start_test_date)[0][0]
    
    ## ------ msoa split ----------
    # find index for MSOA selection to split train and validate
    # randomly sample msoa's
    msoa_index = np.unique(np.array(X.index.droplevel(1)))
    N_train_msoa = np.floor(len(msoa_index)*train_size).astype(int)
    msoa_train = np.random.choice(msoa_index,N_train_msoa,replace=False)
    msoa_val = np.array(list(set(msoa_index)-set(msoa_train)))
    
    ## -------- split data into test, validate, train
    X_train =  X.loc[idx[msoa_train,ordered_dates[:index_start_test]],:]
    y_train = y.loc[idx[msoa_train,ordered_dates[:index_start_test]],:]
    
    X_val = X.loc[idx[msoa_val,ordered_dates[:index_start_test]],:]
    y_val = y.loc[idx[msoa_val,ordered_dates[:index_start_test]],:]
    
    X_test = X.loc[idx[:,ordered_dates[index_start_test:]],:]
    y_test = y.loc[idx[:,ordered_dates[index_start_test:]],:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test