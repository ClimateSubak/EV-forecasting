import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_single_msoa(y_train, y_test, y_pred,msoa):
    
    # Map Y-pred array to a dataframe with consistent indices  to test set
    df_pred = pd.DataFrame(index=y_test.index)
    df_pred['ev_count'] = y_pred
    df_pred.head()
    
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(y_train.loc[msoa,:], color='k', label='Train')
    ax.plot(y_test.loc[msoa,:], color='b', label='Test')
    ax.plot(df_pred.loc[msoa,:], color='b', linestyle=':',label='Predicted')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('EV Count')
    plt.xticks([min(y_train.loc[msoa,:].index), 
        max(y_train.loc[msoa,:].index),
        max(y_test.loc[msoa,:].index)],
        rotation=45)
    plt.axvline(x=max(y_train.loc[msoa,:].index), color='k', linestyle='--')
    plt.legend()

    return ax

def plot_dated_evdist(y_test, y_pred, msoas, date):
    
    idx = pd.IndexSlice
    
    # Map Y-pred array to a dataframe with consistent indices  to test set
    df_pred = pd.DataFrame(index=y_test.index)
    df_pred['ev_count'] = y_pred
    
    ax =  sns.distplot(df_pred.loc[idx[msoas,date,:]]['ev_count'],
                 hist_kws={
                 'rwidth': 0.85,
                 'edgecolor': 'black',
                 'alpha': 0.2}, label='Predicted')
    ax = sns.distplot(y_test.loc[idx[msoas,:]]['ev_count'],
                 hist_kws={
                 'rwidth': 0.85,
                 'edgecolor': 'black',
                 'alpha': 0.2},
                  label='Test')
    ax.set_title('Nonzero EV Distribution')
    plt.legend()
    return ax

def plot_steady_evdist(y_test, y_pred, msoas):
    
    df_pred = pd.DataFrame(index=y_test.index)
    df_pred['ev_count'] = y_pred

    ax =  sns.distplot(df_pred.loc[msoas]['ev_count'],
                 hist_kws={
                 'rwidth': 0.85,
                 'edgecolor': 'black',
                 'alpha': 0.2}, label='Predicted')
    ax = sns.distplot(y_test.loc[msoas]['ev_count'],
                 hist_kws={
                 'rwidth': 0.85,
                 'edgecolor': 'black',
                 'alpha': 0.2},
                  label='Test')
    ax.set_title('Nonzero EV Distribution')
    plt.legend()
    
    return ax
    

def plot_single_msoa_train_val_test(y_train, y_val, y_test, y_pred_test, y_pred_val, msoa):
    
    # Map Y-pred array to a dataframe with consistent indices  to test set
    df_pred_test = pd.DataFrame(index=y_test.index)
    df_pred_test['ev_count'] = y_pred_test
    df_pred_test.head()
    
    # Map Y-pred array to a dataframe with consistent indices  to test set
    df_pred_val = pd.DataFrame(index=y_val.index)
    df_pred_val['ev_count'] = y_pred_val
    df_pred_val.head()
    
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(y_train.loc[msoa,:], color='k', label='Train')
    ax.plot(y_val.loc[msoa,:], color='k',linestyle='-', label='Validation')
    ax.plot(df_pred_val.loc[msoa,:], color='b', linestyle=':',label='Predicted (val)')
    ax.plot(y_test.loc[msoa,:], color='b', label='Test')
    ax.plot(df_pred_test.loc[msoa,:], color='b', linestyle=':',label='Predicted (test)')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('EV Count')
    
    plt.xticks([min(y_train.loc[msoa,:].index), 
        max(y_train.loc[msoa,:].index),
        max(y_val.loc[msoa,:].index),
        max(y_test.loc[msoa,:].index)],
        rotation=45)
    
    plt.axvline(x=max(y_train.loc[msoa,:].index), color='k', linestyle='--')
    plt.axvline(x=max(y_val.loc[msoa,:].index), color='k', linestyle='--')
    plt.legend()

    return ax