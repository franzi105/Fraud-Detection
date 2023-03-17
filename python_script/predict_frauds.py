# Import all neccessary libraries
import numpy as np
import pandas as pd

# sklearn
from sklearn.model_selection import train_test_split

# for XGBoost
import xgboost as xgb
from xgboost import XGBClassifier

# for grid search
from sklearn.model_selection import RandomizedSearchCV

import warnings
warnings.simplefilter('ignore')


# function to load .csv client and invoice data tables each for train and test data
def load_data(file_client_train, file_invoice_train, file_client_test, file_invoice_test): 
    """loads .csv client and invoice data tables each for train and test data

    Args:
        file_client_train (str): path to client train data table, e.g. 'data/original/client_train.csv'
        file_invoice_train (str): path to invoice train data table, e.g. 'data/original/invoice_train.csv'
        file_client_test (str): path to client test data table, e.g. 'data/original/client_test.csv'
        file_invoice_test (str): path to invoice test data table, e.g. 'data/original/invoice_test.csv'

    Returns:
        dataframes: client_train, invoice_train, client_test, invoice_test
    """

    # read client and invoice data tables for each train and test data from .csv to pandas DataFrame
    client_train = pd.read_csv(file_client_train, low_memory=False)
    invoice_train = pd.read_csv(file_invoice_train, low_memory=False)
    client_test = pd.read_csv(file_client_test, low_memory=False)
    invoice_test = pd.read_csv(file_invoice_test, low_memory=False)

    return client_train, invoice_train, client_test, invoice_test


# function for datetime conversion, one-hot encoding, calculating delta index, removing data from before 2005
def feature_engineering(client_train, invoice_train, client_test, invoice_test): 
    """datetime conversion, one-hot encoding, calculate delta index and remove data from before 2005

    Args:
        client_train (dataframe): _description_
        invoice_train_agg (dataframe): _description_
        client_test (dataframe): _description_
        invoice_test_agg (dataframe): _description_

    Returns:
        dataframes: client_train, invoice_train_agg, client_test, invoice_test_agg
    """

    #convert the column invoice_date to date time format on both the invoice train and invoice test
    for df in [invoice_train,invoice_test]:
        df['invoice_date'] = pd.to_datetime(df['invoice_date'])

    # one-hot encode labels in categorical column
    d={"ELEC":0,"GAZ":1}
    invoice_train['counter_type']=invoice_train['counter_type'].map(d)
    invoice_test['counter_type']=invoice_test['counter_type'].map(d)

    #convert categorical columns to int for model
    #client_train['client_catg'] = client_train['client_catg'].astype(int)
    #client_train['disrict'] = client_train['disrict'].astype(int)

    #client_test['client_catg'] = client_test['client_catg'].astype(int)
    #client_test['disrict'] = client_test['disrict'].astype(int)

    # calculate delta index as the difference between new_index and old_index
    invoice_train['delta_index'] = invoice_train.new_index - invoice_train.old_index
    invoice_train.drop(['old_index', 'new_index'], axis=1, inplace=True)

    invoice_test['delta_index'] = invoice_test.new_index - invoice_test.old_index
    invoice_test.drop(['old_index', 'new_index'], axis=1, inplace=True)

    # remove all invoices before 2005 as there were no frauds detected / documented
    invoice_train = invoice_train.query('invoice_date.dt.year >= 2005') 

    # aggregating invoice data by client_id 
    invoice_train_agg = aggregate_by_client_id(invoice_train)
    invoice_test_agg = aggregate_by_client_id(invoice_test)

    return client_train, invoice_train_agg, client_test, invoice_test_agg


# function for aggregating invoice data
def aggregate_by_client_id(invoice_data):
    """aggregating invoice data by client_id by taking mean (num) or mode (object)

    Args:
        invoice_data (_type_): _description_

    Returns:
        _type_: _description_
    """
    aggs = {}
    aggs['consommation_level_1'] = ['mean']
    aggs['consommation_level_2'] = ['mean']
    aggs['consommation_level_3'] = ['mean']
    aggs['consommation_level_4'] = ['mean']
    aggs['tarif_type'] = ['mean']
    aggs['counter_number'] = ['mean']
    aggs['counter_statue'] = [pd.Series.mode]
    aggs['counter_code'] = ['mean']
    aggs['reading_remarque'] = ['mean']
    aggs['counter_coefficient'] = ['mean']
    aggs['delta_index'] = ['mean']
    aggs['months_number'] = ['mean']
    aggs['counter_type'] = [pd.Series.mode]

    agg_trans = invoice_data.groupby(['client_id']).agg(aggs)
    agg_trans.columns = ['_'.join(col).strip() for col in agg_trans.columns.values]
    agg_trans.reset_index(inplace=True)

    df = (invoice_data.groupby('client_id')
            .size()
            .reset_index(name='{}transactions_count'.format('1')))
    return pd.merge(df, agg_trans, on='client_id', how='left')


# function to drop redundant columns
def drop_redundant_columns(train, test): 
    """drop redudant columns

    Args:
        train (dataframe): merged train data from clients and invoices
        test (dataframe): merged test data from clients and invoices
    """
    #drop redundant columns
    sub_client_id = test['client_id']
    drop_columns = ['client_id', 'creation_date']

    for col in drop_columns:
        if col in train.columns:
            train.drop([col], axis=1, inplace=True)
        if col in test.columns:
            test.drop([col], axis=1, inplace=True)
    
    return train, test


##################################################################################
# main function 
import os
import sys
import argparse

def main(file_client_train, file_invoice_train, file_client_test, file_invoice_test):

    # load data
    print('Loading data from: ')
    print(str(file_client_train) + '\n' + str(file_invoice_train) + '\n' + str(file_client_test) + '\n' + str(file_invoice_test))
    print('...')
    client_train, invoice_train, client_test, invoice_test = load_data(file_client_train, file_invoice_train, file_client_test, file_invoice_test)
    print('Finished loading data.')
    print('------------------------------')

    # feature engineering
    print('Engineering features...')
    client_train, invoice_train_agg, client_test, invoice_test_agg = feature_engineering(client_train, invoice_train, client_test, invoice_test)
    print('Finished engineering features...')
    print('------------------------------')

    # left join invoice to client table
    print('Merging client and invoice data... ')
    train = pd.merge(client_train, invoice_train_agg, on='client_id', how='left')
    test = pd.merge(client_test, invoice_test_agg, on='client_id', how='left')
    print('Finishe merging client and invoice data... ')
    print('------------------------------')

    # drop redundant columns
    print('Dropping redundant columns in dataframes...')
    train, test = drop_redundant_columns(train, test)
    print('Finished dropping redundant columns in dataframes. ')
    print('------------------------------')

    # Final training data cleaning 
    print('Final cleaning of the training data...')
    train.drop(['counter_type_mode', 'counter_statue_mode'], axis=1, inplace=True)
    train.dropna(axis=0, inplace=True)
    train.drop_duplicates(inplace=True)
    print('Finished final cleaning of the training data.')
    print('------------------------------')

    # Splitting training data 
    print('Preparing (splitting) training data for model building...')
    X = train.drop('target', axis=1)
    y = train.target
    X_train, X_test, y_train, y_test= train_test_split(X, y, stratify=y, test_size=0.2, random_state=42) 
    print('Finished preparing (splitting) training data for model building.')
    print('------------------------------')

    # Fitting final XGB model to train data
    print('Building best XGB model on training data...')
    model = XGBClassifier(max_depth = 5, learning_rate = 0.3, 
                        subsample = 0.7999999999999999, colsample_bytree = 0.8999999999999999, colsample_bylevel = 0.4, 
                        n_estimators = 500)
    model.fit(X_train, y_train)
    print('Finished building best XGB model on training data.')
    print('------------------------------')

    # Predicting with final XGB model
    print('Predicting targets (fraud clients) from test data...')
    y_pred = model.predict(X_test)
    print('Finished predicting targets (fraud clients) from test data.')
    print('------------------------------')

    # Save prediction to .csv file
    import pickle
    # save the model to disk
    filename = 'finalized_model.sav'
    pickle.dump(model, open(filename, 'wb'))

if __name__ == '__main__':
    file_client_train = sys.argv[1]
    file_invoice_train = sys.argv[2]
    file_client_test = sys.argv[3]
    file_invoice_test = sys.argv[4]

    main(file_client_train, file_invoice_train, file_client_test, file_invoice_test)


# python python_script/predict_frauds.py ./data/original/client_train.csv ./data/original/invoice_train.csv ./data/original/client_test.csv ./data/original/invoice_test.csv