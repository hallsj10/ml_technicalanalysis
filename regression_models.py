# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 12:44:20 2020

@author: halls35
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

def regression_predictions(x_train, y_train, x_test, y_test):
    
    dts = y_test.index
    # ESTIMATORS
    ESTIMATORS = {
        "Tree": DecisionTreeRegressor(max_depth=10),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=10, random_state=0),
        "Forest": RandomForestRegressor(n_estimators=10, random_state=0),
        "KNN": KNeighborsRegressor(),
        "LinearRegression": LinearRegression()}
    
    y_test_predict = dict()
    # KNN Hyperparameters
    num_neighbors = [50, 100, 150]
    
    for name, estimator in ESTIMATORS.items():
        if name == 'KNN':
            for i, wgts in enumerate(['uniform', 'distance']):
                for j, k in enumerate(num_neighbors):
                    model = KNeighborsRegressor(k, weights=wgts)
                    model.fit(x_train, y_train)
                    prediction = model.predict(x_test)
                    y_test_predict[name+'_'+wgts+'_'+str(k)] = pd.Series(prediction)
        else:
            model = estimator.fit(x_train, y_train)
            prediction = model.predict(x_test)
            y_test_predict[name] = pd.Series(prediction)
    
    # Add straight line to forecast (baseline)
    ave_y_train = y_train.mean()
    y_test_predict['Naive'] = pd.Series(np.full((len(x_test),),ave_y_train))
    
    cols = list(y_test_predict.keys())
    out = pd.concat(y_test_predict.values(), axis=1, ignore_index=True)
    out.columns = cols
    out.set_index(dts, inplace=True)
    
    return out

def regression_results(y_test, y_predict):
    
    # Iterate through keys in dictionary to get test residuals and RMSE
    residuals = dict()
    rmse = dict()
    for (name, prediction) in y_predict.iteritems():        
        residuals[name] = y_test - prediction
        rmse[name] = np.sqrt(mean_squared_error(y_test, prediction))
        
    return residuals, rmse