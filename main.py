import numpy as np
from pdb import set_trace
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from libs.constrain_cubes_standard import *
from libs.plot_maps import *
from libs.read_variable_from_netcdf import *


def fit_linear_to_data(Y, X):
    """Use scikit learn to fit a linear equation.

    Arguments:
        Y -- numpy array of target variables 
        X -- numpy array of feature space variabels

    Returns.
        regr.coef_ -- regression coefficients
        Y_pred -- model prediction of the Y values

    **OR**
        regr -- return the regression model
    """


    regr = linear_model.LinearRegression()
    
    regr.fit(X, Y)

    Y_pred = regr.predict(X)

 

    #return regr.coef_, Y_pred

    return regr

def fit_logistic_to_data(Y, X):
    """Use scikit learn to fit a linear equation.

    Arguments:
        Y -- numpy array of target variables 
        X -- numpy array of feature space variabels

    Returns.

        logr.coef_ -- regression coefficients
        Y_pred -- model prediction of the Y values

    **OR**
        logr -- return the regression model
    """
    
   
    # Split data into train and test sets
    #X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    logr = LogisticRegression()
    logr.fit(X, Y)
    y_pred = logr.predict(X)

    return logr

if __name__=="__main__":

    ## Open data
    dir = "D:/Doutorado/Sanduiche/research/maxent-test/driving_and_obs_overlap/AllConFire_2000_2009/"
    dir = "../ConFIRE_attribute/isimip3a/driving_data/GSWP3-W5E5-20yrs/Brazil/AllConFire_2000_2009/"
    y_filen = ["GFED4.1s_Burned_Fraction.nc", "Date"]
    
    x_filen_list=["precip.nc", "tas.nc", "crop.nc", "humid.nc","vpd.nc", "csoil.nc", 
                  "lightn.nc", "rhumid.nc", "cveg.nc", "pas.nc", "soilM.nc", 
                   "totalVeg.nc", "popDens.nc", "trees.nc"]

    Y, X, lmask = read_all_data_from_netcdf(y_filen, x_filen_list, 
                                           add_1s_columne = True, dir = dir, y_threshold = 0.1,
                                           subset_function = [sub_year_months, constrain_GFED,
                                                              ar6_region], 
                                           subset_function_args = [{'months_of_year': [6, 7, 8]},
                                                                   {'region': [4,5]},
                                                                   {'region_code' : ['NWS', 'NSA']}])
    
    ## Perform regression
    #reg = fit_linear_to_data(Y, X)
    #plt.plot(Y, reg.predict(X), '.')
    
    logr = fit_logistic_to_data(Y, X)
    #plt.plot(logr.predict_proba(X)[:,1], Y, '.')
    
    ## Predict and plot training period from fitted model
    Obs = read_variable_from_netcdf(y_filen, dir, 
                                           subset_function = [sub_year_months, ar6_region],  
                                           subset_function_args = [{'months_of_year': [6, 7, 8]},
                                                                   {'region_code' : ['NWS', 'NSA']}])
    Pred = Obs.copy()
    pred = Pred.data.copy().flatten()
    pred[lmask] = logr.predict_proba(X)[:,0]
    Pred.data = pred.reshape(Pred.data.shape)

    levels = [0, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
    #levels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    cmap = 'OrRd'
    plot_annual_mean(Obs, levels, cmap, plot_name = "Observtations", scale = 100*12, 
                     Nrows = 1, Ncols = 2, plot_n = 1)

    plot_annual_mean(Pred, levels, cmap, plot_name = "Model", scale = 100*12, 
                     Nrows = 1, Ncols = 2, plot_n = 2)
    
    plt.show()

    
    print(Y)
    print(X)
    set_trace()
