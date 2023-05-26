import numpy as np
import netCDF4 as nc
from pdb import set_trace
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import iris
from   libs.iris_plus import *
from   libs.constrain_cubes_standard import *
from   libs.plot_maps    import *

def read_variable_from_netcdf(filename, dir = '', subset_function = None, 
                              make_flat = False, subset_function_args = None,
                              *args, **kw):
    """Read data from a netCDF file 
        Assumes that the variables in the netcdf file all have the name "variable"
        Assunes that values < -9E9, you dont want. This could be different in some circumstances
    Arguments:
        filename -- a string with filename or two element python list containing the name of 
            the file and the target variable name. If just the sting of "filename" 
            assumes variable name is "variable"
        dir -- directory file is in. Path can be in "filename" and None means no 
            additional directory path needed.
    subset_function -- a function to be applied to each data set
    make_flat - should the output variable to flattened or remain cybe

    Returns:
        Y - if make_flat, a numpy vector of the target variable, otherwise returns iris cube
    """

    print("Opening:")
    print(filename)
    
    if isinstance(filename, str): 
        dataset = iris.load_cube(dir + filename, callback=sort_time)        
    else:
        try:
            dataset = iris.load_cube(dir + filename[0], filename[1], callback=sort_time)
        except:
            set_trace()

    if subset_function is not None:
        if isinstance(subset_function, list):
            for FUN, args in zip(subset_function, subset_function_args):
                dataset = FUN(dataset, **args)
        else: dataset = subset_function(dataset, **subset_function_args)     
        
    
    if make_flat: dataset = dataset.data.flatten()
    
    return dataset

def read_all_data_from_netcdf(y_filename, x_filename_list, add_1s_columne = False, 
                              y_threshold = None, *args, **kw):
    """Read data from netCDF files 
        Assunes that values < -9E9, you dont want. This could be different in some circumstances
    Arguments:
        y_filename -- a two element python list containing the name of the file and the target 
            variable name
        x_filename_list -- a python list of filename containing the feature variables
        y_threshold -- if converting y into boolean, the threshold we use to spit into 
            0's and 1's
        add_1s_columne --useful for if using for regressions. Adds a variable of 
            just 1's t rperesent y = SUM(a_i * x_i) + c
        see read_variable_from_netcdf comments for *arg and **kw.
    Returns:
        Y - a numpy array of the target variable
        X - an n-D numpy array of the feature variables 
    """
    Y = read_variable_from_netcdf(y_filename, make_flat = True, *args, **kw)
    # Create a new categorical variable based on the threshold
    if y_threshold is not None:
        Y = np.where(Y >= y_threshold, 0, 1)
        #count number of 0 and 1 
        counts = np.bincount(Y)
        #print(f"Number of 0's: {counts[0]}, Number of 1's: {counts[1]}")
    
    n=len(Y)
    m=len(x_filename_list)
    X=np.zeros([n,m])

    for i, filename in enumerate(x_filename_list):
        X[:, i]=read_variable_from_netcdf(filename, make_flat = True, *args, **kw)
    
    cells_we_want = np.array([np.all(rw > -9e9) for rw in np.column_stack((X, Y))])
    Y = Y[cells_we_want]
    X = X[cells_we_want, :]

    if add_1s_columne: 
        X = np.column_stack((X, np.ones(len(X)))) # add a column of ones to X 

    return Y, X, cells_we_want

def read_data_from_csv(filename):
    """Read data from a file 
    """
    pass

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
                                           subset_function = [sub_year_months, ar6_region], 
                                           subset_function_args = [{'months_of_year': [6, 7, 8]},
                                                                   {'region_code' : 'NSA'}])
    
    ## Perform regression
    #reg = fit_linear_to_data(Y, X)
    #plt.plot(Y, reg.predict(X), '.')
    
    logr = fit_logistic_to_data(Y, X)
    #plt.plot(logr.predict_proba(X)[:,1], Y, '.')
    
    ## Predict and plot training period from fitted model
    Obs = read_variable_from_netcdf(y_filen, dir, 
                                           subset_function = [sub_year_months, ar6_region],  
                                           subset_function_args = [{'months_of_year': [6, 7, 8]},
                                                                   {'region_code' : 'NSA'}])
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
