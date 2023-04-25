import numpy as np
import netCDF4 as nc
from pdb import set_trace
from sklearn import linear_model
import matplotlib.pyplot as plt

def read_data_from_netcdf(y_filename, x_filename_list, 
                          subset_function=None):
    """Read data from a netCDF file 
    Assumes that the variables in the netcdf file all have the name "variable"
    Assunes that values < -9E9, you dont want. This could be different in some circumstances.
    Arguments:

    y_filename -- a two element python list containing the name of the file and the target variable name
    x_filename_list -- a python list of filename containing the feature variables
    subset_function -- a function to be applied to each data set

    Returns:

    Y - a numpy array of the target variable
    X - an n-D numpy array of the feature variables 
    """
    
    y_dataset=nc.Dataset(y_filename[0])[y_filename[1]]
    y_dataset = np.array(y_dataset).flatten()
    
    if subset_function is not None:
        Y=subset_function(y_dataset)
    else:
        Y=y_dataset

    n=len(Y)
    m=len(x_filename_list)
    X=np.zeros([n,m])

    for i, filename in enumerate(x_filename_list):

        x_dataset = nc.Dataset(filename[0])[filename[1]]
        x_dataset = np.array(x_dataset).flatten()
        X[:, i]=x_dataset
    
    cells_we_want = np.array([np.all(rw > -9e9) for rw in X])
    Y = Y[cells_we_want]
    X = X[cells_we_want, :]
    return Y, X

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

if __name__=="__main__":
    dir = "/home/h02/dkelley/ConFIRE_attribute/isimip3a/driving_data/GSWP3-W5E5-20yrs/Brazil/AllConFire_2000_2009/"
    y_filen = [dir +"GFED4.1s_Burned_Fraction.nc", "Date"]
    
    x_filen_list=[]
    x_filen_list.append([dir + "precip.nc", "variable"])    
    x_filen_list.append([dir + "tas.nc", "variable"])    
    
    Y, X=read_data_from_netcdf(y_filen, x_filen_list)
    
    reg = fit_linear_to_data(Y, X)
    plt.plot(Y, reg.predict(X), '.')
    set_trace()
    print(Y)
    print(X)

