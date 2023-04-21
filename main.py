import numpy as np
import netCDF4 as nc

def read_data_from_netcdf(y_filename, x_filename_list, subset_function=None):
    """Read data from a netCDF file 
    Assumes that the variables in the netcdf file all have the name "variable"

    Arguments:

    y_filename -- the name of the file containing the target variables
    x_filename_list -- a python list of filename containing the feature variables
    subset_function -- a function to be applied to each data set

    Returns:

    Y - a numpy array of the target variable
    X - an n-D numpy array of the feature variables 
    """
    y_dataset=nc.Dataset(y_filename)['variable'].flatten()

    if subset_function is not None:
        Y=subset_function(y_dataset)
    else:
        Y=y_dataset

    n=len(Y)
    m=len(x_filename_list)
    X=np.zeros([n,m])

    for i, filename in enumerate(x_filename_list):

        x_dataset=nc.Dataset(filename)['variable'].flatten()
        X[i,:]=x_dataset

    return Y, X

def read_data_from_csv(filename):
    """Read data from a file 
    """
    pass

def fit_linear_to_data(Y, X):
    """Fit equation to data
    """

    return A

if __name__=="__main__":
    
    y_filen="foo.nc"
    x_filen_list=[]
    x_filen_list.append("bar1.nc")    
    x_filen_list.append("bar2.nc")    

    Y, X=read_data_from_netcdf(y_filen, x_filen_list)

    print(Y)
    print(X)

