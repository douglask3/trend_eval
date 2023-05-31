import numpy as np
from pdb import set_trace

import iris
from libs.iris_plus import *
from libs.constrain_cubes_standard import *
from libs.read_variable_from_netcdf import *

def read_variable_from_netcdf(filename, dir = '', subset_function = None, 
                              make_flat = False, subset_function_args = None,
                              time_series = None, 
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
        subset_function -- a function or list of functions to be applied to each data set
        subset_function_args -- If subset_function is a function, dict arguments for that function. 
                            If subset_function is a list, a  list or dict constaining arguments 
                                for those functions in turn.
        make_flat -- should the output variable to flattened or remain cube
        time_series -- list comtaining range of years. If making flat and returned a time series, 
            checks if that time series contains year.
    Returns:
        Y - if make_flat, a numpy vector of the target variable, otherwise returns iris cube
    """

    print("Opening:")
    print(filename)
    
    try:
        if isinstance(filename, str):        
            dataset = iris.load_cube(dir + filename, callback=sort_time)
        else:
            dataset = iris.load_cube(dir + filename[0], filename[1], callback=sort_time)
    except:
        set_trace()

    if subset_function is not None:
        if isinstance(subset_function, list):
            for FUN, args in zip(subset_function, subset_function_args):
                dataset = FUN(dataset, **args)
        else: dataset = subset_function(dataset, **subset_function_args)     
        
    
    if make_flat: 
        if time_series is not None: years = dataset.coord('year').points
        dataset = dataset.data.flatten()
        if time_series is not None:
            if not years[ 0] == time_series[0]:
                dataset = np.append(np.repeat(np.nan, years[ 0]-time_series[0]), dataset)
            if not years[-1] == time_series[1]:
                dataset = np.append(dataset, np.repeat(np.nan, time_series[1]-years[-1]))
    
    return dataset

def read_all_data_from_netcdf(y_filename, x_filename_list, add_1s_columne = False, 
                              y_threshold = None, check_mask = True, *args, **kw):
    """Read data from netCDF files 
        
    Arguments:
        y_filename -- a two element python list containing the name of the file and the target 
            variable name
        x_filename_list -- a python list of filename containing the feature variables
        y_threshold -- if converting y into boolean, the threshold we use to spit into 
            0's and 1's
        add_1s_columne --useful for if using for regressions. Adds a variable of 
            just 1's t rperesent y = SUM(a_i * x_i) + c
        check_mask -- Boolean. If True, simple checks if there are any large negtaive numbers 
            and makes them out. Assunes that values < -9E9, you dont want. This could be different 
            in some circumstances
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
    
    X = np.zeros([n,m])
    
    for i, filename in enumerate(x_filename_list):
        X[:, i]=read_variable_from_netcdf(filename, make_flat = True, *args, **kw)

    if add_1s_columne: 
        X = np.column_stack((X, np.ones(len(X)))) # add a column of ones to X 
    
    if check_mask:
        cells_we_want = np.array([np.all(rw > -9e9) for rw in np.column_stack((X, Y))])
        Y = Y[cells_we_want]
        X = X[cells_we_want, :]

        return Y, X, cells_we_want
    return Y, X
