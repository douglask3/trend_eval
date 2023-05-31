import numpy as np
import pandas as pd

import iris
from libs.iris_plus import *

from pdb import set_trace

def calculate_distance(x, y):    
    """calculates the minumum distance between y and each row of x, returning '0' if y is 
       between x's rows min and max values.
       Assumes we are interested in minimum diffence alog rows.
    Arguments:
        x -- 2-d numpy array of N by M where M is >=2 
             (if M was 1, the  this would just be np.abs(x-y)
        y -- either single value of 1-d numpy array of length N
    Returns.
        distances -- numpy 1-d array of length N.
    """
    lower_bound = np.min(x, axis=1)   # Extract the first column of x
    upper_bound = np.max(x, axis=1)   # Extract the second column of x
    
    below_lower = y < lower_bound
    above_upper = y > upper_bound

    distances = np.where(below_lower, lower_bound - y, 0)
    distances = np.where(above_upper, y - upper_bound, distances)
    return distances

def NME(X, Y, W = None, step1Only = False, x_range = False, premasked = False):   
    """ Nomalised Mean Error, step 1-3 (Kelley et al. 2013) and 
        over relative anomolie (Burton and Lampe et al. submitted). 
        
        Assumes that if X has more than one columne, then each column is a different 
        observtations, and we are interested in comparing each column in turn,
        and the minimum distance between or columns..
    Arguments:
        X -- 1-d or 2-d numpy array, equivlent to "observed" in Kelley et al. 2013
        Y -- 1-d numpy array equivlent to "simulation" in Kelley et al. 2013
        W -- 1-d numpy of length len(Y), representing how much to weight each comparison.
            Can be, e.g, grid cell area.
        step1Only -- boolean. If True, only return NME step 1
        x_range -- Boolean. If True, returns only results over range of obsvervations when
                X is 2-d
        premasked -- Boolean. This function will mask out nan values unnless this option is 
                set to True. Only to speed things up. If in doubt, leave as defulat of 'False'
    Returns.
        if X is 1-d and step1Only, just NME step 1 is returned as a single lonely value
        otherwise, a pandas datafrom of (index) NME 1, 2, 3 and relative anomolie and 
        columns for each observation and, if X is 2-d, combined observtations

    References
        Kelley DI, Prentice IC, Harrison SP, Wang H, Simard M, Fisher JB, Willis KO. 
        A comprehensive benchmarking system for evaluating global vegetation models. 
        Biogeosciences. 2013 May 17;10(5):3313-40.

        Burton C and Lampe S et al. "Is climate change driving an increase in global fires?" 
        submitted
    """ 
    if len(X.shape) > 1 and X.shape[1] > 1 and not x_range:
        axis = 0 if np.isscalar(Y) or X.shape[0] == Y.shape[0] else 1
        out_each = np.apply_along_axis(NME, axis=axis, arr=X, Y=Y)
        out_each = np.reshape(out_each, (out_each.shape[0], out_each.shape[2]))

        #X_range = np.array([np.min(X, axis=1),  np.max(X, axis=1)]).T
        out_all = NME(X, Y, step1Only, x_range = True)

        colnames = ['observation ' + str(i) for i in range(out_each.shape[1])] + ['All']
        out = np.append(out_each, np.array(out_all), axis = 1)
        out = pd.DataFrame(out, index=['NME1', 'NME2','NME3', 'NMEA'],  columns = colnames)
        return out
 
    if W is not None:
        X = X * W
        Y = Y * W

    if x_range:        
        if not premasked:
            mask = ~np.isnan(np.sum(X, axis = 1) + Y) 
            X = X[mask, :]
        def metric(z, y):
            disty = calculate_distance(z, y)
            distx = calculate_distance(z, np.mean(z))
            distx = result = list(map(lambda x: calculate_distance(z, x), np.mean(z, axis = 0)))
            
            return np.mean(np.abs(disty))/np.mean(np.abs(np.array(distx)))            
    else:
        if not premasked:
            mask = ~np.isnan(X + Y)
            X = X[mask]
        
        def metric(x, y):
            return np.sum(np.abs(x-y))/np.sum(np.abs(x - np.mean(x)))
    
    if not premasked and not np.isscalar(Y): Y = Y[mask]
   
    nme1 = metric(X, Y)
    
    if step1Only: return nme1


    def removeMean(x): return x - np.mean(x)
    X2 = removeMean(X)
    Y2 = removeMean(Y)
    nme2 = metric(X2, Y2)

    def removeVar(x): return x / np.mean(np.abs(x))
    X3 = removeVar(X2) 
    Y3 = removeVar(Y2)
    nme3 = metric(X3, Y3)
    
    def relativeAnom(x): 
        mu = np.mean(x)
        return (x - mu)/mu
    XA = removeVar(X) 
    YA = removeVar(Y)

    nmeA = metric(XA, YA)

    return pd.DataFrame(np.array([nme1, nme2, nme3, nmeA]), 
                        index=['NME1', 'NME2','NME3', 'NMEA'])
  
def NME_null(X, axis = 0, x_range = False, return_RR_sample = False, premasked = False, 
            *args, **kw):   
    """ Nomalised Mean Error mean and randomly resampled (Kelley et al. 2013) and 
        median (Burton et al. 2019) null model. 
        
        Assumes that if X has more than one columne, then each column is a different 
        observtations, and we are interested in comparing each column in turn,
        and the minimum distance between or columns..
    Arguments:
        X -- 1-d or 2-d numpy array, equivlent to "observed" in Kelley et al. 2013
        x_range -- Boolean. If True, returns only results over range of obsvervations when
                X is 2-d
        return_RR_sample -- Boolean. If True, returns all random resample results. 
                                     If False, returns summery.
        *args, **kw -- arugemts passed to NME.
    Returns.
        A pandas datafrom of (index) median, meanm randomly resampled mean score and 
        randomly resampled standard deviation. Columns for each observation and, if X is 2-d,
        combined observtations

    References
        Kelley DI, Prentice IC, Harrison SP, Wang H, Simard M, Fisher JB, Willis KO. 
        A comprehensive benchmarking system for evaluating global vegetation models. 
        Biogeosciences. 2013 May 17;10(5):3313-40.

        Burton C, Betts R, Cardoso M, Feldpausch TR, Harper A, Jones CD, Kelley DI, 
        Robertson E, Wiltshire A. Representation of fire, land-use change and vegetation 
        dynamics in the Joint UK Land Environment Simulator vn4. 9 (JULES). Geoscientific Model
        Development. 2019 Jan 9;12(1):179-93.
    """ 
    index_names = ['median null', 'mean null',
                   'Ranomly-resampled null - mean', 'Ranomly-resampled null - sdev']
    if len(X.shape) > 1 and X.shape[1] > 1 and not x_range:
        out_each = np.apply_along_axis(NME_null, axis=axis, arr=X)
        out_each = np.reshape(out_each, (out_each.shape[0], out_each.shape[2]))

        out_all = NME_null(X, x_range = True)
        colnames = ['observation ' + str(i) for i in range(out_each.shape[1])] + ['All']
        out = np.append(out_each, np.array(out_all), axis = 1)
        out = pd.DataFrame(out, index=index_names,  columns = colnames)
        
        return out

    if not premasked:
        if x_range:        
            mask = ~np.isnan(np.sum(X, axis = 1)) 
            X = X[mask, :]
        else:
            X = X[~np.isnan(X)]
    
    median_null = NME(X, np.median(X), x_range = x_range, step1Only = True, premasked = True)
    
    mean_null = 1
    
    def randonly_resample_null(X, XR):
        if x_range: XR =  np.random.choice(X.flatten(), size=X.shape[0])
        else: np.random.shuffle(XR)
        return NME(X, XR, x_range = x_range, step1Only = True, premasked = True)
        
    RR = np.empty(0)
    XR = X.copy()
    for i in range(0, 1000):
        RR = np.append(RR, randonly_resample_null(X, XR))
    
        if i > 10: 
            RR_mean0 = RR_mean
            RR_sdev0 = RR_sdev

        RR_mean = np.mean(RR)
        RR_sdev = np.std(RR)
        
        if i > 10: 
            if np.abs(RR_mean - RR_mean0) < 0.001 and np.abs(RR_sdev - RR_sdev0) < 0.001:
                break
    if return_RR_sample:
        return median_null, mean_null, RR_mean, RR_sdev, RR
    else:
        return pd.DataFrame(np.array([median_null, mean_null, RR_mean, RR_sdev]), 
                            index=index_names)


def NME_cube(X, Y, *args, **kw):
    weights = iris.analysis.cartography.area_weights(X).flatten()

    XD = X.data.flatten()
    YD = Y.data.flatten()

    mask = XD.mask | YD.mask | np.isnan(XD.data) | np.isnan(YD.data)
    mask = ~mask
    XD = XD[mask]
    YD = YD[mask]
    weights = weights[mask]
    
    return NME(XD, YD, W = weights, premasked = True)

def NME_null_cube(X, *args, **kw):
    weights = iris.analysis.cartography.area_weights(X).flatten()
    XD =  X.data.flatten()
    mask = XD.mask | np.isnan(XD.data)
    mask = ~mask
    XD = XD[mask]
    weights = weights[mask]
    return NME_null(XD, W = weights, premasked = True)
    
    
