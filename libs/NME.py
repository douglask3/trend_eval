import numpy as np
import pandas as pd

def calculate_distance(x, y):
    lower_bound = np.min(x, axis=1)   # Extract the first column of x
    upper_bound = np.max(x, axis=1)   # Extract the second column of x
    
    below_lower = y < lower_bound
    above_upper = y > upper_bound

    distances = np.where(below_lower, lower_bound - y, 0)
    distances = np.where(above_upper, y - upper_bound, distances)
    return distances

def NME(X, Y, step1Only = False, x_range = False, premasked = False):    
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
  
def NME_null(X, axis = 0, x_range = False, return_RR_sample = False):
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
    
