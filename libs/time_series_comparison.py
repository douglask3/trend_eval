import pymc  as pm
import arviz as az

import numpy  as np
import pandas as pd
from pdb import set_trace
import os

def find_and_compare_gradients(Y0, X0, tracesID_save, *args, **kw):
    
    Y = np.log(Y0 + 0.000000000000001)
    X = np.log(X0 + 0.000000000000001)
    
    Y_grad = run_time_series_regression(Y, tracesID_save + '-Y', *args, **kw)
    X_grad = run_time_series_regression(X, tracesID_save + '-X', *args, **kw)

    def get_values(trace, var): 
        out = trace.posterior[var].values
        return out.reshape((-1,) + out.shape[2:])
    
    betaY = get_values(Y_grad, 'beta')
    betaX = get_values(X_grad, 'beta')
    alphaY =  get_values(Y_grad, 'y0')
    alphaX =  get_values(X_grad, 'y0')

    prob = compare_gradients(betaY, betaX)
    
    outFile = tracesID_save + '-XY.csv'
    outarr = np.vstack((Y0, X0.T)).T
    np.savetxt(outFile, outarr, delimiter=',')

    outFile = tracesID_save + '-TRACE.csv'
    
    outarr = np.vstack((betaY, alphaY, betaX, alphaX.T)).T
    np.savetxt(outFile, outarr, delimiter=',')
    
    out = np.concatenate(([prob], np.percentile(betaY, [15, 95]), 
                                  np.percentile(betaX, [15, 95])))
    out = pd.DataFrame(out, index=['Gradient Overlap', 
                                   'Model coefficant - 10%', 'Model coefficant - 90%', 
                                   'Observed coefficant - 10%', 'Observed coefficant - 90%'])
    
    return out



def run_time_series_regression(ys, tracesID_save, grab_trace = True, save_trace = True, n_itertations = 100):
    
    tracesID_save = tracesID_save + '_' + str(ys.shape[0]) 
    if len(ys.shape) > 1:  tracesID_save = tracesID_save + '_' + str(ys.shape[1]) 
    tracesID_save = tracesID_save + 'n_itertations' + '-' + str(n_itertations) + '.nc'
    
    if grab_trace:
        if os.path.isfile(tracesID_save): return az.from_netcdf(tracesID_save)
        print(tracesID_save)
    tm = np.arange(0, len(ys))
    with pm.Model() as model:  # model specifications in PyMC are wrapped in a with-statement
        # Define priors
        epsilon = pm.LogNormal("epsilon", 0, 10)
        ys_no_nan = ys.T[~np.isnan(ys.T)]
        beta = pm.Normal("beta", 0, (np.max(ys_no_nan) - np.min(ys_no_nan))/(\
                                     np.max(tm) - np.min(tm)))

        if len(ys.shape) == 1: 
            y0 = pm.Normal("y0", np.mean(ys_no_nan), sigma=np.std(ys_no_nan))
            prediction = y0 + beta * tm
        else:

            y0 = pm.Normal('y0', np.mean(ys_no_nan), sigma=np.std(ys_no_nan), 
                           shape = ys.shape[1])
            
            
            tm_in = np.empty(0)
            y0_id = np.empty(0)
            for i in range(ys.shape[1]):
                tm_in = np.append(tm_in, tm[~np.isnan(ys[:,i])])
                y0_id = np.append(y0_id, np.repeat(i, np.sum(~np.isnan(ys[:,i]))))
        
            prediction = y0[[int(id) for id in y0_id]] + beta * tm_in

        # Define likelihood
        likelihood = pm.Normal("mod", mu=prediction, sigma=epsilon, observed=ys_no_nan)
        
        # Inference!
        # draw n_itertations posterior samples using NUTS sampling
        try:
            trace = pm.sample(n_itertations, return_inferencedata=True)
        except:
            print("trace gone wrong")
            set_trace()
    if save_trace:
        trace.to_netcdf(tracesID_save)
    return(trace)

def compare_gradients(beta_Y, beta_X):
    beta_Y = beta_Y.flatten()
    beta_X = beta_X.flatten()
    
    min_beta = np.min(np.append(beta_Y, beta_X))
    max_beta = np.max(np.append(beta_X, beta_X))
    nbins = int(np.ceil(np.sqrt(beta_X.size))) 

    bins = np.linspace(min_beta, max_beta, nbins)
    def normHist(beta) :
        out = np.histogram(beta, bins)[0]
        return out / np.max(out)

    distY = normHist(beta_Y)
    distX = normHist(beta_X)
    distZ = np.min(np.array([distY, distX]), axis = 0)
    
    return np.sqrt(np.sum(distY*distZ)/np.sum(distY*distY))

