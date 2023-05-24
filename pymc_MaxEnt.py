from main import *

import os
from   io     import StringIO
import numpy  as np
import math

import pymc  as pm
from   aesara import tensor as tt

import matplotlib.pyplot as plt
import re

import sys
import arviz as az
#import multiprocessing as mp
#mp.set_start_method('forkserver')

def tt_sigmoid(x): 
    return 1.0/(1.0 + tt.exp(-x))


def np_sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def np_MaxEnt(x, mu):
    return np.log(np_sigmoid(mu)**x *(1-np_sigmoid(mu)**(1.0-x)))

def MaxEnt_on_prob(x, mu):
    '''return tt.sw1itch(
        tt.lt(x, -150),
        -p0,
        -(1.0 - p0) *(1.0/(sigma * 2.506))*tt.exp(-0.5 * ((x-mu)/sigma)**2)
    )
    '''
    
    return np.log(np_sigmoid(mu)**x *(1-np_sigmoid(mu)**(1.0-x)))
    


def fit_MaxEnt_probs_to_data(Y, X, niterations):
    with pm.Model() as fire_model:
        
        #betas = pm.Normal("betas", mu=0, sigma=1, shape = X.shape[1])
        betas = pm.Normal('betas', mu = 0, sigma = 1, shape = X.shape[1])
        mu = pm.math.dot(X, betas)
        
        #y = pm.math.sum(betas * X)
        #y = beta1 * X[:,0] + beta2 * X[:,1] + beta3# * X[:,3]
        
        
        error = pm.DensityDist("error", mu, logp = MaxEnt_on_prob, observed = Y)
        trace = pm.sample(niterations, return_inferencedata=True, cores = 1)
    return trace


if __name__=="__main__":
    dir = "../ConFIRE_attribute/isimip3a/driving_data/GSWP3-W5E5-20yrs/Brazil/AllConFire_2000_2009/"
    y_filen = "GFED4.1s_Burned_Fraction.nc"
    
    x_filen_list=["precip.nc", "lightn.nc", "crop.nc"]#, "humid.nc","vpd.nc", "csoil.nc", 
                  #"lightn.nc", "rhumid.nc", "cveg.nc", "pas.nc", "soilM.nc", 
                  # "totalVeg.nc", "popDens.nc", "trees.nc"]

    niterations = 100

    Y, X, lmask = read_all_data_from_netcdf(y_filen, x_filen_list, 
                                           add_1s_columne = True, dir = dir, 
                                           subset_function = sub_year_months, 
                                           months_of_year = [6, 7, 8])
    
    Obs = read_variable_from_netcdf(y_filen, dir, subset_function = sub_year_months, 
                                     months_of_year = [6, 7, 8])

    Pred = Obs.copy()
    pred = Pred.data.copy().flatten()
    trace = fit_MaxEnt_probs_to_data(Y, X, niterations = niterations)

    set_trace()
    def select_post_param(name): return np.ndarray.flatten(trace.posterior[name].values)

    def sample_model(i): 
        return select_post_param('T0')[i] + select_post_param('beta')[i] *coverChange

    idx    = range(0, len(select_post_param('beta1')), 10)
    T0_m   = select_post_param('beta1').mean()
    beta_m = select_post_param('beta2').mean()

    
    set_trace()
