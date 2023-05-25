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


def MaxEnt_on_prob(x, mu):
    """calculates the log-transformed continuous logit likelihood for x given mu when x 
       and mu are probabilities between 0-1. 
       Works with tensor variables.   
    Arguments:
        x -- x in P(x|mu). tensor 1-d array
	mu -- mu in P(x|mu). tensor 1-d array
    Returns:
        1-d tensor array of liklihoods.
    """
    mu = tt.switch(
        tt.lt(mu, 0.0000000000000000001),
        0.0000000000000000001, mu)
    return tt.log(mu**x) + tt.log((1-mu)**(1.0-x))
    

def fire_model(betas, X, inference = False):
    """base fire model which takes indepedant variables and coefficants. 
        At the moment, just a linear model fed through a logistic function to convert to 
        burnt area/fire probablity. But we'll adapt that.   
    Arguments:
        betas -- numpy or tensor 1-d array of coefficants in linear model
                y = betas[1] + X[:,1] + betas[2] + X[:,2] + .....
	X -- numpy or tensor 2d array of indepenant variables, each columne a different 
                variable, no. columns (no. variables) is same as length of betas.
	inference -- boolean. If True, then used in bayesian inference and uses tensor maths. 
			      If False, used in normal mode or prior/posterior sampling and 
                                uses numpy.
    Returns:
        numpy or tensor (depdaning on 'inference' option) 1 d array of length equal to 
	no. rows in X of burnt area/fire probabilities.
    """
    if inference: 
        numPCK =  __import__('aesara').tensor
    else:
        numPCK = __import__('numpy')
    
    y = numPCK.dot(X, betas)

    BA = 1.0/(1.0 + numPCK.exp(-y))
    
    return BA
   

def fit_MaxEnt_probs_to_data(Y, X, niterations, 
                             out_dir = 'outputs/', filename = '', grab_old_trace = True):
    """ Bayesian inerence routine that fits independant variables, X, to dependant, Y.
        Based on the MaxEnt solution of probabilities. 
    Arguments:
        Y-- dependant variable as numpy 1d array
	X -- numpy 2d array of indepenant variables, each columne a different variable
	niterations -- number of iterations per chain when sampling the postior during 
                NUTS inference 
		(note default chains is normally 2 and is set by *args or **kw)
	out_dir --string of path to output location. This is where the traces netcdf file 
                will be saved.
		Defauls is 'outputs'.
	filename -- string of the start of the traces output name. Detault is blank. 
		Some metadata will be saved in the filename, so even blank will 
                save a file.
	grab_old_trace -- Boolean. If True, and a filename starting with 'filename' and 
                containing some of the same setting (saved in filename) exists,  it will open 
                and return this rather than run a new one. Not all settings are saved for 
                identifiation, so if in doubt, set to 'False'.
	*args, **kw -- arguemts passed to 'pymc.sample'

    Returns:
        pymc traces, returned and saved to [out_dir]/[filneame]-[metadata].nc
    """

    trace_file = out_dir + '/' + filename + '-nvariables_' + '-ncells_' + str(X.shape[0]) + \
                str(X.shape[1]) + '-niterations_' + str(niterations) + '.nc'
    
    if os.path.isfile(trace_file) and grab_old_trace: 
        return az.from_netcdf(trace_file)

    with pm.Model() as max_ent_model:
        
        #betas = pm.Normal("betas", mu=0, sigma=1, shape = X.shape[1])
        betas = pm.Normal('betas', mu = 0, sigma = 1, shape = X.shape[1], 
                          initval =np.repeat(0.5, X.shape[1]))
        
        mu = fire_model(betas, X, inference = True)
        
        #y = pm.math.sum(betas * X)
        #y = beta1 * X[:,0] + beta2 * X[:,1] + beta3# * X[:,3]
        
        
        error = pm.DensityDist("error", mu, logp = MaxEnt_on_prob, observed = Y)
        try:
            trace = pm.sample(niterations, return_inferencedata=True, cores = 1)
        except:
            browser()
        
        trace.to_netcdf(trace_file)
    return trace


if __name__=="__main__":
    dir = "../ConFIRE_attribute/isimip3a/driving_data/GSWP3-W5E5-20yrs/Brazil/AllConFire_2000_2009/"
    y_filen = "GFED4.1s_Burned_Fraction.nc"
    
    x_filen_list=["precip.nc", "lightn.nc", "crop.nc", "humid.nc",#"vpd.nc", "csoil.nc", 
                  "lightn.nc", "rhumid.nc", "cveg.nc", "pas.nc", "soilM.nc", 
                   "totalVeg.nc", "popDens.nc", "trees.nc"]

    niterations = 100
    sample_for_plot = 20

    levels = [0, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100] 
    cmap = 'OrRd'

    months_of_year = [7]

    Y, X, lmask = read_all_data_from_netcdf(y_filen, x_filen_list, 
                                           add_1s_columne = True, dir = dir, 
                                           subset_function = sub_year_months, 
                                             subset_function_args = {'months_of_year': months_of_year})
    
    filename = '_'.join([file[:-3] for file in x_filen_list]) + '-Month_' + \
               '_'.join([str(mn) for mn in months_of_year])
    
    trace = fit_MaxEnt_probs_to_data(Y, X, filename = filename,  niterations = niterations)
    
    Obs = read_variable_from_netcdf(y_filen, dir, subset_function = sub_year_months, 
                                     subset_function_args = {'months_of_year': months_of_year})

    def select_post_param(name): 
        out = trace.posterior[name].values
        return out.reshape((-1, out.shape[-1]))

    def sample_model(i): 
        betas =select_post_param('betas')[i,:]
        return fire_model(betas, X)

    nits = np.prod(trace.posterior['betas'].values.shape[0:2])
    idx = range(0, nits, int(np.floor(nits/sample_for_plot)))

    Sim = np.array(list(map(sample_model, idx)))
    Sim = np.percentile(Sim, q = [10, 90], axis = 0)
    
    def insert_sim_into_cube(x):
        Pred = Obs.copy()
        pred = Pred.data.copy().flatten()
        
        pred[lmask] = x
        Pred.data = pred.reshape(Pred.data.shape)
        return(Pred)

    def plot_map(cube, plot_name, plot_n):
        plot_annual_mean(cube, levels, cmap, plot_name = plot_name, scale = 100*12, 
                     Nrows = 1, Ncols = 3, plot_n = plot_n)
  
    plot_map(Obs, "Observtations", 1)
    plot_map(insert_sim_into_cube(Sim[0,:]), "Simulation - 10%", 2)
    plot_map(insert_sim_into_cube(Sim[1,:]), "Simulation - 90%", 3)
    
    set_trace()
