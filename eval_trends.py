from main import *
import numpy  as np
import matplotlib.pyplot as plt
from pdb import set_trace
import os

import pymc  as pm
import arviz as az

def run_time_series_regression(ys, tracesID_save, grab_trace = True, n_itertations = 100):
    
    if grab_trace:
        tracesID_save = tracesID_save + '_' + str(ys.shape[0]) 
        if len(ys.shape) > 1:  tracesID_save = tracesID_save + '_' + str(ys.shape[1]) 
        tracesID_save = tracesID_save + 'n_itertations' + '-' + str(n_itertations) + '.nc'
        if os.path.isfile(tracesID_save): return az.from_netcdf(tracesID_save)
        print(tracesID_save)
    tm = np.arange(0, len(ys))
    with pm.Model() as model:  # model specifications in PyMC are wrapped in a with-statement
        # Define priors
        epsilon = pm.LogNormal("epsilon", 0, 10)
        ys_no_nan = ys[~np.isnan(ys)]
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
        likelihood = pm.Normal("mod", mu=prediction, sigma=epsilon, observed=ys)
    
        # Inference!
        # draw n_itertations posterior samples using NUTS sampling
        try:
            trace = pm.sample(n_itertations, return_inferencedata=True)
        except:
            set_trace()
    if grab_trace:
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
    

    
def find_and_compare_gradients(Y, X, tracesID_save, *args, **kw):
    
    Y = np.log(Y + 0.000000000000001)
    X = np.log(X + 0.000000000000001)
    set_trace()
    Y_grad = run_time_series_regression(Y, tracesID_save + '-Y', *args, **kw)
    X_grad = run_time_series_regression(X, tracesID_save + '-X', *args, **kw)
    prob = compare_gradients(Y_grad.posterior['beta'].values,
                             X_grad.posterior['beta'].values)
    return prob

if __name__=="__main__":
    filename_model = "/scratch/hadea/isimip3a/u-cc669_isimip3a_fire/20CRv3-ERA5_obsclim/jules-vn6p3_20crv3-era5_obsclim_histsoc_default_burntarea-total_global_monthly_1901_2021.nc"

    dir_observation = "/data/dynamic/dkelley/fireMIPbenchmarking/data/benchmarkData/"
    filenames_observation = ["ISIMIP3a_obs/GFED4.1s_Burned_Fraction.nc", \
                             "ISIMIP3a_obs/FireCCI5.1_Burned_Fraction.nc", \
                             "ISIMIP3a_obs/GFED500m_Burned_Percentage.nc"]
    filenames_observation = [dir_observation + file for file in filenames_observation]

    year_range = [1996, 2020]
    ar6_regions =  regionmask.defined_regions.ar6.land.region_ids
    n_itertations = 1000
    tracesID = 'burnt_area_u-cc669'

    
    def trend_prob_for_region(ar6_region_code, value):# in ar6_regions:   
     
        if not isinstance(ar6_region_code, str): return 
        if ar6_region_code == 'EAN' or ar6_region_code == 'WAN': return
        if len(ar6_region_code) > 5: return
        print(ar6_region_code)
        subset_functions = [sub_year_range, ar6_region, make_time_series]
        subset_function_args = [{'year_range': year_range},
                            {'region_code' : [ar6_region_code]}, 
                            {'annual_aggregate' : iris.analysis.SUM}]

        tracesID_save = 'temp/eval_trends' + tracesID + '-' + \
                            '_'.join(ar6_region_code) + '-' + \
                            '_'.join([str(year) for year in year_range])
        
        
        Y_temp_file = tracesID_save + '-Y' + '.npy'
        X_temp_file = tracesID_save + '-X' + '.npy'
        if os.path.isfile(Y_temp_file) and os.path.isfile(X_temp_file): 
            Y = np.load(Y_temp_file)
            X = np.load(X_temp_file)
        else :
            Y, X = read_all_data_from_netcdf(filename_model, filenames_observation, 
                                             time_series = year_range, check_mask = False,
                                             subset_function = subset_functions, 
                                             subset_function_args = subset_function_args)
            np.save(Y_temp_file, Y)  
            np.save(X_temp_file, X)
        
        prob = find_and_compare_gradients(Y, X, tracesID_save, n_itertations = n_itertations)
        return ar6_region_code, value, prob

    result = list(map(lambda item: trend_prob_for_region(item[0], item[1]), \
                                    ar6_regions.items()))
    result = list(filter(lambda x: x is not None, result))
    set_trace()
#"/"

#"GFED4.nc", "MCD45.nc", "meris_v2.nc"
