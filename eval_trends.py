from main import *
from   libs.plot_AR6_hexagons    import *
from   libs.NME    import *
import numpy  as np
import matplotlib.pyplot as plt
from pdb import set_trace
import os

import pymc  as pm
import arviz as az

def flatten_list(lst):
    flattened = []
    for item in lst:
        if isinstance(item, str):
            flattened.append(item)
        elif isinstance(item, list):
            flattened.extend(flatten_list(item))
    return flattened

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
    
    output_file = 'outputs/trend_burnt_area_metric_results.csv'
    
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
                            'REGION---' + ar6_region_code + '---' + \
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
        
        gradient_compare = find_and_compare_gradients(Y, X, tracesID_save, 
                                                      n_itertations = n_itertations)
        
        #NME_temp_file =
        #    'temp/eval_trendsburnt_area_u-cc669-REGION---GIC---1996_2020-NME-nObs_3.npy'
        nme = NME(X, Y)        
        nme_null = NME_null(X)
        obs_mean = np.nanmean(X, axis = 0)

        Yspread = np.reshape(np.repeat(Y, X.shape[1]), X.shape)
        Yspread[np.isnan(X)] = np.nan
        mod_mean = np.nanmean(Yspread, axis = 0)
        
        
        out =  ar6_region_code, value, obs_mean, mod_mean, nme
        out = [out[0], out[1]] + list(np.concatenate(out[2:4])) + \
               list(out[4].values.flatten()) + list(nme_null.values.flatten()) + \
               list(gradient_compare.values.flatten())
        
        return(out)

    models = [str(i) for i in range(len(filenames_observation))] + ['All']
    null_models = ['Median', 'Mean', 'Randomly-resampled mean', 'Randomly-resampled - sd']
    index = ['Region Code', 'Region ID'] + \
            ['observation ' + str(i) for i in range(len(filenames_observation))] + \
            ['simulations ' + str(i) for i in range(len(filenames_observation))] + \
            [['NME ' + j + ' obs. ' + i for i in models] for j in ['1', '2', '3', 'A']]  + \
            [[j + 'Null model obs. ' + i for i in models] for j in null_models] + \
            ['Gradient overlap', 'Obs trend - 10%', 'Obs trend - 90%', 
                                  'Mod trend - 10%', 'Mod trend - 90%']
    index = flatten_list(index)
    
    result = list(map(lambda item: trend_prob_for_region(item[0], item[1]), \
                                    ar6_regions.items()))
    result = list(filter(lambda x: x is not None, result))

    result = pd.DataFrame(np.array(result).T, index = index, columns = np.array(result)[:,0])
    result.to_csv(output_file)
    plot_AR6_hexagons(result, resultID = 2, colorbar_label = 'Gradient Overlap')
    plt.show()
  

    
#"/"

#"GFED4.nc", "MCD45.nc", "meris_v2.nc"
