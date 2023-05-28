from main import *
from   libs.plot_AR6_hexagons    import *
import numpy  as np
import matplotlib.pyplot as plt
from pdb import set_trace
import os

import pymc  as pm
import arviz as az

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
    return prob

def calculate_distance(x, y):
    lower_bound = np.min(x, axis=1)   # Extract the first column of x
    upper_bound = np.max(x, axis=1)   # Extract the second column of x

    below_lower = y < lower_bound
    above_upper = y > upper_bound

    distances = np.where(below_lower, lower_bound - y, 0)
    distances = np.where(above_upper, y - upper_bound, distances)
    return distances




def NME(X, Y, step1Only = False, x_range = False):    
    if len(X.shape) > 1 and X.shape[1] > 1 and not x_range:
        axis = 0 if X.shape[0] == Y.shape[0] else 1
        out_each = np.apply_along_axis(NME, axis=axis, arr=X, Y=Y)
        out_each = np.reshape(out_each, (out_each.shape[0], out_each.shape[2]))

        #X_range = np.array([np.min(X, axis=1),  np.max(X, axis=1)]).T
        out_all = NME(X, Y, step1Only, x_range = True)

        colnames = ['model ' + str(i) for i in range(out_each.shape[1])] + ['All']
        out = np.append(out_each, np.array(out_all), axis = 1)
        out = pd.DataFrame(out, index=['NME1', 'NME2','NME3', 'NMEA'],  columns = colnames)
        return out
 
    if x_range:        
        mask = ~np.isnan(np.sum(X, axis = 1) + Y) 
        X = X[mask, :]
        def metric(x, y):
            disty = calculate_distance(x, y)
            distx = calculate_distance(x, np.mean(x))
            return(np.sum(np.abs(disty))/np.sum(np.abs(distx)))
    else:
        mask = ~np.isnan(X + Y)
        X = X[mask]
        
        def metric(x, y):
            return np.sum(np.abs(x-y))/np.sum(np.abs(x - np.mean(x)))
    Y = Y[mask]

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
    
#def NME_null(X):
    

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
        
        prob = find_and_compare_gradients(Y, X, tracesID_save, n_itertations = n_itertations)
        
        #NME_temp_file =
        #    'temp/eval_trendsburnt_area_u-cc669-REGION---GIC---1996_2020-NME-nObs_3.npy'
        NME = NME(X, Y)

        #nme_median, nme_mean, nme_RR = NME_null(X)
        return ar6_region_code, value, prob, NME

    result = list(map(lambda item: trend_prob_for_region(item[0], item[1]), \
                                    ar6_regions.items()))
    result = list(filter(lambda x: x is not None, result))

    plot_AR6_hexagons(result, resultID = 2, colorbar_label = 'Gradient Overlap')
    plt.show()
  

    
#"/"

#"GFED4.nc", "MCD45.nc", "meris_v2.nc"
