from main import *
from libs.plot_AR6_hexagons import *
from libs.NME import *
from libs.flatten_list import *
from libs.time_series_comparison import *
import numpy  as np
import matplotlib.pyplot as plt
from pdb import set_trace
import os
import pickle
    
def trend_prob_for_region(ar6_region_code, value, observations_names, filename_model, mod_scale,
                          obs_scale, year_range, n_itertations, tracesID):# in ar6_regions:   
    
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
                     '_'.join([str(year) for year in year_range]) + \
                     '-model_scale_' +  str(mod_scale) + \
                     '-obs_scale_' + '_'.join([str(i) for i in obs_scale])
    
    Y_temp_file = tracesID_save + '-Y' + '.npy'
    X_temp_file = tracesID_save + '-X' + '.npy'
    
    if os.path.isfile(Y_temp_file) and os.path.isfile(X_temp_file) and False: 
        Y = np.load(Y_temp_file)
        X = np.load(X_temp_file)
    else :
        Y, X = read_all_data_from_netcdf(filename_model, filenames_observation, 
                                         time_series = year_range, check_mask = False,
                                         subset_function = subset_functions, 
                                         subset_function_args = subset_function_args)
        Y = Y * mod_scale
        X = X * obs_scale
        np.save(Y_temp_file, Y)  
        np.save(X_temp_file, X)
        
    gradient_compare = find_and_compare_gradients(Y, X, tracesID_save, 
                                                  n_itertations = n_itertations)
        
    nme = NME(X, Y)        
    nme_null = NME_null(X)
    obs_mean = np.nanmean(X, axis = 0)

    Yspread = np.reshape(np.repeat(Y, X.shape[1]), X.shape)
    Yspread[np.isnan(X)] = np.nan
    mod_mean = np.nanmean(Yspread, axis = 0)
        
        
    out =  ar6_region_code, value, obs_mean, mod_mean, nme
    
    out = [out[0], out[1]] + list(np.concatenate(out[2:4])) + list(out[4].values.flatten()) + \
          list(nme_null.values.flatten()) + \
          list(gradient_compare.values.flatten())
    
    return(out)

def eval_trends_over_AR6_regions(filenames_observation, observations_names = None,
                                 output_file = '', grab_output = False,
                                 *args, **kw):
    
    if grab_output and os.path.isfile(output_file): 
        return pd.read_csv(output_file, index_col = 0)
    ar6_regions =  regionmask.defined_regions.ar6.land.region_ids

    if observations_names is None:
        observations_names = [str(i) for i in range(len(filenames_observation))] + ['All']
    
    NME_obs = observations_names + ['All']
        
    null_models = ['Median', 'Mean', 'Randomly-resampled mean', 'Randomly-resampled - sd']
    index = ['Region Code', 'Region ID'] + \
            ['observation ' + str(i) for i in observations_names] + \
            ['simulations ' + str(i) for i in observations_names] + \
            [['NME ' + j + ' obs. ' + i for i in NME_obs] for j in ['1', '2', '3', 'A']]  + \
            [[j + 'Null model obs. ' + i for i in NME_obs] for j in null_models] + \
            ['Gradient overlap', 'Obs trend - 10%', 'Obs trend - 90%', 
                                  'Mod trend - 10%', 'Mod trend - 90%']
    index = flatten_list(index)
    
    result = list(map(lambda item: trend_prob_for_region(item[0], item[1], *args, **kw), \
                                    ar6_regions.items()))
    result = list(filter(lambda x: x is not None, result))
    
    result = pd.DataFrame(np.array(result).T, index = index, columns = np.array(result)[:,0])
    result.to_csv(output_file)
    
    return result

def NME_by_obs(obs_name):
    
    X = result.loc['observation ' + obs_name].values.astype(float)
    Y = result.loc['simulations ' + obs_name].values.astype(float)
    
    nme = NME(X, Y)        
    nme_null = NME_null(X)
    
    return pd.DataFrame(np.append(nme_null, nme), index = np.append(nme_null.index, nme.index))
    
def makeDir(directory):
    try:
        os.mkdir(directory)
    except:
        pass

if __name__=="__main__":    
    filename_model = "/scratch/hadea/isimip3a/u-cc669_isimip3a_fire/20CRv3-ERA5_obsclim/jules-vn6p3_20crv3-era5_obsclim_histsoc_default_burntarea-total_global_monthly_1901_2021.nc"

    dir_observation = "/data/dynamic/dkelley/fireMIPbenchmarking/data/benchmarkData/"
    filenames_observation = ["ISIMIP3a_obs/GFED4.1s_Burned_Fraction.nc", \
                             "ISIMIP3a_obs/FireCCI5.1_Burned_Fraction.nc", \
                             "ISIMIP3a_obs/GFED500m_Burned_Percentage.nc"]
    filenames_observation = [dir_observation + file for file in filenames_observation]
    
    observations_names = ['GFED4.1s', 'FireCCI5.1', 'GFED500m']

    year_range = [1996, 2020]
    n_itertations = 1000
    tracesID = 'burnt_area_u-cc669'
    mod_scale = 1.0/100.0
    obs_scale = [1.0, 1.0, 1.0/100.0]

    units = '1'
    output_file = 'outputs/trend_burnt_area_metric_results.csv'
    output_maps = 'outputs/burnt_area/'

    result = eval_trends_over_AR6_regions(filenames_observation, observations_names,
                                          output_file, True,
                                          observations_names, filename_model, 
                                          mod_scale, obs_scale,
                                          year_range, n_itertations, tracesID)

    subset_functions = [sub_year_range, annual_average]
    subset_function_args = [{'year_range': year_range},
                            {'annual_aggregate' : iris.analysis.SUM}]
    
    def open_compare_obs_mod(filename_obs, scale, name_obs, output_maps, openOnly = False):
        makeDir(output_maps)
        output_maps = output_maps + '/' + name_obs + '/'
        makeDir(output_maps)
        print(name_obs)
        def readFUN(filename, subset_function_args):
            return read_variable_from_netcdf(filename,subset_function = subset_functions, 
                                             make_flat = False, 
                                             subset_function_args = subset_function_args)
        

        X, year_range = readFUN(filename_obs, subset_function_args)
        subset_function_args[0]['year_range'] = year_range
        Y, nn = readFUN(filename_model, subset_function_args)
        if mod_scale is not None: Y.data = Y.data * mod_scale
        if scale is not None: X.data = X.data * scale
        if units is not None: Y.units = units
        if units is not None: X.units = units
        
        if openOnly: return X, Y
        X_filename = output_maps + 'observation.nc'
        Y_filename = output_maps + 'simulation.nc'
        iris.save(X, X_filename)
        iris.save(Y, Y_filename)
        
        nme = NME_cube(X, Y)
        nme_null = NME_null_cube(X)
        scores = pd.DataFrame(np.append(nme_null, nme), 
                              index = np.append(nme_null.index, nme.index))
        
        return scores, X, Y, year_range
    
    temp_file_path = 'temp/eval_trends_nme_over_obs_pickle-' + tracesID + '.pkl'

    if os.path.isfile(temp_file_path):
        with open(temp_file_path, "rb") as file:
            nme_over_obs = pickle.load(file)
    else:
        nme_over_obs = list(map(lambda x, y, z: open_compare_obs_mod(x, y, z, output_maps), 
                                filenames_observation, obs_scale, observations_names))
        
        with open(temp_file_path, "wb") as file:
            pickle.dump(nme_over_obs, file)
    
    year_range = np.array([out[3] for out in nme_over_obs])
    year_range = [np.min(year_range[:,0]), np.max(year_range[:,1])]
    subset_function_args[0]['year_range'] = year_range

    XYs = list(map(lambda x, y, z: open_compare_obs_mod(x, y, z, output_maps, openOnly = True), 
                                filenames_observation, obs_scale, observations_names))
    
    X = [xy[0] for xy in XYs]
    cubes = []    
    for i in range(len(X)):       
        coord = iris.coords.DimCoord(i, "realization")  
        cube = X[0].copy()      
        cube.add_aux_coord(coord)
        cube.data = X[i].data.astype(np.float32)
        cubes.append(cube)
    
    X = iris.cube.CubeList(cubes).merge_cube()    
    Y = XYs[0][1]
    output_maps = output_maps + '/All/'
    makeDir(output_maps)

    X_filename = output_maps + 'observation.nc'
    Y_filename = output_maps + 'simulation.nc'
    iris.save(X, X_filename)
    iris.save(Y, Y_filename)

    set_trace()
    
    #open_compare_obs_mod(filenames_observation[0], observations_names[0], output_maps)
    nme_obs = list(map(NME_by_obs, observations_names))
    
    
    
    #plot_AR6_hexagons(result, resultID = 41, colorbar_label = 'Gradient Overlap')

