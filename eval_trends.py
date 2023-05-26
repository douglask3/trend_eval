from main import *
import numpy  as np
import matplotlib.pyplot as plt
from pdb import set_trace

if __name__=="__main__":
    filename_model = "/scratch/hadea/isimip3a/u-cc669_isimip3a_fire/20CRv3-ERA5_obsclim/jules-vn6p3_20crv3-era5_obsclim_histsoc_default_burntarea-total_global_monthly_1901_2021.nc"

    dir_observation = "/data/dynamic/dkelley/fireMIPbenchmarking/data/benchmarkData/"
    filenames_observation = ["ISIMIP3a_obs/GFED4.1s_Burned_Fraction.nc", \
                             "ISIMIP3a_obs/FireCCI5.1_Burned_Fraction.nc", \
                             "ISIMIP3a_obs/GFED500m_Burned_Percentage.nc"]
    filenames_observation = [dir_observation + file for file in filenames_observation]

    year_range = [1996, 2020]

    subset_functions = [sub_year_range, ar6_region, make_time_series]
    subset_function_args = [{'year_range': year_range},
                            {'region_code' : ['NWS', 'NSA', 'SAH', 'WAF']}, 
                            {'annual_aggregate' : iris.analysis.SUM}]
        

    Y, X = read_all_data_from_netcdf(filename_model, filenames_observation, 
                                     time_series = year_range, check_mask = False,
                                     subset_function = subset_functions, 
                                     subset_function_args = subset_function_args)
    set_trace()
#"/"

#"GFED4.nc", "MCD45.nc", "meris_v2.nc"
