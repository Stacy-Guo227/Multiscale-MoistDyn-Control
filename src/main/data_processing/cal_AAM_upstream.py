import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
from functools import partial
import multiprocessing
import create_polar_info      # .py file

if __name__=='__main__':
    IVT_fpath  = input('Please assign IVT file path:')
    wtab_all   = pd.read_csv(f"../../../data/processed/weather_table_self/all_withlv.csv")       # weather table
    datelist   = wtab_all['yyyymmdd']
    
    # Luzon
    regime_condA_luzon = ((wtab_all['ERA5_all_lv0']>=-4)&
                          (wtab_all['ERA5_all_lv0']<2)&
                          (wtab_all['ERA5_all_lv1']>=-4)&
                          (wtab_all['ERA5_all_lv1']<0))
    regime_condB_luzon = ((wtab_all['ERA5_all_lv0']>=0)&
                          (wtab_all['ERA5_all_lv0']<4)&
                          (wtab_all['ERA5_all_lv1']>=-6)&
                          (wtab_all['ERA5_all_lv1']<-4))
    regime_condC_luzon = ((wtab_all['ERA5_all_lv0']>=2)&
                          (wtab_all['ERA5_all_lv0']<4)&
                          (wtab_all['ERA5_all_lv1']>=-4)&
                          (wtab_all['ERA5_all_lv1']<-2))
    regime_index_luzon = wtab_all[regime_condA_luzon|regime_condB_luzon|regime_condC_luzon].index
    
    luzon_lon, luzon_lat = (115, 118), (13, 16)
    luzon_r, luzon_theta = create_polar_info.cal_for_all_dates(datelist=datelist, func=create_polar_info.Get_r_theta, func_config={'IVT_fpath':IVT_fpath, 'env':'IVT', 'lon_range':luzon_lon, 'lat_range':luzon_lat})
    luzon_dict = {'lon_range':luzon_lon, 'lat_range':luzon_lat, 'r':luzon_r, 'theta':luzon_theta}
    np.save('../../../data/processed/weather_table_self/AAM_other_area/luzon_upstream.npy', luzon_dict)
    print(f"Upstream polar info. of Luzon saved.")
    
    # Eastern Coast of Bay of Bengal
    regime_condA_bob = ((wtab_all['ERA5_all_lv0']>=-2)&
                          (wtab_all['ERA5_all_lv0']<2)&
                          (wtab_all['ERA5_all_lv1']>=-4)&
                          (wtab_all['ERA5_all_lv1']<0))
    regime_index_bob = wtab_all[regime_condA_bob].index
    bob_lon, bob_lat = (88, 91), (15, 18)
    bob_r, bob_theta = create_polar_info.cal_for_all_dates(datelist=datelist, func=create_polar_info.Get_r_theta, func_config={'IVT_fpath':IVT_fpath, 'env':'IVT', 'lon_range':bob_lon, 'lat_range':bob_lat})
    bob_dict = {'lon_range':bob_lon, 'lat_range':bob_lat, 'r':bob_r, 'theta':bob_theta}
    np.save('../../../data/processed/weather_table_self/AAM_other_area/bob_upstream.npy', bob_dict)
    print(f"Upstream polar info. of BoB saved.")
    
    # Western Ghats
    regime_condA_india = ((wtab_all['ERA5_all_lv0']>=-2)&
                          (wtab_all['ERA5_all_lv0']<2)&
                          (wtab_all['ERA5_all_lv1']>=-4)&
                          (wtab_all['ERA5_all_lv1']<0))
    regime_index_india = wtab_all[regime_condA_india].index
    india_lon, india_lat = (70, 72), (12, 17)
    india_r, india_theta = create_polar_info.cal_for_all_dates(datelist=datelist, func=create_polar_info.Get_r_theta, func_config={'IVT_fpath':IVT_fpath, 'env':'IVT', 'lon_range':india_lon, 'lat_range':india_lat})
    india_dict = {'lon_range':india_lon, 'lat_range':india_lat, 'r':india_r, 'theta':india_theta}
    np.save('../../../data/processed/weather_table_self/AAM_other_area/india_upstream.npy', india_dict)
    print(f"Upstream polar info. of Western Ghats saved.")
    
    # southern New Guinea
    regime_condA_nguinea = ((wtab_all['ERA5_all_lv0']>=-4)&
                            (wtab_all['ERA5_all_lv0']<2)&
                            (wtab_all['ERA5_all_lv1']>=-2)&
                            (wtab_all['ERA5_all_lv1']<2))
    regime_condB_nguinea = ((wtab_all['ERA5_all_lv0']>=-2)&
                            (wtab_all['ERA5_all_lv0']<2)&
                            (wtab_all['ERA5_all_lv1']>=-4)&
                            (wtab_all['ERA5_all_lv1']<-2))
    regime_index_nguinea = wtab_all[regime_condA_nguinea|regime_condB_nguinea].index
    nguinea_lon, nguinea_lat = (143, 148), (-12, -10)
    nguinea_r, nguinea_theta = create_polar_info.cal_for_all_dates(datelist=datelist, func=create_polar_info.Get_r_theta, func_config={'IVT_fpath':IVT_fpath, 'env':'IVT', 'lon_range':nguinea_lon, 'lat_range':nguinea_lat})
    nguinea_dict = {'lon_range':nguinea_lon, 'lat_range':nguinea_lat, 'r':nguinea_r, 'theta':nguinea_theta}
    np.save('../../../data/processed/weather_table_self/AAM_other_area/nguinea_upstream.npy', nguinea_dict)
    print(f"Upstream polar info. of southern New Guinea saved.")
    