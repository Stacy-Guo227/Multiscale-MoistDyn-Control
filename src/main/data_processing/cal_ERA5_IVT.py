"""
Calculate low-level (1000–700 hPa) IVT in the Asian-Australian monsoon region (15S–30 N, 66–154 E) from ERA5 daily data.
"""

# Import
import numpy as np
import xarray as xr

def Cal_IVT(year:int, ERA5_fpath:str, save_fpath:str, level_range:tuple, lat_range:tuple, lon_range:tuple):
    # Load ERA5 u, v, q
    mds_ua = xr.open_mfdataset(f'{ERA5_fpath}/u/{year}/*')
    mds_va = xr.open_mfdataset(f'{ERA5_fpath}/v/{year}/*')
    mds_qv = xr.open_mfdataset(f'{ERA5_fpath}/q/{year}/*')
    mds_ua.persist()
    mds_va.persist()
    mds_qv.persist()
    # Expand "level" into 4-dim
    lev4d = mds_ua['level'].expand_dims(dim={'time': 1}, axis=0).expand_dims(dim={'latitude': 1}, axis=2).expand_dims(dim={'longitude': 1}, axis=3)
    # Calculate IVT (x-, y-, total-): np.ndarray
    ivtu  = np.trapz(mds_ua['u'].sel(level=slice(min(level_range), max(level_range)), latitude=slice(max(lat_range), min(lat_range)), longitude=slice(min(lon_range), max(lon_range))).data*\
                     mds_qv['q'].sel(level=slice(min(level_range), max(level_range)), latitude=slice(max(lat_range), min(lat_range)), longitude=slice(min(lon_range), max(lon_range))).data, 
                     lev4d.sel(level=slice(min(level_range), max(level_range))), axis=1)/9.8*100
    
    ivtv  = np.trapz(mds_va['v'].sel(level=slice(min(level_range), max(level_range)), latitude=slice(max(lat_range), min(lat_range)), longitude=slice(min(lon_range), max(lon_range))).data*\
                     mds_qv['q'].sel(level=slice(min(level_range), max(level_range)), latitude=slice(max(lat_range), min(lat_range)), longitude=slice(min(lon_range), max(lon_range))).data, 
                     lev4d.sel(level=slice(min(level_range), max(level_range))), axis=1)/9.8*100
    
    ivt_total = np.sqrt(ivtu**2+ivtv**2)
    # Assigned as DataArray
    da_ivtu = xr.DataArray(data=ivtu, dims=['time', 'lat', 'lon'],
                           attrs=dict(description='IVT in x-direction',
                                      unit='kg/m/s'), 
                           name='IVTx')
    da_ivtu = da_ivtu.assign_coords(time=mds_ua.time, 
                                    lat=np.arange(min(lat_range), max(lat_range)+0.1, 0.25)[::-1], 
                                    lon=np.arange(min(lon_range), max(lon_range)+0.1, 0.25))
    
    da_ivtv = xr.DataArray(data=ivtv, dims=['time', 'lat', 'lon'],
                           attrs=dict(description='IVT in y-direction',
                                      unit='kg/m/s'), 
                           name='IVTy')
    da_ivtv = da_ivtv.assign_coords(time=mds_va.time, 
                                    lat=np.arange(min(lat_range), max(lat_range)+0.1, 0.25)[::-1], 
                                    lon=np.arange(min(lon_range), max(lon_range)+0.1, 0.25))
    
    da_ivt = xr.DataArray(data=ivt_total, dims=['time', 'lat', 'lon'],
                      attrs=dict(description='Total IVT magnitude',
                                 unit='kg/m/s'), 
                      name='IVT_total')
    da_ivt = da_ivt.assign_coords(time=mds_ua.time, 
                                  lat=np.arange(min(lat_range), max(lat_range)+0.1, 0.25)[::-1], 
                                  lon=np.arange(min(lon_range), max(lon_range)+0.1, 0.25))
    
    # Only summer data (Apr–Sep)
    da_ivtu_sum= da_ivtu.sel(time=((da_ivtu['time.month'] >= 4) & (da_ivtu['time.month'] <= 9))).rename('sum_IVTx')
    da_ivtv_sum= da_ivtv.sel(time=((da_ivtv['time.month'] >= 4) & (da_ivtv['time.month'] <= 9))).rename('sum_IVTy')
    da_ivt_sum = da_ivt.sel(time=((da_ivt['time.month'] >= 4) & (da_ivt['time.month'] <= 9))).rename('sum_IVT_total')
    ds_ivt     = xr.merge([da_ivtu_sum, da_ivtv_sum, da_ivt_sum], combine_attrs='drop_conflicts')
    
    # Save it as .nc file (single year)
    ds_ivt.to_netcdf(f'{save_fpath}/{year}.nc')


if __name__=='__main__':
    year        = 2001       # can be a for-loop
    ERA5_fpath  = input('Please assign path of ERA5 variable files (U, V, Q):')
    level_range = (1000, 700) 
    lat_range   = (-15, 30)
    lon_range   = (66, 154)
    save_fpath  = input('Please assign the path for saving calculated-IVT files:')
    
    Cal_IVT(year=year, ERA5_fpath=ERA5_fpath, save_fpath=save_fpath, level_range=level_range, lat_range=lat_range, lon_range=lon_range)