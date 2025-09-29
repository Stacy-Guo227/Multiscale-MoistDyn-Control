import numpy as np
import xarray as xr
import h5py
import glob
from datetime import *
import concurrent.futures
import time

def get_gpm_pcp(imerg_path:str, date:str):
    # Date object
    date_obj= datetime.strptime(date, '%Y%m%d')
    date_yr = date_obj.year
    date_idx= (datetime.strptime(date, '%Y%m%d')-datetime(date_yr-1, 12, 31)).days
    # Load dataset
    fpath   = f'{imerg_path}/{date_yr}/{date_idx:03d}/*'
    flist   = glob.glob(fpath)
    # Sum daily precip.
    for i, fname in enumerate(flist):
        h5_gpm  = h5py.File(fname, 'r')
        latAll, lonAll   = h5_gpm['Grid']['lat'][:], h5_gpm['Grid']['lon'][:]
        latCond, lonCond = (latAll>=-20)&(latAll<=30), (lonAll>=60)&(lonAll<=160)
        latReg, lonReg   = latAll[latCond], lonAll[lonCond]
        latIdx, lonIdx   = np.arange(latAll.shape[0])[latCond], np.arange(lonAll.shape[0])[lonCond]
        pcpReg  = h5_gpm['Grid']['precipitation'][0, lonIdx[0]:lonIdx[-1]+1, latIdx[0]:latIdx[-1]+1]
        if i == 0:
            temp= np.rollaxis(pcpReg, axis=1)
        else:
            temp= temp+np.rollaxis(pcpReg, axis=1)
    return latReg, lonReg, temp/2

def process_date(imerg_path:str, now_date):
    now_date_str = now_date.strftime('%Y%m%d')
    reg_lat_gpm, reg_lon_gpm, daily_pcp = get_gpm_pcp(imerg_path, date=now_date_str)  # Get daily precip. (numpy array)
    lat_coord    = xr.DataArray(reg_lat_gpm, dims='lat', name='lat')      # xarray coordination
    lon_coord    = xr.DataArray(reg_lon_gpm, dims='lon', name='lon')
    da_daily_pcp = xr.DataArray(data=daily_pcp, coords={'lat': lat_coord, 'lon': lon_coord}, 
                                dims=['lat', 'lon'],
                                attrs=dict(description='daily accumulated precip.', 
                                           unit='mm/day'),
                                name='daily_precip')
    da_daily_pcp = da_daily_pcp.assign_coords(lat=reg_lat_gpm, lon=reg_lon_gpm)
    ds_daily_pcp = da_daily_pcp.to_dataset(promote_attrs=True)            # xr.Dataset
    ds_daily_pcp.to_netcdf(f'/data/ch995334/DATA/GPM/AsianMonsoon/'\
                           f'{now_date.year:4d}/{now_date.month:02d}/{now_date_str}.nc')
    print(now_date)

def main():
    start_date, end_date = datetime(2001, 1, 1), datetime(2020, 1, 1)
    now_date = start_date   # Initialize
    imerg_path = input('Please input IMERG 0.5hr data path:')

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        while now_date < end_date:
            futures.append(executor.submit(process_date, imerg_path, now_date))
            now_date = now_date + timedelta(days=1)

        # Wait for all threads to finish
        concurrent.futures.wait(futures)

if __name__ == "__main__":
    tic = time.time()
    main()
    toc = time.time()
    print(f"Elapsed time: {(toc - tic)/3600.} hours")
