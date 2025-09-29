import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from functools import partial
import multiprocessing


class AAM_mean_precip_map():
    def __init__(self, year_list:list, month_list:list, 
                 special_start_date:str=False, special_end_date:str=False):
        self.YEARS     = year_list
        self.MONTHS    = month_list
        self.STARTDATE = self._convert_to_dobj(special_start_date) if special_start_date else None
        self.ENDDATE   = self._convert_to_dobj(special_end_date) if special_end_date else None
        self._create_date_list()
    
    def _convert_to_dobj(self, date):
        """
        Convert a date string into a datetime object.
        Two types of string format are supported: 20051218 or 2005-12-18.
        """
        if isinstance(date, str):
            if len(date)>8:
                dateobj = datetime.strptime(date, '%Y-%m-%d')
            else:
                dateobj = datetime.strptime(date, '%Y%m%d')
        else:
            dateobj = date
        return dateobj
        
    def _create_date_list(self):
        """
        Create date list (of strings and datetime objects) in specific months for specific year range.
        Supports discrete month choices, designated start date and end date.
        !!Future adjustment!!: discrete year choices
        """
        start_date = self.STARTDATE if self.STARTDATE is not None else datetime(self.YEARS[0], 1, 1)
        end_date   = self.ENDDATE if self.ENDDATE is not None else datetime(self.YEARS[-1], 12, 31)
        # All dates in the year range (datetime objects and strings)
        self._dlist= [start_date+timedelta(days=i) for i in range((end_date-start_date).days+1)]
        self.DLIST = [(start_date+timedelta(days=i)).strftime("%Y%m%d") for i in range((end_date-start_date).days+1)]
        # Addtionally, extract dates in selected months
        self._dlist_month= [dobj for dobj in self._dlist if dobj.month in self.MONTHS]
        self.DLIST_Month = [dobj.strftime("%Y%m%d") for dobj in self._dlist if dobj.month in self.MONTHS]
    
    def _cal_for_all_dates(self, datelist, func, func_config={}, cores=10):
        """
        Call the calculation methods (for single day) and return results for a range of days.
        *Can handle single/multiple outputs from func.
        
        :param datelist: List of dates for iterating calculation
        :type  datelist: list
        :param func: Funciton(method) to call
        :type  func: function
        :param func_config: Parameters for func
        :type  func_config: dict, optional, default={}
        
        :return: Calculation result for each day
        :rtype : tuple or list
        """
        # Create a partial function that pre-binds the config to the func
        func_with_config = partial(func, **func_config)
        with multiprocessing.Pool(processes=cores) as pool:
            results = pool.map(func_with_config, datelist)  # func result * number of processes
            
        # Create nested list to handle various amount of outputs
        output_num = len(results[0]) if isinstance(results[0], tuple) else 1  # check multi-/single output
        nest_list  = [[] for _ in range(output_num)]        # nested list handling func output
        # Store outputs in individual lists
        for output in results:                              # output: output for single call of func
            if output_num > 1:
                for i, val in enumerate(output):
                    nest_list[i].append(val)
            else:
                nest_list[0].append(output)
                
        return tuple(nest_list) if output_num > 1 else nest_list[0]
    
    def get_AAM_precip(self, date:str):
        # Load gpm file
        try:
            gpm_date = xr.open_dataset(f"DATA/GPM/AsianMonsoon/{date[:4]}/{date[4:6]}/{date}.nc")
        else:
            print(f"GPM IMERG daily data are needed.")
        # Masked precip
        gpm_aam  = gpm_date.daily_precip.sel(lon=slice(66, 156), lat=slice(-15, 30))
        return gpm_aam
    
    def cal_mean_pcp(self, datelist:list=[]):
        # Initiate dict with datelist entry
        datelist   = datelist if len(datelist)>0 else self.DLIST_Month
        # Average pcp
        daily_pcps = self._cal_for_all_dates(datelist=datelist,
                                             func=self.get_AAM_precip,
                                             cores=10)
        precip_stack= xr.concat(daily_pcps, dim='time')
        pcp_marknan = xr.where(precip_stack>=0, precip_stack, np.nan)
        final_mean  = pcp_marknan.mean(dim='time', skipna=True)
        return pcp_marknan, final_mean
    
if __name__=='__main__':
    aam_pcp = AAM_mean_precip_map(year_list=np.arange(2001, 2020).tolist(), month_list=np.arange(4, 10).tolist())
    aam_pcp_stack, aam_pcp_mean = aam_pcp.cal_mean_pcp()
    aam_avg_pcp_dict = {'mean':aam_pcp_mean}
    np.save('../../data/processed/AAM_avg_daily_pcp_summer.npy', aam_avg_pcp_dict)