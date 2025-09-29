"""
Calculate and save the regional mean precip. of Apr–Sep 2001–2019 from IMERG.
Regions include: the Western Ghats, the eastern coast of the Bay of Bengal, western Luzon, and southern New Guinea.
"""
import numpy as np
import pandas as pd
import xarray as xr
import glob
from datetime import datetime, timedelta
import logging
from functools import partial
import multiprocessing


class precip_table():
    def __init__(self, mask_region_label:str, 
                 year_list:list, month_list:list, 
                 special_start_date:str=False, special_end_date:str=False):
        # Input arguments
        self.REGIONLIST= ['india', 'bob', 'luzon', 'sumatra', 'new guinea']
        if mask_region_label in self.REGIONLIST:
            self.REGION    = mask_region_label  
            self.YEARS     = year_list
            self.MONTHS    = month_list
            self.STARTDATE = self._convert_to_dobj(special_start_date) if special_start_date else None
            self.ENDDATE   = self._convert_to_dobj(special_end_date) if special_end_date else None
            self._create_date_list()
            # Mask template
            ds_gpm           = xr.open_dataset("../../data/processed/demo_data/AAM_IMERG_daily_20010701.nc")
            self._mask_empty = np.zeros(ds_gpm.daily_precip.shape)
            self.mask_reg    = self._self_defined_regional_mask()
        else: raise ValueError(f"Inavailable region. Please choose from {self.REGIONLIST}")
        
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
            
    def _self_defined_regional_mask(self):
        mask_reg = self._mask_empty.copy()   # Initialize
        if self.REGION == 'india':
            mask_reg[387:411, 126:141] = 1
            mask_reg[375:387, 127:143] = 1
            mask_reg[363:375, 130:146] = 1
            mask_reg[351:363, 133:151] = 1
            mask_reg[339:351, 140:157] = 1
            mask_reg[327:339, 143:161] = 1
            mask_reg[315:327, 148:166] = 1
            mask_reg[303:315, 155:173] = 1
            mask_reg[280:303, 162:176] = 1
        elif self.REGION == 'bob':
            mask_reg[423:435, 310:331] = 1
            mask_reg[411:423, 313:334] = 1
            mask_reg[399:411, 318:339] = 1
            mask_reg[387:399, 330:351] = 1
            mask_reg[375:387, 336:357] = 1
            mask_reg[360:375, 335:356] = 1
        elif self.REGION == 'luzon':
            mask_reg[375:387, 598:619] = 1
            mask_reg[363:375, 595:616] = 1
            mask_reg[339:363, 590:611] = 1
        elif self.REGION == 'new guinea':
            mask_reg[151:156, 765:793] = 1
            mask_reg[146:151, 770:803] = 1
            mask_reg[141:146, 775:817] = 1
            mask_reg[138:141, 780:822] = 1
            
        return mask_reg    
    
    def get_masked_precip(self, date:str):
        # Load gpm file
        try:
            gpm_date = xr.open_dataset(f"/DATA/GPM/AsianMonsoon/{date[:4]}/{date[4:6]}/{date}.nc")
        except:
            print("GPM IMERG daily files are needed.")
        # Masked precip
        gpm_mask = gpm_date.daily_precip.where(self.mask_reg)
        return gpm_mask
    
    def cal_mean_dpcp(self, date:str):
        gpm_mask = self.get_masked_precip(date)
        return np.nanmean(gpm_mask)
    
    def cal_top10mean_dpcp(self, date:str):
        gpm_mask = self.get_masked_precip(date).values
        top10    = np.sort(gpm_mask[~np.isnan(gpm_mask)])[-10:]
        return np.nanmean(top10)
    
    def cal_max_dpcp(self, date:str):
        gpm_mask = self.get_masked_precip(date)
        return np.nanmax(gpm_mask)
    
    def cal_std_dpcp(self, date:str):
        gpm_mask = self.get_masked_precip(date)
        return np.nanstd(gpm_mask)
    
    def getdf_metrics_maskpcp(self, datelist:list=[], Mean=False, Top10_mean=False, Max=False, Std=False):        
        # Initiate dict with datelist entry
        datelist = datelist if len(datelist)>0 else self.DLIST_Month
        pcp_table= {'yyyymmdd':datelist}
        # Add other entries
        if Mean:
            mean = self._cal_for_all_dates(datelist=datelist, func=self.cal_mean_dpcp)
            pcp_table['mean'] = mean
        if Top10_mean:
            top10= self._cal_for_all_dates(datelist=datelist, func=self.cal_top10mean_dpcp)
            pcp_table['top10_mean'] = top10
        if Max:
            _max = self._cal_for_all_dates(datelist=datelist, func=self.cal_max_dpcp)
            pcp_table['max']  = _max
        if Std:
            std  = self._cal_for_all_dates(datelist=datelist, func=self.cal_std_dpcp)
            pcp_table['std']  = std
        return pd.DataFrame(pcp_table)
    
if __name__=='__main__':
    """
    - Save with `.csv` for human readability
    - Save with `.pkl` for metadata (pcp. table & regional mask)
    The pcp table in both files should be identical.
    """
    # India
    ptable_india_all  = precip_table(mask_region_label='india', 
                                     year_list=np.arange(2001, 2020).tolist(), month_list=np.arange(4, 10).tolist())
    df_india_all      = ptable_india_all.getdf_metrics_maskpcp(Mean=True, Top10_mean=True, Max=True, Std=True)
    df_india_all.to_csv('../../data/processed/weather_table_self/india_pcp_all.csv', index=False)
    df_india_all_wattr= df_india_all.copy()
    df_india_all_wattr.attrs["mask_reg_array"] = ptable_india_all.mask_reg
    df_india_all_wattr.to_pickle("../../data/processed/weather_table_self/india_pcp_all_wattr.pkl")
    
    # BoB
    ptable_bob_all  = precip_table(mask_region_label='bob', 
                                   year_list=np.arange(2001, 2020).tolist(), month_list=np.arange(4, 10).tolist())
    df_bob_all      = ptable_bob_all.getdf_metrics_maskpcp(Mean=True, Top10_mean=True, Max=True, Std=True)
    df_bob_all.to_csv('../../data/processed/weather_table_self/bob_pcp_all.csv', index=False)
    df_bob_all_wattr= df_bob_all.copy()
    df_bob_all_wattr.attrs["mask_reg_array"] = ptable_bob_all.mask_reg
    df_bob_all_wattr.to_pickle("../../data/processed/weather_table_self/bob_pcp_all_wattr.pkl")
    
    # Luzon
    ptable_luzon_all  = precip_table(mask_region_label='luzon', 
                                     year_list=np.arange(2001, 2020).tolist(), month_list=np.arange(4, 10).tolist())
    df_luzon_all      = ptable_luzon_all.getdf_metrics_maskpcp(Mean=True, Top10_mean=True, Max=True, Std=True)
    df_luzon_all.to_csv('../../data/processed//weather_table_self/luzon_pcp_all.csv', index=False)
    df_luzon_all_wattr= df_luzon_all.copy()
    df_luzon_all_wattr.attrs["mask_reg_array"] = ptable_luzon_all.mask_reg
    df_luzon_all_wattr.to_pickle("../../data/processed//weather_table_self/luzon_pcp_all_wattr.pkl")
    
    # New Guinea
    ptable_nguinea_all  = precip_table(mask_region_label='new guinea', 
                                       year_list=np.arange(2001, 2020).tolist(), month_list=np.arange(4, 10).tolist())
    df_nguinea_all      = ptable_nguinea_all.getdf_metrics_maskpcp(Mean=True, Top10_mean=True, Max=True, Std=True)
    df_nguinea_all.to_csv('../../data/processed/weather_table_self/nguinea_pcp_all.csv', index=False)
    df_nguinea_all_wattr= df_nguinea_all.copy()
    df_nguinea_all_wattr.attrs["mask_reg_array"] = ptable_nguinea_all.mask_reg
    df_nguinea_all_wattr.to_pickle("../../data/processed/weather_table_self/nguinea_pcp_all_wattr.pkl")