"""
Used for the loading and reorganizing of POPS data.
"""

# load packages
import os
import calendar
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta


class dataRetrival:
    """
    This class is used to load and concatenate the POPS dataset for the desired time period.
    """

    def __init__(self):
        pass

    def create_datasets(self, sites, start_date, end_date, subsample=None, remove_dates=None):

        """
        Function to be called to load and organize all data.
        
        Loads each data file in the date range for the desired sites, 
        converts netcdf file to a pandas df, and concatenated df's into a single df.
        Each df is saved in a dict under the name of the site.
         
        Input:
        - sites: list of sites wanted in analysis
        - start_date: start of date range in form 'yyyymmdd' (str)
        - end_date: end of date range in form 'yyyymmdd' (str)
        - subsample: number of gaps between 5 second samples, 
            i.e. 12 would subsample data every 1 minute (int), defaults to None
        - remove_dates: list of dates in form 'yyyymmdd' to remove from analysis if desired
            defaults to None

        Returns: dict of dfs  
        """

        # get dates in daterange
        dates = self._make_date_range(start_date, end_date)

        if remove_dates is not None:
            # remove listed dates 
            try:
                for day in remove_dates:
                    dates.remove(day)
            except:
                print('Error removing dates')
        
        # make empty dict for data
        data_dict = {}
        
        # begin loading all data
        for site in sites:
            print('loading data for site ', site)

            # made empty df to hold all data
            all_data = pd.DataFrame()

            for day in dates:
                print('loading day ', day)

                # load data
                df = self._load_file(site, day, subsample)

                # concat into larger df
                all_data = pd.concat([all_data, df])
            
            # add complete df to dict
            data_dict[site] = all_data
        
        return data_dict
    

    def _make_date_range(self, start_date, end_date):

        """
        Created a list of all dates in the given date range for loading data.
         
        Input:
        - start_date: start of date range in form 'yyyymmdd' (str)
        - end_date: end of date range in form 'yyyymmdd' (str)

        Returns: array of dates as a string
        """

        # convert strings into datetime objects
        date_format = '%Y%m%d'
        start_date = datetime.strptime(start_date, date_format)
        end_date   = datetime.strptime(end_date, date_format)
        delta = timedelta(days=1)

        date_list = []
        current_date = start_date

        # create list of daily datetimes and convert to str of form 'yyyymmdd
        while current_date <= end_date:
            date_list.append(current_date.strftime(date_format))
            current_date += delta
        
        return date_list
    
    def _load_file(self, site, day, subsample):
        """
        Loads the desired netCDF file and converts to a Pandas df.
        Note that in order to load data without making any changes, data should be in a folder 
        called "data" in this directory.

        Parameters:
        - site: name of site (str)
        - day: specific date in format yyyymmdd (str)
        - subsample: - subsample: number of gaps between 5 second samples, 
            i.e. 12 would subsample data every 1 minute (int), defaults to None

        Returns: pandas df of data. If file does not exist, returns  df of nans.
        """

        # construct name of file
        filename = 'sailnet.pops.'+site+'.postcorrected.'+day+'.nc'

        # make path to file
        filepath = os.path.join('./data', filename)

        # load file if it exists
        if os.path.exists(filepath):
            dataset = xr.open_dataset(filepath)


            # convert to df
            df = dataset.to_dataframe()

            # subsample data
            df = df[::subsample]

            # make sure times are in a human-readable format (UTC)
            df['DateTime'] = pd.to_datetime(df['DateTime'], origin='unix')



        else:
            # create df of nans for the daterange

            # made empty df
        
            date_form = datetime.strptime(day, '%Y%m%d')
            tomorrow = date_form + timedelta(days=1)
            tomorrow = tomorrow.strftime('%Y%m%d')

            # create datetime objects for start and end 
            start_datetime = datetime.strptime(f'{day}', '%Y%m%d')
            end_datetime = datetime.strptime(f'{tomorrow}', '%Y%m%d')
            
            # create unix start and end times (UTC)
            start_unix = calendar.timegm(start_datetime.utctimetuple())
            end_unix = calendar.timegm(end_datetime.utctimetuple())

            # get list of all unix times 
            unix_times = np.arange(start_unix, end_unix, 1)

            # convert all unix times into datetimes
            datetimes = []
            for time in unix_times:
                datetimes.append(datetime.utcfromtimestamp(time))
            if subsample is not None:
                datetimes = datetimes[::5*subsample]

            df = pd.DataFrame({'DateTime':datetimes})
            bins = ['b' + str(i) for i in range(16)]

            for bin in bins:
                df[bin] = np.nan



        return df


class dataGroupings:
    """
    This class is used for preforming various groupings of the data, such as grouping temporally (i.e. averaging data over hours, days, etc.) 
    or grouping the data size bins for various analysis. 
    """

    def __init__(self):
        pass
