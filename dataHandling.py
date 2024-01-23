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
    This class is used for preforming various groupings of the data, such as grouping temporally (i.e. averaging data over hours, days, etc.),
    grouping the data size bins for various analysis,
    or "grouping" into the network mean (i.e. averaging over all sites.)
    """

    def __init__(self):
        pass

    def temporal_grouping(self, df, averaging_frequency):
        """
        Bins data temporally by averaging over time intervals.
        
        Input:
        - df: df of data
        - averaging_frequency: frequency to average over
            in form 'nMin', 'nH', or 'nD' where n is an integer
        
        Returns: df of time binned data
        """

        bin_intervals = pd.date_range(start=df['DateTime'].min(), end=df['DateTime'].max(), freq=averaging_frequency)
    

        if 'D' in averaging_frequency:
            new_times = bin_intervals.strftime('%Y-%m-%d').tolist()
        else:
            new_times = bin_intervals.strftime('%Y-%m-%d %H:%M:%S').tolist()

        new_df = pd.DataFrame()
        new_df['DateTime'] = new_times[:-1]


        # cut the data into time bins using the defined intervals
        df['time_bin'] = pd.cut(df['DateTime'], bins=bin_intervals)

        # calculate binned averages for each bin
        bins = ['b' + str(i) for i in range(16)]
        for bin in bins:
            binned_avg = df.groupby('time_bin')[bin].mean().tolist()
            new_df[bin] = binned_avg
        

        return new_df
    
    def bin_groupings(self, df, grouping_option):
        """
        Groups bins for analysis by summing bins.

        If one of the bins doesn't contain data, the result of the sum is also empty.
        
        Inputs:
        - df: df of data
        - grouping_option: int, accepts 1, 2, or 3 corresponding to the three options below:
            - option 1: dn_140_170, dn_170_200, dn_200_300, dn_300_870, dn_870_3400, dn_170_3400, total
            - option 2: dn_140_155, dn_155_170, dn_170_300, dn_870_3400, dn_170_3400, dn_170_8700, total
            - option 3: submicron, supermicron, total
        
        Returns: df of binned data
        """

        bins = ['b' + str(i) for i in range(16)]
        grouped_df = pd.DataFrame()

        if grouping_option == 1:
            grouped_df['DateTime'] = df['DateTime']
            grouped_df['dn_140_170'] = df[['b0', 'b1']].sum(axis=1, skipna=False)
            grouped_df['dn_170_200'] = df[['b2', 'b3']].sum(axis=1, skipna=False)
            grouped_df['dn_200_300'] = df[['b4', 'b5', 'b6']].sum(axis=1, skipna=False)
            grouped_df['dn_300_870'] = df[['b7', 'b8', 'b9', 'b10']].sum(axis=1, skipna=False)
            grouped_df['dn_870_3400'] = df[['b' + str(i) for i in range(11,16)]].sum(axis=1, skipna=False)
            grouped_df['dn_170_3400'] = df[['b' + str(i) for i in range(2,16)]].sum(axis=1, skipna=False)
            grouped_df['total'] = df[bins].sum(axis=1, skipna=False)
        
        if grouping_option == 2:
            grouped_df['DateTime'] = df['DateTime']
            grouped_df['dn_140_155'] = df['b0']
            grouped_df['dn_155_170'] = df['b1']
            grouped_df['dn_170_300'] = df[['b2', 'b3', 'b4', 'b5', 'b6']].sum(axis=1, skipna=False)
            grouped_df['dn_300_870'] = df[['b7', 'b8', 'b9', 'b10']].sum(axis=1, skipna=False)
            grouped_df['dn_870_3400'] = df[['b11', 'b12', 'b13', 'b14', 'b15']].sum(axis=1, skipna=False)
            grouped_df['dn_170_3400'] = df[['b' + str(i) for i in range(2,16)]].sum(axis=1, skipna=False)
            grouped_df['dn_170_870'] = df[['b' + str(i) for i in range(2,11)]].sum(axis=1, skipna=False)
            grouped_df['total'] = df[bins].sum(axis=1, skipna=False)
        
        if grouping_option == 3:
            grouped_df['DateTime'] = df['DateTime']
            grouped_df['submircon'] = df[['b' + str(i) for i in range(11)]].sum(axis=1, skipna=False)
            grouped_df['supermicron'] = df[['b' + str(i) for i in range(11, 16)]].sum(axis=1, skipna=False)
        

        return grouped_df
    
    def network_mean(self, dict_of_data):
        """
        Averages over all dfs in the dict to get a network mean, equal to the average of the sites at time t.
        
        Note that this function only accepts 16 bins, and other rebinning/grouping should be done AFTER.
        
        Inputs:
        - dict_of_data: dictionary of data in 16 bin structure
        
        Rturns: df of the network mean for all 16 bins 
        """

        bins = ['b' + str(i) for i in range(16)]
        bin_dict = {}
        for bin in bins:
            bin_dict[bin] = pd.DataFrame()


        sites = []
        for site, df in dict_of_data.items():
            sites.append(site)
            for bin in bins:
                bin_dict[bin]['DateTime'] = df['DateTime']
                bin_dict[bin][site] = df[bin]
        
        
        # compute mean across all sites and save to new df
        network_mean_df = pd.DataFrame()
        for bin in bins:
            network_mean_df[bin] = bin_dict[bin][sites].mean(axis=1).to_list()
        network_mean_df.insert(0, 'DateTime', bin_dict[bin]['DateTime'].to_list())
        #network_mean_df['DateTime'] = bin_dict[bin]['DateTime'].to_list()
        # add in 'total' column
        network_mean_df['total'] = network_mean_df[bins].sum(axis=1)

        return network_mean_df



        


