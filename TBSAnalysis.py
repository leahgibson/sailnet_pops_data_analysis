"""
Code for analysis and plotting of comparison between SAIL-Net and TBS data.

Download TBS data from: https://adc.arm.gov/discovery/#/results/s::tbspops
"""

# import packages
from dataHandling import POPSDataRetrival, dataGroupings, dataCompletenessVisualization

import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import xarray as xr
import os
import re

import warnings

# Suppress all future warnings
warnings.filterwarnings("ignore")

# Set the font size for different plot elements
plt.rcParams.update({
    'font.size': 8,               # Font size for general text
    'axes.titlesize': 8,          # Font size for plot titles
    'axes.labelsize': 8,          # Font size for axis labels
    'xtick.labelsize': 8,         # Font size for x-axis ticks
    'ytick.labelsize': 8,         # Font size for y-axis ticks
    'legend.fontsize': 8,         # Font size for legend
    'lines.linewidth': 1.5         # Set linewidth 
})





### functions ###
def get_day_filenames(day):
    """
    Returns list of filenames needed for analysis for the specified dat.
    """

    root_dir = './TBS_data'

    files = os.listdir(root_dir)

    # Initialize a list to store matching files
    matching_files = []

    # Iterate through each file in the directory
    for file_name in files:
        # Check if the file ends with '.nc' and contains the search string in its name
        if file_name.endswith('.nc') and day in file_name:
            matching_files.append(file_name)
    
    return matching_files

def get_all_filenames():
    """
    Returns list of all filenames
    """

    root_dir = './TBS_data'

    files = os.listdir(root_dir)

    matching_files = []

    for filename in files:
        if filename.endswith('.nc'):
            matching_files.append(filename)
    
    return matching_files


def load_tbs_data(filename):
    """
    Given filename, loads the .nc TBS file and converts to a df.

    Parameters:
    - filename: full name of .nc file to load
    """
    print('loading tbs data')
    path_to_dir = './TBS_data'
    path_to_file = os.path.join(path_to_dir, filename)
    dataset = xr.open_dataset(path_to_file)

    # convert to pandas df
    df = dataset.to_dataframe()
    df = df.reset_index(level='num_pops')


    return df

def process_tbs(df):
    """
    Given TBD df, does all cleaning, organizing, and plotting.
    """

    df, start_time, end_time = _clean_tbs_data(df)
    df = _altitude_time_bin_tbs(df)

    return df, start_time, end_time
    

def _clean_tbs_data(df):
    """
    Given the df of TBS data, data are cleaned:
    - removes data where alt is -9999
    - only keeps data when qc checks are passed

    Returns cleaned df, and start and end time of flight
    """

    print('cleaning tbs data')
    # only keep data where qc ckecks are passed
    df = df[df['qc_total_concentration'] == 0]

    # remove data where alt is -9999
    df = df[df['alt'] != -9999]

    # remove if alt is larger than 4100
    df = df[df['alt'] <= 3800]

    # remove if smaller than 2800
    df = df[df['alt'] >= 2600]

    # make time index a bin
    df = df.reset_index()

    start_time = df['time'][0]
    end_time = df['time'].iloc[-1]

    return df, start_time, end_time


def _altitude_time_bin_tbs(df):
    """
    Bins TBS data into 5m averages to remove noise.
    
    Also computes sum of 170 nm and higher for concentation plot.

    Returns this averaged df
    """

    print('time and altitude binning tbs data')

    # Define altitude bins and time window
    altitude_bins = np.arange(df['alt'].min(), df['alt'].max() + 5, 5)
    print('done making altitude bins')
    time_window = pd.Timedelta(minutes=5)
    print('made time window')

    # Convert 'Time' column to datetime if it's not already
    df['time'] = pd.to_datetime(df['time'])

    #print(len(altitude_bins))

    

    df['alt_bins'] = pd.cut(df['alt'], altitude_bins)

    print('doing groupby')

    # Group by altitude bins and time windows
    groups = df.groupby(['alt_bins', pd.Grouper(key='time', freq=time_window)])

    print('done with groupings')

    # Create a new DataFrame to store the averaged data
    headers = ['time', 'alt', 'dn_150_170','dn_170_195',
                'dn_195_220','dn_220_260','dn_260_335', 'dn_335_510',
                'dn_510_705', 'dn_705_1380', 'dn_1380_1760', 'dn_1760_2550', 
                'dn_2550_3615']
    averaged_df = pd.DataFrame(columns=headers)


    for name, group in groups:
        data = {}
        # create new data to add a new row
        for name in headers:
            data[name] = group[name].mean()
        # Create a Series with scalar values and an index
        new_row = pd.Series(data, index=headers)
    
        # Convert the Series to a DataFrame and append it
        averaged_df = averaged_df.append(new_row, ignore_index=True)

    # add in column of sum of 170 and higher
    averaged_df['dn_170_3615'] = averaged_df[['dn_170_195',
                'dn_195_220','dn_220_260','dn_260_335', 'dn_335_510',
                'dn_510_705', 'dn_705_1380', 'dn_1380_1760', 'dn_1760_2550', 
                'dn_2550_3615']].sum(axis=1)

    averaged_df = averaged_df.sort_values('time')
    averaged_df = averaged_df.reset_index(drop=True)

    return averaged_df

def compute_error(tbs_data, data_dict, day):
    """
    Computes the error berween site concentrations and TBS concentrations only if 170nm + bins are shared.

    Parameres:
    - tbs_data: df of tbs data
    - data_dict: dict of the site data
    - day: date of the flight in the form 'yyyymmdd'

    Returns: dict of average error for the flight
    """

    elevations = {
    'pumphouse':2770,
    'gothic':2915,
    'cbmid':3138,
    'irwin':3177,
    'snodgrass':3330,
    'cbtop':3468
    }

    bins = ['dn_150_170','dn_170_195','dn_195_220','dn_220_260','dn_260_335', 
            'dn_335_510','dn_510_705', 'dn_705_1380', 'dn_1380_1760', 'dn_1760_2550', 
            'dn_2550_3615']

    bin_index = [i for i in range(len(bins))]


    
    # FIGURE 14: example flight
    time = tbs_data['time'].to_list()
    dn_170_3615 = tbs_data['dn_170_3615'].to_list()
    alt = tbs_data['alt'].to_list()
    # Create a colormap for the line
    cmap = plt.get_cmap('viridis', len(time))

    # Plot a line that changes colors over time
    fig, ax = plt.subplots(figsize=(6, 3), dpi=300)
    for i in range(len(time) - 1):
        plt.plot(dn_170_3615[i:i+2], alt[i:i+2], color=cmap(i))
    # Create a color bar and format the time labels as HH:MM
    colorbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax, label='Time (UTC)')

    # get the time part of the timestamp
    # Convert the timestamp string to a datetime object
    times = []
    for t in time:
        times.append(t.strftime("%H:%M"))

    # Set custom tick locations and labels on the colorbar
    step = 25
    tick_values = times[::step]
    # map ticks to value between 0 and 1
    normalized_ticks = np.linspace(0,1,len(time))
    ticks = normalized_ticks[::step]

    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels(tick_values)
    
    

    
    # for each site, find where the flight passes through the site altitude
    percent_errors = {}
    absolute_errors = {}
    for site, elevation in elevations.items():
        percent_errors[site] = []
        absolute_errors[site] = []
        filtered_tbs = tbs_data[np.abs(tbs_data['alt'] - elevation) <= 2.5]
        #print(site, filtered_tbs)
        
        
        for index, row in filtered_tbs.iterrows():
            tbs_hist = []
            site_hist = []

            # get the timeframe to average over for the site (+/- 30 seconds)
            tbs_time = row['time']
            tbs_start = tbs_time - timedelta(seconds=30)
            tbs_end = tbs_time + timedelta(seconds=30)

            # select that part of the df
            site_df = data_dict[site]
            df = site_df[(site_df['DateTime'] > tbs_start) & (site_df['DateTime'] < tbs_end)]

            # # sum bins 170+
            # df['dn_170_3615'] = df[['dn_170_195',
            #     'dn_195_220','dn_220_260','dn_260_335', 'dn_335_510',
            #     'dn_510_705', 'dn_705_1380', 'dn_1380_1760', 'dn_1760_2550', 
            #     'dn_2550_3615']].sum(axis=1, skipna=False) # if there is a nan, whole sum is nan
            
            #print(df)
            
            # if make_plots:
            #     # get the bin data
            #     for bin in bins:
            #         tbs_hist.append(row[bin])
            #         site_hist.append(df[bin].mean())

            #     # plot the distributions
            #     plt.plot(bin_index, tbs_hist, marker='o', label='TBS')
            #     plt.plot(bin_index, site_hist, marker='^', label=site)
            #     plt.legend()
            #     plt.title(tbs_time)
            #     plt.show()
            
            # compute error for 170+ sized particles and plot with flight data
            site_avg = df['dn_170_3400'].mean()

            # compute % error = abs(site-flight)/flight
            error = round(((site_avg - row['dn_170_3615'])/row['dn_170_3615'])*100, 2)
            percent_errors[site].append(abs(error))

            #compute absolute error abs(site-flight)
            error = round(abs(site_avg - row['dn_170_3615']),2)
            absolute_errors[site].append(abs(error))

            
            
            # FIG 14 cont
            plt.plot(site_avg, elevation, marker='*', color=cmap(index), markersize=8) # plot marker of concentration next to line
            # plt.text(site_avg+0.5, elevation, str(error), color=cmap(index), fontsize=15) # plot error values
            plt.text(np.min(dn_170_3615)+0.1, elevation, site, fontsize=8) # plot site names


    plt.xlabel('Concentration cm$^{-3}$')
    plt.ylabel('Altitude (m)')
    plt.title(day)
    plt.show()
    
    
    
    # compute the average errors for the number of times the flight passed the site and save to df
    avg_percent_errors = {}
    avg_percent_errors['date'] = day 
    avg_absolute_errors = {}
    avg_absolute_errors['date'] = day
    for site, list_of_errors in percent_errors.items():
        avg_percent_errors[site] = np.mean(list_of_errors)
    
    for site, list_of_errors in absolute_errors.items():
        avg_absolute_errors[site] = np.mean(list_of_errors)
    

    return avg_percent_errors, avg_absolute_errors


### body ###

sites = ['pumphouse', 'gothic', 'cbmid', 'irwin', 'snodgrass', 'cbtop']


# proceed with analysis for all data
filenames = get_all_filenames()

headers = ['date'] + sites
# date df for storing errors
avg_percent_errors_df = pd.DataFrame(columns=headers)
median_percent_errors_df = pd.DataFrame(columns=headers)
avg_absolute_errors_df = pd.DataFrame(columns=headers)
median_absolute_errors_df = pd.DataFrame(columns=headers)

# analyze one by one
for tbs_filename in filenames:
    print(tbs_filename)
    # pull out identifying data
    site_code = tbs_filename.split('.')[1][-2:]
    yyyymmdd = re.search(r'\d{8}', tbs_filename).group()

    # load & clean tbs data
    tbs_data = load_tbs_data(tbs_filename)
    tbs_data, start_time, end_time = process_tbs(tbs_data)

    # load SAIL-Net data for given date (this is slow, maybe fix later)
    dr = POPSDataRetrival()
    groupings = dataGroupings()
    data_dict = dr.create_datasets(sites=sites, start_date=yyyymmdd, end_date=yyyymmdd, subsample=None)

    grouped_data = {}
    for site, df in data_dict.items():
        df = groupings.bin_groupings(df, grouping_option=2)
        df = df[(df['DateTime'] > start_time) & (df['DateTime'] < end_time)]
        grouped_data[site] = df
    
    # compute error
    avg_percent_errors, average_absolute_errors = compute_error(tbs_data, grouped_data, day=yyyymmdd)
    
    # append to df
    avg_percent_errors_df = avg_percent_errors_df.append(avg_percent_errors, ignore_index=True)
    avg_absolute_errors_df = avg_absolute_errors_df.append(average_absolute_errors, ignore_index=True)


# convert dates to datetimes
avg_percent_errors_df['date']= pd.to_datetime(avg_percent_errors_df['date'])
avg_absolute_errors_df['date']= pd.to_datetime(avg_absolute_errors_df['date'])



# group by dates and compute median for that site for that day
#daily_site_mean = avg_errors_df.groupby('date').mean()
daily_site_median_percent = avg_percent_errors_df.groupby('date').median()
daily_site_median_absolute = avg_absolute_errors_df.groupby('date').median()


# colorblind colors
colors = ['#377eb8', '#ff7f00', '#4daf4a',
        '#f781bf', '#a65628', '#984ea3',
        '#999999', '#e41a1c', '#dede00']


# average site medians across that date
row_means_percent = daily_site_median_percent.mean(axis=1)
# compute median of sites medians across that date
row_medians_percent = daily_site_median_percent.median(axis=1)

row_means_absolute = daily_site_median_absolute.mean(axis=1)
row_medians_absolute = daily_site_median_absolute.median(axis=1)



# plot means
# clear any remaining plot data
plt.clf()

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(6,3), dpi=300)
ax[0].plot(row_means_percent, marker='o', linestyle='None', color='#377eb8', label='Mean Error')
ax[0].plot(row_medians_percent, marker='^', linestyle='None', color='#ff7f00', label='Median Error')
ax[0].set_ylabel('Percent Error')

ax[1].plot(row_means_absolute, marker='o', linestyle='None', color='#377eb8', label='Mean Error')
ax[1].plot(row_medians_absolute, marker='^', linestyle='None', color='#ff7f00', label='Median Error')
ax[1].set_ylabel('Absolute Error')
ax[1].set_xlabel('UTC')
plt.legend()
plt.show()



    
    















