"""
Main file for the analysis of POPS data.

Analysis tools are ONLY compatible with the postcorrected POPS datasets are publically available on the ARM Data Discovery:
https://doi.org/10.5439/2203692) as netCDF files.
"""


# import packages
from dataHandling import dataRetrival, dataGroupings
from networkMeanAnalysis import basicVisualization, temporalAnalysis
from spatialAnalysis import timeseriesVisualization, spatialVariability, networkDesign

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd


# set up date range and sites for analysis
start_date = '20220611'
end_date = '20220616'

sites = ['snodgrass', 'cbtop', 'cbmid', 'gothic', 'pumphouse', 'irwin']

# load data
dr = dataRetrival()
data_dict = dr.create_datasets(sites, start_date, end_date, subsample=None)

# time bin data
grouping = dataGroupings()
grouped_dict_30M = {}
for site in sites:
    grouped_dict_30M[site] = grouping.temporal_grouping(data_dict[site], '30Min')

cbtop_data = grouping.temporal_grouping(data_dict['snodgrass'], '30Min')

grouped_30M = grouping.bin_groupings(cbtop_data, grouping_option=2)
grouped = grouping.bin_groupings(data_dict['snodgrass'], grouping_option=2)

grouped_30M['DateTime'] = pd.to_datetime(grouped_30M['DateTime'])
grouped['DateTime'] = pd.to_datetime(grouped['DateTime'])


# plot
plt.plot(grouped['DateTime'], grouped['total'], label='1s resolution', color='gray', linewidth='1')
plt.plot(grouped_30M['DateTime'], grouped_30M['total'], label='averaged', color='green')
plt.legend()
plt.ylabel('cm$^{-3}$')
plt.xlabel('Time (UTC)')
plt.show()


