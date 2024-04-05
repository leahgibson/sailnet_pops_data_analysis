"""
Main file for the analysis of POPS data.

Analysis tools are ONLY compatible with the postcorrected POPS datasets are publically available on the ARM Data Discovery:
https://adc.arm.gov/discovery/#/results/iopShortName::amf2021SAILCAIVIMT/instrument_code::pops 
as netCDF files.
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


# data_groupings = dataGroupings()

# group = {}
# for site in sites:
#     group[site] = data_groupings.bin_groupings(grouped_dict[site], grouping_option=2)


# temp_analysis = temporalAnalysis()
# temp_analysis.plot_psd_timeseries(network_df)

# #### get second set of data
# start_date='20220616'
# end_date='20220616'

# # load data
# data_dict=dr.create_datasets(sites, start_date, end_date, subsample=12)


# # time bin data
# grouping = dataGroupings()
# grouped_dict_2 = {}
# for site in sites:
#     grouped_dict_2[site] = grouping.temporal_grouping(data_dict[site], '1H')

# network_df_2 = grouping.network_mean(grouped_dict_2)

# group2={}
# for site in sites:
#     group2[site] = data_groupings.bin_groupings(grouped_dict_2[site], grouping_option=2)

# # plot psd for this data
# basic_vis = basicVisualization()
# basic_vis.plot_psd(grouped_dict['snodgrass'], grouped_dict_2['snodgrass'])


# #### get third set of data
# start_date='20230523'
# end_date='20230523'

# # load data
# data_dict=dr.create_datasets(sites, start_date, end_date, subsample=12)


# # time bin data
# grouping = dataGroupings()
# grouped_dict_3 = {}
# for site in sites:
#     grouped_dict_3[site] = grouping.temporal_grouping(data_dict[site], '1H')

# network_df_3 = grouping.network_mean(grouped_dict_3)

# group3={}
# for site in sites:
#     group2[site] = data_groupings.bin_groupings(grouped_dict_3[site], grouping_option=2)

#### visualization and analysis

# # plot data to gether
# basic_vis = basicVisualization()

# basic_vis.plot_different_time_segments(group['snodgrass'], group2['snodgrass'], bin_name='dn_170_3400')





# # plot supermicron sized particles overlapping
# basic_vis = basicVisualization()
# basic_vis.plot_psd(network_df)

# basic_vis.plot_network_timeseries(network_grouped_2, bin_name='dn_155_170')

# basic_vis.plot_overlapping_timeseries(network_grouped, bin_name='supermicron')
# basic_vis.plot_overlapping_timeseries(network_grouped_2, bin_name='dn_170_3400')
# basic_vis.plot_overlapping_timeseries(network_grouped_2, bin_name='total')

# plot diurnal for months
temp = temporalAnalysis()
temp.plot_monthly_psd(network_df)

temp.plot_psd_timeseries(network_df)



# temp.plot_monthly_diurnal(network_grouped_2, bin_names=['dn_170_3400'])



# sv = spatialVariability()
# sv.sudo_variogram(group, bin_names=['dn_170_300', 'dn_300_870', 'dn_870_3400'], distance_type='vertical', sum_headers=False)
# cv = sv.coefficient_of_variation(group, bin_names=['dn_170_3400'], rolling=5, sum_headers=False)
# cv = sv.coefficient_of_variation(group, bin_names=['dn_170_300', 'dn_300_870', 'dn_870_3400'], rolling=7, sum_headers=False)
# cv = sv.coefficient_of_variation(group, bin_names=['dn_170_300', 'dn_300_870', 'dn_870_3400'], rolling=7)


# vis = timeseriesVisualization()
# vis.plot_timeseries_together(group, bin_name='dn_870_3400')




# sv.sudo_variogram(grouped_dict, ['b3', 'b4', 'b5'], distance_type='euclidean', sum_headers=False)

# ta = temporalAnalysis()
# ta.plot_monthly_diurnal(cv, bin_names=['b3', 'b4'])

# network = networkDesign(group, bin_headers=['dn_170_300', 'dn_300_870', 'dn_870_3400'])
# network.plot_representation_timeseries()
# network.plot_representation_bars()

# basic_vis = basicVisualization()
# basic_vis.plot_overlapping_timeseries(network_grouped, bin_name='dn_170_3400')



