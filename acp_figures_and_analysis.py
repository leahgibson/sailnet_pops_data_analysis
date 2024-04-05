""" 
Code to generate figures in the paper 

Data are accessible on the ARM Data Discovery website: https://adc.arm.gov/discovery/#/results/iopShortName::amf2021SAILCAIVIMT/instrument_code::pops 
"""

# import packages
from dataHandling import dataRetrival, dataGroupings, dataCompletenessVisualization
from networkMeanAnalysis import basicVisualization, temporalAnalysis
from spatialAnalysis import timeseriesVisualization, spatialVariability, networkDesign

# set up date range and sites for analysis
start_date = '20211001'
end_date = '20230722'

sites = ['pumphouse', 'gothic', 'cbmid', 'irwin', 'snodgrass', 'cbtop']

# load data
dr = dataRetrival()
data_dict = dr.create_datasets(sites=sites, start_date=start_date, end_date=end_date, subsample=12)


# time bin data
grouping = dataGroupings()
time_grouped_dict_1H = {}   # group data by hours
time_grouped_dict_1D = {}   # group data by 1 day intervals
time_grouped_dict_15D = {}
for site in sites:
    time_grouped_dict_1H[site] = grouping.temporal_grouping(data_dict[site], averaging_frequency='1H')
    time_grouped_dict_1D[site] = grouping.temporal_grouping(data_dict[site], averaging_frequency='1D')
    time_grouped_dict_15D[site] = grouping.temporal_grouping(data_dict[site], averaging_frequency='15D')


# group bins 
bin_grouped_dict_1D = {}
bin_grouped_dict_15D = {}
for site in sites:
    bin_grouped_dict_1D[site] = grouping.bin_groupings(time_grouped_dict_1D[site], grouping_option=2)
    bin_grouped_dict_15D[site] = grouping.bin_groupings(time_grouped_dict_15D[site], grouping_option=2)

"""
# FIGURE: data completion
data_completeness = dataCompletenessVisualization()
data_completeness.plot_total_completeness(bin_grouped_dict_1D, bin_name='dn_170_3400')


# FIGURE: timeseries of daily averaged POPS data for all six sites
timeseries_vis = timeseriesVisualization()
timeseries_vis.plot_timeseries_together(bin_grouped_dict_1D, bin_name='dn_170_3400')

# compute network mean
network_mean_1H = grouping.network_mean(time_grouped_dict_1H)
network_mean_1D = grouping.network_mean(time_grouped_dict_1D)

# bin groupings of network mean
bin_grouped_network_mean_1D = grouping.bin_groupings(network_mean_1D, grouping_option=2)
bin_grouped_network_mean_1H = grouping.bin_groupings(network_mean_1H, grouping_option=2)

# print network mean stats
network_analysis = temporalAnalysis()
network_analysis.basic_stats(data=bin_grouped_network_mean_1D, bin_name='dn_170_3400')

# FIGURE: network mean of POPS overlaid by day
network_basic_vis = basicVisualization()
network_basic_vis.plot_overlapping_timeseries(bin_grouped_network_mean_1D, bin_name='dn_170_3400')

# FIGURE: daily diurnal cycle averaged monthly
network_analysis.plot_monthly_diurnal(bin_grouped_network_mean_1H, bin_names=['dn_170_3400'])

# FIGURE: average particle size distribution averaged monthly
network_analysis.plot_monthly_psd(network_mean_1H)

# FIGURE: timeseries of PSD from the network mean
network_analysis.plot_psd_timeseries(network_mean_1D)
"""

# FIGURE: average percent diff
spatial_analysis = spatialVariability()
spatial_analysis.sudo_variogram(bin_grouped_dict_1D, bin_names=['dn_170_300', 'dn_300_870', 'dn_870_3400'], distance_type='vertical', sum_headers=False)

# FIGURE: CV of data
spatial_analysis.coefficient_of_variation(bin_grouped_dict_1D, bin_names=['dn_170_3400'], sum_headers=False)
spatial_analysis.coefficient_of_variation(bin_grouped_dict_1D, bin_names=['dn_170_3400'], rolling=30, sum_headers=False)
"""
# # FIGURE: max and min counts
# spatial_analysis.compute_number_max_min_concentrations(dict_of_data=bin_grouped_dict_1D, bin_name='dn_170_3400')


# FIGURE: representation error timeseries
network = networkDesign(bin_grouped_dict_1D, bin_headers=['dn_170_300', 'dn_300_870', 'dn_870_3400'])
network.plot_representation_timeseries()

# FIGURE: rep error bars
network.plot_representation_bars()


"""



