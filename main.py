"""
Main file for the analysis of POPS data.

Analysis tools are ONLY compatible with the postcorrected POPS datasets are publically available on the ARM Data Discovery:
https://adc.arm.gov/discovery/#/results/iopShortName::amf2021SAILCAIVIMT/instrument_code::pops 
as netCDF files.
"""

#### DO I NEED TO DETREND DATA TO SEE COV...idea: normalize data by taking difference and look at timeseries and cov###

# import packages
from dataHandling import dataRetrival, dataGroupings
from networkMeanAnalysis import basicVisualization, temporalAnalysis
from spatialAnalysis import timeseriesVisualization, spatialVariability, networkDesign


# set up date range and sites for analysis
start_date = '20211010'
end_date = '20230722'

sites = ['cbmid', 'gothic', 'pumphouse', 'snodgrass', 'irwin', 'cbtop']

# load data
dr = dataRetrival()
data_dict = dr.create_datasets(sites, start_date, end_date, subsample=12)

# time bin data
grouping = dataGroupings()
grouped_dict = {}
for site in sites:
    grouped_dict[site] = grouping.temporal_grouping(data_dict[site], '1D')



# get network mean
network_df = grouping.network_mean(grouped_dict)

network_grouped = grouping.bin_groupings(network_df, grouping_option=2)

data_groupings = dataGroupings()

group = {}
for site in sites:
    group[site] = data_groupings.bin_groupings(grouped_dict[site], grouping_option=2)



sv = spatialVariability()
sv.sudo_variogram(group, bin_names=['dn_170_300', 'dn_300_870', 'dn_870_3400'], distance_type='horizontal', sum_headers=False)
# cv = sv.coefficient_of_variation(group, bin_names=['dn_170_3400'], rolling=5, sum_headers=False)
# cv = sv.coefficient_of_variation(group, bin_names=['dn_170_300', 'dn_300_870', 'dn_870_3400'], rolling=7, sum_headers=False)
# cv = sv.coefficient_of_variation(group, bin_names=['dn_170_300', 'dn_300_870', 'dn_870_3400'], rolling=7)


# vis = timeseriesVisualization()
# vis.plot_timeseries_together(group, bin_name='dn_170_3400')




# sv.sudo_variogram(grouped_dict, ['b3', 'b4', 'b5'], distance_type='euclidean', sum_headers=False)

# ta = temporalAnalysis()
# ta.plot_monthly_diurnal(cv, bin_names=['b3', 'b4'])

# network = networkDesign(group, bin_headers=['dn_170_300', 'dn_300_870', 'dn_870_3400'])
# network.plot_representation_timeseries()
# network.plot_representation_bars()





