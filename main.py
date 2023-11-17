"""
Main file for the analysis of POPS data.

Analysis tools are ONLY compatible with the postcorrected POPS datasets are publically available on the ARM Data Discovery:
https://adc.arm.gov/discovery/#/results/iopShortName::amf2021SAILCAIVIMT/instrument_code::pops 
as netCDF files.
"""

# import packages
from dataHandling import dataRetrival, dataGroupings
from networkMeanAnalysis import basicVisualization, temporalAnalysis
from spatialAnalysis import spatialVariability, networkDesign


# set up date range and sites for analysis
start_date = '20220620'
end_date = '20220701'

sites = ['cbtop', 'cbmid']

# load data
dr = dataRetrival()
data_dict = dr.create_datasets(sites, start_date, end_date, subsample=12)

# time bin data
grouping = dataGroupings()
grouped_dict = {}
for site in sites:
    grouped_dict[site] = grouping.temporal_grouping(data_dict[site], '30Min')


# get network mean
network_df = grouping.network_mean(grouped_dict)


# group data
group = grouping.bin_groupings(network_df, grouping_option=1)


sv = spatialVariability()
cv = sv.coefficient_of_variation(grouped_dict, ['b3', 'b4'])
print(cv)

ta = temporalAnalysis()
ta.plot_monthly_diurnal(cv, bin_names=['b3', 'b4'])





