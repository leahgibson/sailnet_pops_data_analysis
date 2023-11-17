"""
Main file for the analysis of POPS data.

Analysis tools are ONLY compatible with the postcorrected POPS datasets are publically available on the ARM Data Discovery:
https://adc.arm.gov/discovery/#/results/iopShortName::amf2021SAILCAIVIMT/instrument_code::pops 
as netCDF files.
"""

# import packages
from dataHandling import dataRetrival, dataGroupings
from networkMeanAnalysis import basicVisualization, temporalAnalysis


# set up date range and sites for analysis
start_date = '20220620'
end_date = '20230115'

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
print(network_df)


# group data
group = grouping.bin_groupings(network_df, grouping_option=1)
print(group)

ta = temporalAnalysis()
ta.plot_monthly_diurnal(group, bin_names=['total', 'dn_140_170'])
ta.plot_monthly_psd(network_df)
ta.plot_monthly_bin_average(group, ['dn_140_170', 'dn_200_300', 'dn_870_3400'])








