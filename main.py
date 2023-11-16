"""
Main file for the analysis of POPS data.

Analysis tools are ONLY compatible with the postcorrected POPS datasets are publically available on the ARM Data Discovery:
https://adc.arm.gov/discovery/#/results/iopShortName::amf2021SAILCAIVIMT/instrument_code::pops 
as netCDF files.
"""

# import packages
from dataHandling import dataRetrival


# set up date range and sites for analysis
start_date = '20221003'
end_date = '20221004'

sites = ['cbtop', 'cbmid']

# load data
dr = dataRetrival()
data_dict = dr.create_datasets(sites, start_date, end_date, subsample=12)






