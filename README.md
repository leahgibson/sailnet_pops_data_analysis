# SAIL-Net POPS Data Analysis

### Introduction to SAIL-Net

SAIL-Net is a DOE funded project in the East River Watershed near Crested Butte, Colorado with the goal of advancing our understanding of aerosol-cloud interactions in complex, mountainous regions. 
Through the deployment of a network of six low cost microphysics nodes in Fall 2021 in the same domain at the SAIL campaign, SAIL-Net provides data on aerosol size distributions, cloud condensation nuclei (CCN), and ice nucleations particles (INP). 
This network enables the investigation of small-scale variations in complex terrain, thus enhancing our understanding of aerosol's role in precipitation formation and water resource dynamics.

### Introduction to the Dataset and Code

The code here can be used to preform basic spatiotemporal data analysis and visualization of the POPS data, which are publicallyl avaiilable on the [ARM Data Discovery](https://adc.arm.gov/discovery/#/results/iopShortName::amf2021SAILCAIVIMT/instrument_code::pops) as netCDF files.
These POPS data consist of over 1.5 years of aerosol size data (specifically PM 2.5) at 5 second resolution.
The POPS itself measures aerosol at 1 second frequency, the but data to be used with this code have been postcorrected, cleaned, and subsampled to 5-second resolution for speed of analysis.
This is the code that I have used for all data analysis thus far, and my hope is this code can be used by others as a foundation for futher analysis of this valuable dataset. 

### Using the Code

To use this code, download all postcorrected and cleaned POPS data from the ARM Data Discovery and put in a directory within the sailnet_pops_data_analysis directory called ''data''.
From there, all analysis can be done in the main.py file.

This code is still a work in progress.





