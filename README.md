# SAIL-Net POPS Data Analysis

### Introduction to SAIL-Net

SAIL-Net is a DOE funded project in the East River Watershed near Crested Butte, Colorado with the goal of advancing our understanding of aerosol-cloud interactions in complex, mountainous regions. 
Through the deployment of a network of six low cost microphysics nodes in Fall 2021 in the same domain at the SAIL campaign, SAIL-Net provides data on aerosol size distributions, cloud condensation nuclei (CCN), and ice nucleating particles (INP). 
This network enables the investigation of small-scale variations in complex terrain, thus enhancing our understanding of aerosol's role in precipitation formation and water resource dynamics.

### Introduction to the dataset and code

The code here can be used to preform basic spatiotemporal data analysis and visualization of the POPS data, which are available on the [ARM Data Discovery](https://adc.arm.gov/discovery/#/results/iopShortName::amf2021SAILCAIVIMT/instrument_code::pops) as netCDF files.
These POPS data consist of over 1.5 years of aerosol size data (specifically PM 2.5) at 5 second resolution.
The POPS itself measures aerosol at 1 second frequency, the but data to be used with this code have been postcorrected, cleaned, and subsampled to 5-second resolution for speed of analysis.
This is the code that I have used for all data analysis thus far, and my hope is this code can be used by others as a foundation for further analysis of this valuable dataset. 

### Using the code

To use this code, download all postcorrected and cleaned POPS data from the ARM Data Discovery and put in a directory within the sailnet_pops_data_analysis directory called ''data''.
From there, all analysis can be done in the main.py file.
In its current state, the main.py file provides an example of how one would work with the code.

To view the analysis and figures associated with the SAIL-Net paper "SAIL-Net: An investigation of the spatiotemporalvariability of aerosol in the mountainous terrain of the Upper Colorado River Basin" (Gibson, et al., 2024), run the code in acp_figures_and_analysis.py. 
This code will produce all figures and analysis of the paper, except for the figures and analysis associated with the comparison to TBS data. To see these figures, download the [TBS data]{https://adc.arm.gov/discovery/#/results/s::tbspops} from the ARM Data Discovery, put it in a directory called ''TBS_data'' and run the TBSAnalysis.py file.


### A note to the user

The acp_figures_and_analysis.py and TBSAnalysis.py files should run without bugs.
However, there is a possibility for the user to experience bugs if using the code for their own analysis.
If bugs are found, please let me know.


