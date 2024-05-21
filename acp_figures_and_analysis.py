""" 
Code author: Leah Gibson
Code to generate figures in SAIL-Net: An investigation of the spatiotemporal variability of aerosol in the 
mountainous terrain of the Upper Colorado River Basin by Gibson et al.

Data are accessible on the ARM Data Discovery website:
https://adc.arm.gov/discovery/#/results/iopShortName::amf2021SAILCAIVIMT/instrument_code::pops 
"""

# import packages
from dataHandling import POPSDataRetrival, dataGroupings, dataCompletenessVisualization
from networkMeanAnalysis import basicVisualization, temporalAnalysis
from spatialAnalysis import timeseriesVisualization, spatialVariability, networkDesign


# set up date range and sites for analysis
start_date = '20211010'
end_date = '20230722'

sites = ['pumphouse', 'gothic', 'cbmid', 'irwin', 'snodgrass', 'cbtop']

# load data
dr = POPSDataRetrival()
data_dict = dr.create_datasets(sites=sites, start_date=start_date, end_date=end_date, subsample=12)


# time bin data
grouping = dataGroupings()
time_grouped_dict_1H = {}   # group data by hours
time_grouped_dict_1D = {}   # group data by 1 day intervals
time_grouped_dict_15D = {}
for site in sites:
    time_grouped_dict_1H[site] = grouping.temporal_grouping(data_dict[site], averaging_frequency='1H')
    time_grouped_dict_1D[site] = grouping.temporal_grouping(data_dict[site], averaging_frequency='1D')
    #time_grouped_dict_15D[site] = grouping.temporal_grouping(data_dict[site], averaging_frequency='15D')


# group bins 
bin_grouped_dict_1H = {}
bin_grouped_dict_1D = {}
bin_grouped_dict_15D = {}
for site in sites:
    bin_grouped_dict_1H[site] = grouping.bin_groupings(time_grouped_dict_1H[site], grouping_option=2)
    bin_grouped_dict_1D[site] = grouping.bin_groupings(time_grouped_dict_1D[site], grouping_option=2)
    #bin_grouped_dict_15D[site] = grouping.bin_groupings(time_grouped_dict_15D[site], grouping_option=2)

# FIGURE 3: data completion
data_completeness = dataCompletenessVisualization()
data_completeness.plot_total_completeness(bin_grouped_dict_1D, bin_name='dn_170_3400')


# FIGURE 4: timeseries of daily averaged POPS data for all six sites
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

# FIGURE 5: network mean of POPS overlaid by day
network_basic_vis = basicVisualization()
network_basic_vis.plot_overlapping_timeseries(bin_grouped_network_mean_1D, bin_name='dn_170_3400')


# FIGURE 7: average particle size distribution averaged monthly
network_analysis.plot_monthly_psd(network_mean_1H)

# FIGURE 8: timeseries of PSD from the network mean
network_analysis.plot_psd_timeseries(network_mean_1D)


# FIGURE 10: average percent diff
spatial_analysis = spatialVariability()
spatial_analysis.sudo_variogram(bin_grouped_dict_1D, bin_names=['dn_170_300', 'dn_300_870', 'dn_870_3400'], distance_type='vertical', sum_headers=False)

# FIGURE 11: coefficient of variation of data
#spatial_analysis.coefficient_of_variation(bin_grouped_dict_1D, bin_names=['dn_170_3400'], sum_headers=False)
spatial_analysis.coefficient_of_variation(bin_grouped_dict_1D, bin_names=['dn_170_3400'], rolling=30, sum_headers=False)

#spatial_analysis.plot_maximum_range(dict_of_data=bin_grouped_dict_1D, bin_name='dn_170_3400', window=30)


# # FIGURE: max and min counts
# spatial_analysis.compute_number_max_min_concentrations(dict_of_data=bin_grouped_dict_1D, bin_name='dn_170_3400')


# FIGURE 12: representation error timeseries
network = networkDesign(bin_grouped_dict_1D, bin_headers=['dn_170_300', 'dn_300_870', 'dn_870_3400'])
network.plot_representation_timeseries()

# FIGURE 13: rep error bars
network.plot_representation_bars()





# FIGURE 6: diurnal cycles avg monthly without wildfire smoke in June 2022

# set up date range and sites for analysis
start_date = '20211010'
end_date = '20230722'

sites = ['pumphouse', 'gothic', 'cbmid', 'irwin', 'snodgrass', 'cbtop']

# load data
dr = POPSDataRetrival()
remove_dates=['20220613', '20220614', '20220615']
data_dict = dr.create_datasets(sites=sites, start_date=start_date, end_date=end_date, subsample=12, remove_dates=remove_dates)

# time bin data
grouping = dataGroupings()
time_grouped_dict_1H = {}   # group data by hours # group data by 1 day intervals

for site in sites:
    time_grouped_dict_1H[site] = grouping.temporal_grouping(data_dict[site], averaging_frequency='1H')

# group bins 
bin_grouped_dict_1H = {}

for site in sites:
    bin_grouped_dict_1H[site] = grouping.bin_groupings(time_grouped_dict_1H[site], grouping_option=2)


# compute network mean
network_mean_1H = grouping.network_mean(time_grouped_dict_1H)

# bin groupings of network mean
bin_grouped_network_mean_1H = grouping.bin_groupings(network_mean_1H, grouping_option=2)

# print network mean stats
network_analysis = temporalAnalysis()


# FIGURE 6: daily diurnal cycle averaged monthly
network_analysis.plot_monthly_diurnal(bin_grouped_network_mean_1H, bin_names=['dn_170_3400'])
 








# FIGURE 9: examples of variability

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

plt.rcParams.update({
    'font.size': 8,               # Font size for general text
    'axes.titlesize': 8,          # Font size for plot titles
    'axes.labelsize': 8,          # Font size for axis labels
    'xtick.labelsize': 7,         # Font size for x-axis ticks
    'ytick.labelsize': 8,         # Font size for y-axis ticks
    'legend.fontsize': 8,         # Font size for legend
    'lines.linewidth': 1.5        # Set linewidth 
})

fig, axs = plt.subplots(nrows=3, figsize=(6.6,4), dpi=300)
colors = plt.cm.viridis(np.linspace(0, 1, 6))
 
# set up date range and sites for analysis
start_date = '20220104'
end_date = '20220105'
sites = ['cbmid', 'irwin']

# load data
dr = POPSDataRetrival()
data_dict = dr.create_datasets(sites=sites, start_date=start_date, end_date=end_date, subsample=12)

# time bin data
grouping = dataGroupings()
time_grouped_dict_5Min = {}   # group data by 5 min
for site in sites:
    time_grouped_dict_5Min[site] = grouping.temporal_grouping(data_dict[site], averaging_frequency='5Min')

# group bins 
data_5Min = {}
for site in sites:
    data_5Min[site] = grouping.bin_groupings(time_grouped_dict_5Min[site], grouping_option=2)

# first subplot
for i, site in enumerate(sites):
    data = data_5Min[site]['dn_155_170'] + data_5Min[site]['dn_170_300']
    axs[0].plot(data_5Min[site]['DateTime'], data, color=colors[2*i], label=site)
axs[0].legend(loc='upper left', ncol=2)
axs[0].set_ylabel('cm$^{-3}$')
axs[0].xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))


# set up date range and sites for analysis
start_date = '20220525'
end_date = '20220612'
sites = ['gothic', 'pumphouse']

# load data
dr = POPSDataRetrival()
data_dict = dr.create_datasets(sites=sites, start_date=start_date, end_date=end_date, subsample=12)

# time bin data
grouping = dataGroupings()
time_grouped_dict_5Min = {}   # group data by 5 min
for site in sites:
    time_grouped_dict_5Min[site] = grouping.temporal_grouping(data_dict[site], averaging_frequency='15Min')

# group bins 
data_5Min = {}
for site in sites:
    data_5Min[site] = grouping.bin_groupings(time_grouped_dict_5Min[site], grouping_option=2)

# second subplot
for i, site in enumerate(sites):
    axs[1].plot(data_5Min[site] ['DateTime'], data_5Min[site]['dn_170_3400'], color=colors[(2*i) + 1], label=site)

axs[1].legend(loc='upper right', ncol=2)
axs[1].set_ylabel('cm$^{-3}$')
axs[1].xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))


# set up date range and sites for analysis
start_date = '20220613'
end_date = '20220614'
sites = ['irwin', 'snodgrass']

# load data
dr = POPSDataRetrival()
data_dict = dr.create_datasets(sites=sites, start_date=start_date, end_date=end_date, subsample=12)

# time bin data
grouping = dataGroupings()
time_grouped_dict_5Min = {}   # group data by 5 min
for site in sites:
    time_grouped_dict_5Min[site] = grouping.temporal_grouping(data_dict[site], averaging_frequency='5Min')

# group bins 
data_5Min = {}
for site in sites:
    data_5Min[site] = grouping.bin_groupings(time_grouped_dict_5Min[site], grouping_option=2)

# second subplot
for i, site in enumerate(sites):
    axs[2].plot(data_5Min[site] ['DateTime'], data_5Min[site]['dn_170_3400'], color=colors[i+4], label=site)

axs[2].legend(loc='upper left', ncol=3)
axs[2].set_ylabel('cm$^{-3}$')
axs[2].set_xlabel('Time (UTC)')
axs[2].xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))

plt.show()




# FIGURE 1: map of sites

import matplotlib.pyplot as plt
from pyproj import CRS, Transformer
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import geopandas as gpd
from shapely.geometry import Point
import contextily as cx
from matplotlib_scalebar.scalebar import ScaleBar


# Create a Cartopy map with OSM (OpenStreetMap) tiles


# Define the coordinates of the locations (lon, lat)
locations = {
    'Pumphouse': (-106.9495, 38.9211),
    'Gothic': (-106.9858, 38.9561),
    'CBMid': (-106.9431, 38.8983),
    'Irwin': (-107.1087, 38.8874),
    'Snodgrass': (-106.9905, 38.9271),
    'CBTop': (-106.9450, 38.8888)
}

# reorganize dict
site_locations = {}
site_locations['Name'] = []
site_locations['geometry'] = []
for name, coords in locations.items():
    site_locations['Name'].append(name)
    point = Point(coords)
    site_locations['geometry'].append(point)



# convert to gdf
gdf = gpd.GeoDataFrame(site_locations, geometry='geometry', crs="EPSG:4326")
gdf = gdf.set_geometry('geometry')
wm_gdf = gdf.to_crs('EPSG:3857')


fig, ax = plt.subplots(figsize=(6,4), dpi=300)
wm_gdf.plot(ax=ax, color='red', column='Name')
cx.add_basemap(ax, zoom=12, source=cx.providers.OpenStreetMap.Mapnik)

ax.set_xticks([])
ax.set_yticks([])

# annotate points with names
for x, y, label in zip(wm_gdf.geometry.x, wm_gdf.geometry.y, wm_gdf['Name']):
    ax.text(x+100, y+200, label, fontsize=7, ha='right')

# Add scale bar
scalebar = ScaleBar(1, location='upper left', scale_loc='top')
ax.add_artist(scalebar)
plt.tight_layout()
plt.show()

