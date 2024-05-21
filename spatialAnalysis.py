"""
Used for all spatial analysis.
"""

# Python packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from geopy.distance import geodesic
import matplotlib.dates as mdates
from scipy import stats

# Set the font size for different plot elements
plt.rcParams.update({
    'font.size': 8,               # Font size for general text
    'axes.titlesize': 10,          # Font size for plot titles
    'axes.labelsize': 8,          # Font size for axis labels
    'xtick.labelsize': 8,         # Font size for x-axis ticks
    'ytick.labelsize': 8,         # Font size for y-axis ticks
    'legend.fontsize': 8,         # Font size for legend
    'lines.linewidth': 2.5         # Set linewidth 
})

class timeseriesVisualization:
    """
    Class for basic plotting of timeseries with multiple sites.
    """

    def __init__(self):
        
        # colorblind friendly colors
        # self.colors = ['#377eb8', '#ff7f00', '#4daf4a',
        #           '#f781bf', '#a65628', '#984ea3',
        #           '#999999', '#e41a1c', '#dede00']
        
        self.colors = plt.cm.viridis(np.linspace(0, 1, 6))

    def plot_timeseries_together(self, dict_of_data, bin_name):
        """
        Plots timeseries of data for multiple sites.
        
        Inputs:
        - dict_of_data: dict of site data
        - bin_name: (str) name of bin to plot

        Output: plot

        Returns: none
        """


        # structure for getting overall max and min
        overall_max = [0, 'site']
        overall_min = [1000, 'site']


        fig, ax = plt.subplots(figsize=(6.6,3), dpi=300)
        for idx, (site, df) in enumerate(dict_of_data.items()):
            df['DateTime'] = pd.to_datetime(df['DateTime'])
            ax.plot(df['DateTime'], df[bin_name], linewidth=1.5, color=self.colors[idx], label=site)

            # get max and min
            max = df[bin_name].max()
            min = df[bin_name].min()

            if max > overall_max[0]:
                overall_max = [max, site]
            if min < overall_min[0]:
                overall_min = [min, site]

        
        ax.legend()
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax.set_xlabel('UTC')
        ax.set_ylabel('cm$^{-3}$')
        ax.set_title(bin_name)
        plt.show()

        # return max and min values, which site, and when
        max_site = overall_max[1]
        max_index = dict_of_data[max_site][bin_name].idxmax()
        max_date = dict_of_data[max_site].loc[max_index, 'DateTime']

        min_site = overall_min[1]
        min_index = dict_of_data[min_site][bin_name].idxmin()
        min_date = dict_of_data[min_site].loc[min_index, 'DateTime']

        print(f'The maximum concentration occurs at {max_site} on {max_date} wirh a concentration of {overall_max[0]}.')
        print(f'The minimum concentration occurs at {min_site} on {min_date} with a concentration of {overall_min[0]}.')

    
    def plot_timeseries_one_by_one(self, dict_of_data, bin_name):
        """
        Plots each line in a separate plot.

         Inputs:
        - dict_of_data: dict of site data
        - bin_name: (str) name of bin to plot

        Output: plot

        Returns: none
        """

        fig, axs = plt.subplots(nrows=len(dict_of_data.items()), sharex=True, sharey=True)
        for index, (site, df) in enumerate(dict_of_data.items()):
            axs[index].plot(df['DateTime'], df[bin_name])
            axs[index].set_title(site)
        plt.gca().xaxis.set_major_locator(ticker.AutoLocator())
        plt.show()


    
    def difference_data(self, dict_of_data, bin_names):
        """
        Calculates the difference between adjacent points as a
        way of normalizing the data.
        
        Inputs:
        - dict_of_data: dict of site data
        - bin_name: list of names of bin to plot

        Output: none

        Returns: dict of differenced data
        """

        differenced_dict = {}
        for site, df in dict_of_data.items():
            differenced_dict[site] = pd.DataFrame()
            differenced_dict[site]['DateTime'] = df['DateTime']
            for bin in bin_names:
                differenced_dict[site][bin] = df[bin].diff().abs()
        
        return differenced_dict



class spatialVariability:
    """
    Class for the analysis of variability in the data, such as
    - coefficient of variation
    - sudo variograms (coined this term myself)
    """

    def __init__(self):
        # elevation and location data
        self.locations = {
            'pumphouse':(38.92108, -106.94949),
            'gothic':(38.95615, -106.98582),
            'cbmid':(38.89828, -106.94312),
            'irwin':(38.88738, -107.10870),
            'snodgrass':(38.92713, -106.99050),
            'cbtop':(38.88877, -106.94501)
        }

        self.elevations = {
            'pumphouse':2765,
            'gothic':2918,
            'cbmid':3137,
            'irwin':3177,
            'snodgrass':3333,
            'cbtop':3482
        }

        self.years = [2021, 2022, 2023]
        self.months = [1,2,3,4,5,6,7,8,9,10,11,12]

        # colorblind friendly colors
        # self.colors = ['#377eb8', '#ff7f00', '#4daf4a',
        #           '#f781bf', '#a65628', '#984ea3',
        #           '#999999', '#e41a1c', '#dede00']

        self.colors = plt.cm.viridis(np.linspace(0, 1, 6))

    def coefficient_of_variation(self, dict_of_data, bin_names, rolling=None, sum_headers=True):
        """
        Treats the site data at time t as a data set and computes the coefficient of variation.
        Plots the timeseries of these computations to see how CV changes over time.

        First normalizes the data using min-xav scaling.

        Inputs:
        - dict_of_data: dictionary of site data
        - bin_names: list of bin names to use when computing CV
        - rolling: default 'None' or put number of points to use in rolling mean
        - sum_headers: (bool) sums the bin name columns, default True

        Outputs: single plot with different lines for different bins

        Returns: df of the CV
        """

         # compute cov for each bin and plot on subplot
        
        cv_df = pd.DataFrame()

        if sum_headers:
            analysis_df = pd.DataFrame()
            sites = []
            for site, df in dict_of_data.items():
                sites.append(site)
                analysis_df[site] = df[bin_names].sum(axis=1, skipna=False) # if nan, result is nan
            analysis_df['DateTime'] = df['DateTime']
            analysis_df = analysis_df.set_index('DateTime')


            # add to cv_df
            cv_df['cov'] = cov.to_frame(name='cov')

            # plot
            plt.plot(cov)
            plt.gca().xaxis.set_major_locator(ticker.AutoLocator())
            plt.ylabel('Coefficient of Variation')
            plt.show()

        ### also normalizes data using min-max scaling ###
        else:
            for bin in bin_names:
                analysis_df = pd.DataFrame()
                sites = []
                for site, df in dict_of_data.items():
                    sites.append(site)
                    analysis_df[site] = df[bin]


                analysis_df['DateTime'] = df['DateTime']
                analysis_df = analysis_df.set_index('DateTime')


                # apply min_max scaling
                site_mins = analysis_df.min(axis=1, skipna=True)
                site_maxs = analysis_df.max(axis=1, skipna=True)

                # scale rows (i.e. eaxh timestep) using (x-min)/(max-min)
                scaled_data = (analysis_df.T - site_mins)/(site_maxs- site_mins)   
                # transpose back
                analysis_df = scaled_data.T



                # compute cov
                cov = analysis_df.std(axis=1)/analysis_df.mean(axis=1)

                # add to column in df
                cv_df[bin] = cov.to_frame(name=bin)

                # plot
                fig, ax = plt.subplots(figsize=(6.6,3), dpi=300)
                ax.plot(cov, linewidth=1.5, label=bin)
            
                # Set x-axis major tick locator
                ax.set_xlabel('UTC')
                ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
                ax.set_ylabel('Coefficient of Variation')
                ax.legend()
                #plt.show()

        if rolling is not None:
            rolling_cov = cv_df.rolling(window=rolling, min_periods=1).mean()
            print(rolling_cov)
            if sum_headers:
                # compute n rolling mean
                ax.plot(rolling_cov, linewidth=1.5, color='orange')
                #plt.gca().xaxis.set_major_locator(ticker.AutoLocator())
                #plt.ylabel('Coefficient of Variation')
                plt.show()
            else:
                for bin in bin_names:
                    ax.plot(rolling_cov[bin], linewidth=1.5, color='orange', label='rolling mean')
                    #plt.gca().xaxis.set_major_locator(ticker.AutoLocator())
                    #plt.ylabel('Coefficient of Variation')
                plt.legend()
                plt.show()
        else:
            plt.show()



        # make datetime index an actual column
        cv_df = cv_df.reset_index()

        
    
        return cv_df

    def plot_maximum_range(self, dict_of_data, bin_name, window=None):
        """
        Given data dict and the desired header, plots the range within the data for each time step.
        
        Inputs:
        - dict_of_data: dict of SAIL data
        - header: string of the header to use
        - window: int or None

        Returns: plot
        """

        analysis_df = pd.DataFrame()
        sites = []
        for site, df in dict_of_data.items():
            sites.append(site)
            analysis_df[site] = df[bin_name]


        analysis_df['DateTime'] = df['DateTime']
        analysis_df = analysis_df.set_index('DateTime')

        # compute the range of each row
        ranges = analysis_df.apply(self._row_range, axis=1)

        print(ranges)

        plt.plot(ranges, label='Range')
        plt.gca().xaxis.set_major_locator(ticker.AutoLocator())


        if window is not None:
            rolling_range = ranges.rolling(window=window, min_periods=1).mean()
            plt.plot(rolling_range, label='Rolling Mean')
            plt.legend()
            plt.show()
        else:
            plt.ylabel('Ranges')
            plt.show()



    def _row_range(self, row):
        return row.max() - row.min()

    def sudo_variogram(self, dict_of_data, bin_names, distance_type, sum_headers=True):
        """
        Plots the average percent difference between pairs of sites as a function
        of either their vertical difference, distance between them, or 
        normalized euclidean distance.

        Plots the result and computes the Pearson r correlation.
        
        Inputs:
        - dict_of_data: dictionary of site data
        - bin_names: list of bin headers
        - distance_type: str to represent the distance used, pick from:
            - 'horizontal'
            - 'vertical'
            - 'euclidean' 
        - sum_headers: bool, default True to sum the columns provided in bin_names
        
        Output: figure

        Returns: none
        """

        if sum_headers:
            # create single df
            data = pd.DataFrame()
            sites = []
            for site, df in dict_of_data.items():
                sites.append(site)
                data[site] = df[bin_names].sum(axis=1, skipna=False)
            data['DateTime'] = df['DateTime']
            data = data.set_index('DateTime')

            # compute the necessary distance
            if distance_type == 'horizontal':
                distances = self._compute_horizontal_distance(sites)
                distance_name = 'Distance (km)'
            
            if distance_type == 'vertical':
                distances = self._compute_vertical_distance(sites)
                distance_name = 'Elevation Difference (m)'
            
            if distance_type == 'euclidean':
                distances = self._compute_euclidean_distance(sites)
                distance_name = 'Normalized Euclidean Difference'

            print('the distances:', distances)
            
            # compute the absolute differene between normalized by the mean values; ordering based on ordering of distances
            diff_df = pd.DataFrame()
            distances_list = []
            for key, dist in distances.items():
                distances_list.append(dist)
                names = key.split('_') # get two names in dict key
                diffs = (data[names[0]] - data[names[1]]).abs()
                avg = data[[names[0], names[1]]].mean(axis=1)
                diff_df[key] = (diffs/avg)*100


            
            # compute average of all columns for avg distance
            mean_diffs = diff_df.mean().tolist()

            print('the differences:', mean_diffs)

            
            # plot
            # for i, row in diff_df.iterrows():
            #     plt.plot(distances_list, row, marker='o', linestyle='None', color='gray', alpha=0.5)
                
            # plot avg diffs
            plt.plot(distances_list, mean_diffs, marker='o', markersize='26', linestyle='None', color='green')
            # do linear regressions
            slope, intercept, r_value, p_value, std_err = stats.linregress(distances_list, mean_diffs)
            #print('slope=',slope,'intercept=', intercept)
            #print('mean diffs=', mean_diffs)
            regression_line = [slope*x + intercept for x in distances_list]
            plt.plot(distances_list, regression_line, color='black', linewidth=5)
            plt.title('r-value ' + str(round(r_value, 2)))
            plt.xlabel(distance_name)
            plt.ylabel('Percent Difference')
            plt.show()
        
        else:
        
            fig, axs = plt.subplots(ncols=len(bin_names), sharey=True, figsize=(6.6,3), dpi=300)
            colors=['b', 'g', 'r']
            for i, bin in enumerate(bin_names):
                # create single df
                data = pd.DataFrame()
                sites = []
                for site, df in dict_of_data.items():
                    sites.append(site)
                    data[site] = df[bin]
                data['DateTime'] = df['DateTime']
                data = data.set_index('DateTime')

                # compute the necessary distance
                if distance_type == 'horizontal':
                    distances = self._compute_horizontal_distance(sites)
                    distance_name = 'Distance (km)'
                
                if distance_type == 'vertical':
                    distances = self._compute_vertical_distance(sites)
                    distance_name = 'Elevation Difference (m)'
                
                if distance_type == 'euclidean':
                    distances = self._compute_euclidean_distance(sites)
                    distance_name = 'Normalized Euclidean Difference'

                print('the distances:', distances)
                
                # compute the absolute differene between normalized by the mean values; ordering based on ordering of distances
                diff_df = pd.DataFrame()
                distances_list = []
                for key, dist in distances.items():
                    distances_list.append(dist)
                    names = key.split('_') # get two names in dict key
                    diffs = (data[names[0]] - data[names[1]]).abs()
                    avg = data[[names[0], names[1]]].mean(axis=1)
                    diff_df[key] = (diffs/avg)*100
                
                
            
                # compute average of all columns for avg distance
                mean_diffs = diff_df.mean().tolist()

                print('Differences:',mean_diffs)

                
                # plot
                # for i, row in diff_df.iterrows():
                #     plt.plot(distances_list, row, marker='o', linestyle='None', color='gray', alpha=0.5)
                    
                # plot avg diffs
                axs[i].plot(distances_list, mean_diffs, marker='o', markersize='5', linestyle='None', color=self.colors[i*2])
                # do linear regressions
                slope, intercept, r_value, p_value, std_err = stats.linregress(distances_list, mean_diffs)
                #print('slope=',slope,'intercept=', intercept)
                #print('mean diffs=', mean_diffs)
                regression_line = [slope*x + intercept for x in distances_list]
                axs[i].plot(distances_list, regression_line, color='black', linewidth=2)
                axs[i].set_title(bin + ' r=' + str(round(r_value, 2)))
                if i==np.floor(len(bin_names)/2):
                    axs[i].set_xlabel(distance_name)
                if i==0:
                    axs[i].set_ylabel('Percent Difference')
            plt.show()

    def plot_sitess_monthly_diurnal(self, dict_of_data, bin_name):
        """
        Plots the average diurnal cycle for each site for each month of 2022 by 
        averaging over each data in the month.

        Note: there must be more than one month present for plot to work.
        
        Inputs:
        - dict_of_data: df of network mean
        - bin_name: name of bin to use in analysis
        
        Outputs: plot
        
        Returns: none
        """
       

        count=0
        for site, data in dict_of_data.items():

            # make groups into months 
            data['DateTime'] = pd.to_datetime(data['DateTime'])

            data['Time'] = data['DateTime'].dt.time
            data['Month'] = data['DateTime'].dt.month
            data['Year'] = data['DateTime'].dt.year
            data['Day'] = data['DateTime'].dt.day


            # number of months in data
            num_months = data['Month'].nunique()

            if count==0:
                 fig, axs = plt.subplots(nrows=1, ncols=num_months, sharex=True, sharey=True)

        
            daily_averages = data.groupby(['Year', 'Month', 'Time'])[bin_name].mean()
           
            i=0
            for year in self.years:
                for month in self.months:
                    try:
                        axs[i].plot(daily_averages[year][month].values, label=site)
                        axs[i].set_title(str(year)+'-'+str(month))
                        axs[i].set_xticks([0, 23, 47])
                        axs[i].set_xticklabels(['00:00', '12:00', '24:00'])

                        if i==0:
                            axs[i].legend()
                        
                        i+=1
                    except:
                        pass
            count+=1

        
                
        plt.show()
    
    def compute_number_max_min_concentrations(self, dict_of_data, bin_name):
        """
        For the given bin, counts the number of times that each site experiences the maximum and minimum concentrations
        of all of the sites.

        Inputs:
        - dict_of_data: dict of pops data
        - bin_name: name of bin

        Output: bar plot

        Returns: none
        """

        # make df of only the data needed
        analysis_df = pd.DataFrame()
        sites = []
        for site, df in dict_of_data.items():
            sites.append(site)
            analysis_df[site] = df[bin_name]        
        analysis_df['DateTime'] = df['DateTime']
        analysis_df = analysis_df.set_index('DateTime')

        # initialize counters for max and min occurrences
        max_count = {}
        min_count = {}
        for site in sites:
            max_count[site] = 0
            min_count[site] = 0
        
        # iterate over each time step
        for time in analysis_df.index:
            # find which site has the max and min conc
            max_site = analysis_df.loc[time].idxmax()
            min_site = analysis_df.loc[time].idxmin()


            # incrment the counters
            max_count[max_site] += 1
            min_count[min_site] += 1
        
        print('The final counts are:')
        print(f'Max count: {max_count}')
        print(f'Min count: {min_count}')

        # generate the bar chart
        fig, ax = plt.subplots()
        max_values = list(max_count.values())
        min_values = list(min_count.values())

        bar_width = 0.35
        bar_positions_max = range(len(sites))
        bar_positions_min = [pos + bar_width for pos in bar_positions_max]

        ax.bar(bar_positions_max, max_values, bar_width, label='Maximum Concentration', color='red')
        ax.bar(bar_positions_min, min_values, bar_width, label='Minimum Concentration', color='blue')

        ax.set_xticks([pos + bar_width / 2 for pos in bar_positions_max])
        ax.set_xticklabels(sites)
        ax.set_ylabel('Frequency')
        ax.legend()

        plt.show()





    def _compute_horizontal_distance(self, sites):
        distances = {}
        # compute the difference between all sites
        for i in range(len(sites)):
            for j in range(i+1, len(sites)):
                distances[sites[i]+'_'+sites[j]] = round(geodesic(self.locations[sites[i]], self.locations[sites[j]]).kilometers, 2)
        
        # sort dict from smallest to largest
        distances = dict(sorted(distances.items(), key=lambda item: item[1]))
        return distances

    def _compute_vertical_distance(self, sites):
        distances = {}
        # compute the elevation differences
        for i in range(len(sites)):
            for j in range(i+1, len(sites)):
                distances[sites[i]+'_'+sites[j]] = round(abs(self.elevations[sites[i]] - self.elevations[sites[j]]))
        
        # sort dict from smallest to largest
        distances = dict(sorted(distances.items(), key=lambda item: item[1]))
        return distances

    def _compute_euclidean_distance(self, sites):
        # compute elevation diffs and normalize
        distances = {}
        for i in range(len(sites)):
            for j in range(i+1, len(sites)):
                distances[sites[i]+'_'+sites[j]] = round(geodesic(self.locations[sites[i]], self.locations[sites[j]]).kilometers, 2)
        max_dist = max(distances.values())
        normalized_distances = {}
        for key, value in distances.items():
            normalized_distances[key] = value/max_dist
        
        
        # compute the elevation differences
        elevation_diffs = {}
        for i in range(len(sites)):
            for j in range(i+1, len(sites)):
                elevation_diffs[sites[i]+'_'+sites[j]] = round(abs(self.elevations[sites[i]] - self.elevations[sites[j]]))
        max_elevation_diff = max(elevation_diffs.values())
        normalized_elevation_diffs = {}
        for key, value in elevation_diffs.items():
            normalized_elevation_diffs[key] = value/max_elevation_diff
        
        distances = {}
        for key in normalized_distances:
            distances[key] = np.sqrt(normalized_distances[key]**2 + normalized_elevation_diffs[key]**2)

        # sort dict from smallest to largest
        distances = dict(sorted(distances.items(), key=lambda item: item[1]))
        
        return distances
    


class networkDesign:
    """
    Class for the analysis of the network as a whole. Features include:
    - representation error and analysis
    """

    def __init__(self, dict_of_data, bin_headers):
        """
        Calls function that computed the representation error.

        Creates list of sites, array of datetimes, and dict of rep error
        to be used by other functions.
        """
        self.sites, self.datetimes, self.representation_dict = self._compute_representation_error(dict_of_data, bin_headers)

        # self.colors = ['#377eb8', '#ff7f00', '#4daf4a',
        #           '#f781bf', '#a65628', '#984ea3',
        #           '#999999', '#e41a1c', '#dede00']

        self.colors = plt.cm.viridis(np.linspace(0, 1, 6))

    def plot_representation_timeseries(self):
        """
        Plots a timeseries of the representation error for all sites in the data dict
        for all headers provided when class was initialized.

        Output: plot

        Returns: none
        """

        # plot representation error timeseries
        fig, axs = plt.subplots(len(self.representation_dict), sharey=True, sharex=True, figsize=(6.6,4), dpi=300)

        for i, (bin, df) in enumerate(self.representation_dict.items()):
            for idx, site in enumerate(self.sites):
                axs[i].plot(self.datetimes, df[site], linewidth=1.5, color=self.colors[idx], label=site)
            # add label in subplot for the bin
            axs[i].text(0.02, 0.95, bin, transform=axs[i].transAxes, fontsize=10, va='top', ha='left')
        axs[0].legend(loc='upper right', ncol=3)
        axs[i].xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        axs[-1].set_xlabel('UTC')
        axs[1].set_ylabel('Representation Error')
        plt.show()

    def plot_representation_bars(self):
        """
        Averages over the representation timeseries to plot the average and range of the
        representation error.
        
        Inputs: none
        
        Outputs: plot
        
        Returns: none
        """

        # compute average and range of representation errors for plotting
        fig, axs = plt.subplots(ncols=len(self.sites), sharey=True, figsize=(6.6,2.5), dpi=300)

        named_colors = ['blue', 'green', 'red', 'black', 'magenta',
                            'yellow', 'cyan', 'gray', 'orange',
                            'purple', 'pink', 'brown']
        
        

        for i, (bin, df) in enumerate(self.representation_dict.items()):
            print(bin)
            for j, site in enumerate(self.sites):
                print(site)
                # make line for range
                axs[j].vlines(i+1, df[site].min(), df[site].max(), color=self.colors[i*2], linewidth=1.5, label=bin)
                print('range', abs(df[site].min() - df[site].max()))
                # plot dot for average
                axs[j].plot(i+1, df[site].mean(), color=self.colors[i*2], marker='^', markersize=3)
                print('mean', df[site].mean())
                # don't show x-ticks
                axs[j].set_xticks([])

                axs[j].set_xlabel(site)
        
        axs[0].set_ylabel('Representation Error')
        plt.legend()
        plt.show()


    def _compute_representation_error(self, dict_of_data, bin_headers):
        """
        Computes the actual representation error: the normalized difference
        between a site observation and the network mean:

        e = (observation - mean) / mean

        This function computes the representation error for every point in the iven data set for the specified bin_headers.    

        Inputs:
        - dict_of_data: dictionary of all POPS data
        - bin_headers: headers for bins to use in analysis

        Returns: list of site names, array of datetimes, 
        dict of representation error for each site
        """

        representation_dict = {}
        sites = list(dict_of_data.keys())
        for bin in bin_headers:
            # sort data by bins
            representation_dict[bin] = pd.DataFrame()
            for site, df in dict_of_data.items():
                representation_dict[bin][site] = df[bin]
        datetimes = df['DateTime']
        
        # compute rep error
        for bin, df in representation_dict.items():
            # compute avg column
            df['average'] = df.mean(axis=1, skipna=True) # change to False so do analysis when all sites have data 
            for site in sites:
                df[site] = (df[site] - df['average'])/df['average']
        
        

        return sites, datetimes, representation_dict
