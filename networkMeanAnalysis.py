"""
Analysis of the network mean timeseries.
"""

# import Python packages
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import matplotlib.colors as colors
from scipy.stats import gaussian_kde

# Set the font size for different plot elements
plt.rcParams.update({
    'font.size': 24,               # Font size for general text
    'axes.titlesize': 24,          # Font size for plot titles
    'axes.labelsize': 18,          # Font size for axis labels
    'xtick.labelsize': 18,         # Font size for x-axis ticks
    'ytick.labelsize': 18,         # Font size for y-axis ticks
    'legend.fontsize': 18,         # Font size for legend
    'lines.linewidth': 2.5         # Set linewidth 
})

class basicVisualization:
    """
    Class for basic visualization of data.
    """

    def __init__(self):
        self.standard_bins = ['b' + str(i) for i in range(16)]
        # for particle size distributions
        # self.dlogdp = [0.03645458169, 0.03940255269, 0.04033092159, 0.03849895488,
        #             0.03655010672, 0.04559350564, 0.08261548653, 0.06631586816,
        #             0.15575785, 0.1008071129, 0.1428650493, 0.1524763279,
        #             0.07769393472, 0.1571866015, 0.1130751916, 0.0867054262]
        
        # self.dlogdp = [0.035114496, 0.037103713, 0.040219114, 0.044828027, 0.050001836, 0.056403989, 0.129832168,
        #   0.137674163, 0.078941363, 0.09085512, 0.177187651, 0.137678593, 0.096164793, 0.112758467,
        #   0.107949615, 0.10986499]
        
        # CORRECTED USING UPDATED NOAA MIE TABLE
        self.dlogdp = [0.03645458169, 0.03940255269, 0.04033092159, 0.03849895488,
                    0.03655010672, 0.04559350564, 0.08261548653, 0.141566381,
                    0.080507337, 0.1008071129, 0.1428650493, 0.1559862,
                    0.112588743, 0.118781921, 0.1130751916, 0.0867054262]
        

        
        # self.diameter_midpoints = [149, 163, 178, 195, 213, 234, 272, 322, 422, 561, 748,
        #                     1054, 1358, 1802, 2440, 3062]
        
        self.diameter_midpoints = [149, 163, 178, 195, 213, 234, 272, 355, 455, 562, 749,
                            1059, 1431, 1870, 2440, 3062]
        
        # colorblind friendly colors
        self.colors = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

    def plot_network_timeseries(self, df, bin_name, rolling=None):
        """
        Basic plotting of network mean for specified bin.

        Inputs:
        - df: df of network mean
        - bin_name: name of bin to be plotted
        - rolling: default None or int for number of values to use in a rolling mean

        Output: plot

        Returns: none
        """


        if rolling is not None:
            timeseries_df = df.rolling(window=rolling, min_periods=1).mean()
            timeseries_df['DateTime'] = df['DateTime']
        else:
            timeseries_df = df


        plt.plot(timeseries_df['DateTime'], timeseries_df[bin_name])
        plt.gca().xaxis.set_major_locator(ticker.AutoLocator())
        plt.title(bin_name)
        plt.ylabel('cm$^{-3}$')
        plt.show()
    
    def plot_overlapping_timeseries(self, data, bin_name):
        """
        Plots the multiple years of data my overlaying them by day.
        
        Inputs:
        - data: df of notwork mean
        - bin_name: name of bin to be plotted

        Output: plot

        Returns: nothing
        """
        data['DateTime'] = pd.to_datetime(data['DateTime'])

        data['Year'] = data['DateTime'].dt.year

        year_groups = data.groupby('Year')

        fig, ax = plt.subplots()
        for idx, group in enumerate(year_groups):
            df = group[1]
            # replace all years with 2023
            df['DateTime'] = df['DateTime'].apply(lambda x: x.replace(year=2023))
            # now remove year
            #df['DateTime'] = pd.to_datetime(df['DateTime'].dt.strftime('%m-%d %H:%M:%S'))
            

            ax.plot(group[1]['DateTime'], group[1][bin_name], linewidth=2, color=self.colors[idx], label=str(group[0]))
        plt.legend()

        custom_ticks = [
        (datetime(2023, 1, 1, 0, 0, 0), "Jan"),
        (datetime(2023, 2, 1, 0, 0, 0), "Feb"),
        (datetime(2023, 3, 1, 0, 0, 0), 'March'),
        (datetime(2023, 4, 1, 0, 0, 0), 'April'),
        (datetime(2023, 5, 1, 0, 0, 0), 'May'),
        (datetime(2023, 6, 1, 0, 0, 0), 'June'),
        (datetime(2023, 7, 1, 0, 0, 0), 'July'),
        (datetime(2023, 8, 1, 0, 0, 0), "Aug"),
        (datetime(2023, 9, 1, 0, 0, 0), "Sept"),
        (datetime(2023, 10, 1, 0, 0, 0), "Oct"),
        (datetime(2023, 11, 1, 0, 0, 0), "Nov"),
        (datetime(2023, 12, 1, 0, 0, 0), "Dec")
        ]
        ax.set_xticks([tick[0] for tick in custom_ticks])
        ax.set_xticklabels([tick[1] for tick in custom_ticks])
        ax.set_title(bin_name)
        ax.set_ylabel('cm$^{-3}$')
        plt.show()

    def plot_psd(self, data, data2=None, data3=None):
        """
        Given the data, overages over all of it and plots a PSD 

        Option for adding a second psd

        Only takes 16 bin data
        
        Inputs:
        - data: df of data
        - data2: df of second data, default None
        
        Returns: none
        
        Outputs: plot of raw data, smoothed curve, and curve overlaying raw
        
        """

        psd = []
        for i, bin in enumerate(self.standard_bins):
            bin_avg = data[bin].mean()
            dndlogdp_val = bin_avg/self.dlogdp[i]
            psd.append(dndlogdp_val)
        
        # plot raw distribution
        plt.loglog(self.diameter_midpoints, psd, marker='o', color='black', label='2022-06-14')
        
        if data2 is not None:
            psd2 = []
            for i, bin in enumerate(self.standard_bins):
                bin_avg = data2[bin].mean()
                dndlogdp_val = bin_avg/self.dlogdp[i]
                psd2.append(dndlogdp_val)
            
            plt.loglog(self.diameter_midpoints, psd2, marker='^', color='blue', label='2022-06-16')
            plt.legend()
        
        if data3 is not None:
            psd3 = []
            for i, bin in enumerate(self.standard_bins):
                bin_avg = data3[bin].mean()
                dndlogdp_val = bin_avg/self.dlogdp[i]
                psd3.append(dndlogdp_val)
            
            plt.plot(self.diameter_midpoints, psd3, marker='s', color='green', label='2023-05-23')
            plt.legend()
        
        
        
        plt.ylabel('dN/dlogD$_p$')
        plt.xlabel('Diameter (nm)')
        plt.show()

    def plot_different_time_segments(self, data1, data2, bin_name):
        """
        Accepts two dfs of same length and will plot on same plot.
        
        Data can be from different time periods.
        
        Inputs:
        - data1: df of first timeseries
        - data2: df of second timeseries
        
        Output: plot
        
        Returns: none
        """

        data1['DateTime'] = pd.to_datetime(data1['DateTime'])
        data2['DateTime'] = pd.to_datetime(data2['DateTime'])

        fig = plt.figure()
        ax1 = fig.add_subplot(111, label='1')
        ax2 = fig.add_subplot(111, label='2', frame_on=False)

        color1='black'
        ax1.plot(data1['DateTime'], data1[bin_name], color=color1)
        ax1.set_ylabel('cm$^{-3}$', color=color1)  
        ax1.set_xlabel('Date', color=color1)
        ax1.tick_params(axis='x', labelcolor=color1)
        ax1.tick_params(axis='y', labelcolor=color1)

        # second plot
        color2='blue'
        ax2.plot(data2['DateTime'], data2[bin_name], color=color2)
        ax2.xaxis.tick_top()
        ax2.yaxis.tick_right()
        ax2.set_ylabel('cm$^{-3}$', color=color2)       
        ax2.xaxis.set_label_position('top') 
        ax2.yaxis.set_label_position('right') 
        ax2.tick_params(axis='x', colors=color2)
        ax2.tick_params(axis='y', colors=color2)

        plt.show()


        # # create secondary xaxis
        # ax2 = ax1.twiny()
        # ax2.plot(data2['DateTime'], data2[bin_name], color='magenta')
        # ax2.tick_params(axis='x', labelcolor='magenta')

        # # create secondary y-axis
        # ax3 = ax1.twinx()
        # ax3.plot(data1['DateTime'], data2[bin_name], color='magenta')
        # ax3.tick_params(axis='y', labelcolor='magenta')



        plt.show()





class temporalAnalysis:
    """
    Class for more in depth analysis of temporal trends in the network mean.
    """

    def __init__(self):
        self.years = [2021, 2022, 2023]
        self.months = [1,2,3,4,5,6,7,8,9,10,11,12]
        self.standard_bins = ['b' + str(i) for i in range(16)]

        # for particle size distributions
        # self.dlogdp = [0.03645458169, 0.03940255269, 0.04033092159, 0.03849895488,
        #             0.03655010672, 0.04559350564, 0.08261548653, 0.06631586816,
        #             0.15575785, 0.1008071129, 0.1428650493, 0.1524763279,
        #             0.07769393472, 0.1571866015, 0.1130751916, 0.0867054262]
        
        # self.diameter_midpoints = [149, 163, 178, 195, 213, 234, 272, 322, 422, 561, 748,
        #                     1054, 1358, 1802, 2440, 3062]

        self.dlogdp = [0.03645458169, 0.03940255269, 0.04033092159, 0.03849895488,
                    0.03655010672, 0.04559350564, 0.08261548653, 0.141566381,
                    0.080507337, 0.1008071129, 0.1428650493, 0.1559862,
                    0.112588743, 0.118781921, 0.1130751916, 0.0867054262]
        
        self.diameter_midpoints = [149, 163, 178, 195, 213, 234, 272, 355, 455, 562, 749,
                            1059, 1431, 1870, 2440, 3062]
        
        # colorblind friendly colors
        self.colors = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']


    def basic_stats(self, data, bin_name):
        """
        Returns basic stats of the nework mean data for the given bin.
        
        Inputs:
        - data: df of network mean data
        - bin_name: name of bin header
        
        Prints:
        - maximum concentration and the day it occurs
        - minimum concentration and the day it occurs
        - average concentration 
        - average absolute percent change between timesteps
        """

        # max concentration data
        max_conc = data[bin_name].max()
        max_index = data[bin_name].idxmax()
        max_date = data.loc[max_index, 'DateTime']

        # min concentration data
        min_conc = data[bin_name].min()
        min_index = data[bin_name].idxmin()
        min_date = data.loc[min_index, 'DateTime']

        # average concentration
        avg_conc = np.mean(data[bin_name])
        # absolute percent change between time steps
        abs_percent_change = data[bin_name].pct_change().abs() # abs percent change = (|conc_{t+1} - conc_t|/conc_t)
        # compute average
        avg_change = (np.nanmean(abs_percent_change)*100)

        # # plot this data
        # plt.plot(abs_percent_change)
        # plt.title('Absolute Percent Change')
        # plt.show()

        print('BASIC STATISTICS') 
        print(f'Maximum Concentration: {max_conc} Occured on: {max_date}')
        print(f'Minimum Concentration: {min_conc} Occured on {min_date}')
        print(f'The average concentration is {avg_conc}')
        print(f'The average percent change between time steps is {avg_change}')
     
    

    def plot_monthly_diurnal(self, data, bin_names):
        """
        Plots the average diurnal cycle for each month of 2022 by 
        averaging over each day in the month.

        Note: there must be more than one month present for plot to work.
        
        Inputs:
        - data: df of network mean, should be binned hourly
        - bin_name: list of name of bins to analyze
            if data is cov data, use bin_name='' #### FILL THIS IN ###
        
        Outputs: plot, range and percent change of diurnal pattern for each month
        
        Returns: none
        """

        # make groups into months 
        data['DateTime'] = pd.to_datetime(data['DateTime'])

        data['Time'] = data['DateTime'].dt.time
        data['Month'] = data['DateTime'].dt.month
        data['Year'] = data['DateTime'].dt.year
        data['Day'] = data['DateTime'].dt.day

        # number of months in data
        num_months = data['Month'].nunique()
        

        daily_averages = {}
        for bin in bin_names:
            daily_averages[bin] = data.groupby(['Year', 'Month', 'Time'])[bin].mean()
        

        fig, axs = plt.subplots(nrows=1, ncols=num_months, sharex=True, sharey=True)
        #colors=['blue', 'orange', 'green']
        ranges = {}
        percent_changes = {}
        for bin in bin_names:
            idx = 0
            for year in self.years:
                
                for month in self.months:
                    try:
                        axs[month-1].plot(daily_averages[bin][year][month].values, label=bin, color=self.colors[idx])
                        axs[month-1].set_title(str(year)+'-'+str(month))
                        axs[month-1].set_xticks([0, 11, 23])
                        axs[month-1].set_xticklabels(['00:00', '12:00', '24:00'])

                        # compute the range for each month
                        max = np.nanmax(daily_averages[bin][year][month].values)
                        min = np.nanmin(daily_averages[bin][year][month].values)
                        range = max - min
                        percent_change = ((max-min)/min)*100

                        ranges[f'{bin}_{year}_{month}'] = range
                        percent_changes[f'{bin}_{year}_{month}'] = percent_change
                        
                        
                    except:
                        pass
                idx+=1
        axs[0].set_ylabel('cm$^{-3}$')
        
        # Create custom handles and labels for the legend
        legend_handles = [Line2D([0], [0], color='blue', lw=2),
                        Line2D([0], [0], color='orange', lw=2),
                        Line2D([0], [0], color='green', lw=2)]

        legend_labels = ['2021', '2022', '2023']

        # Create a legend with custom handles and labels
        axs[0].legend(handles=legend_handles, labels=legend_labels, loc='upper right')

        
                
        plt.show()

        # print(f'The ranges of each cycle are {ranges}')
        # print(f'The percent changes of each cycle are {percent_changes}')

    
    def plot_monthly_psd(self, data):
        """
        Plots the average monthly psd and the average normalized psd for each month.

        Note: only works for 16 bins and for 2022 year.
        
        Inputs:
        - data: df of network mean

        Outputs: twp diffrent plots

        Reutns: noone
        """

        
        
        # make groups into months 
        data['DateTime'] = pd.to_datetime(data['DateTime'])

        data['Time'] = data['DateTime'].dt.time
        data['Month'] = data['DateTime'].dt.month
        data['Year'] = data['DateTime'].dt.year
        data['Day'] = data['DateTime'].dt.day

        # number of months in data
        num_months = data['Month'].nunique()
        

        # organize dict to contain average for bin and months
        psd_dict = {}
        normalized_psd = {}

        # get averages for the sum
        total_averages = data.groupby(['Year', 'Month', 'Day'])['total'].mean()
        
  
        for i, bin in enumerate(self.standard_bins):
            # compute average over the days in the month for each bin
            daily_avgs = data.groupby(['Year', 'Month', 'Day'])[bin].mean()
            # avg over months
            for year in self.years:
                if str(year) in psd_dict:
                    pass
                else:
                    psd_dict[str(year)] = {}
                    normalized_psd[str(year)] = {}
                for month in self.months:
                    if str(month) in psd_dict[str(year)]:
                        pass
                    else:
                        psd_dict[str(year)][str(month)] = []
                        normalized_psd[str(year)][str(month)] = []
                    try:
                        # compute monthly average of bin
                        monthly_avg = np.mean(daily_avgs[year][month])
                        # convert to dndlogdp
                        dndlogdp_val = monthly_avg#/self.dlogdp[i]
                        psd_dict[str(year)][str(month)].append(dndlogdp_val)

                        # compute normalized average
                        monthly_total = np.mean(total_averages[year][month])
                        normalized_bin = monthly_avg/monthly_total
                        normalized_dndlogdp = normalized_bin/self.dlogdp[i]
                        normalized_psd[str(year)][str(month)].append(normalized_dndlogdp)
            
                    except:
                        pass
       
        
        month_names = ['Jan', 'Feb', 'March', 'April', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']
        
         # plot the particle dize distributions
        fig, axs = plt.subplots(nrows=1, ncols=12, sharex=True, sharey=True)
        #year_colors = ['blue', 'orange', 'green']
        idx=0
        for year in self.years:
            i=0
            for month in self.months:
                try:
                    axs[month-1].loglog(self.diameter_midpoints, psd_dict[str(year)][str(month)], color=self.colors[idx], label=str(year))
                    axs[month-1].set_title(month_names[(month)-1])

                    i+=1
                except:
                    pass
            idx+=1
        axs[0].set_ylabel('cm$^{-3}$')#('dn/dlogdp')
        # Create custom handles and labels for the legend
        legend_handles = [Line2D([0], [0], color='blue', lw=2),
                        Line2D([0], [0], color='orange', lw=2),
                        Line2D([0], [0], color='green', lw=2)]

        legend_labels = ['2021', '2022', '2023']

        # Create a legend with custom handles and labels
        axs[0].legend(handles=legend_handles, labels=legend_labels, loc='upper right')

        axs[int(np.round(num_months/2))].set_xlabel('Diameter (nm)')

        
        plt.show()

        # # normalized psd plot
        # fig, axs = plt.subplots(nrows=1, ncols=num_months, sharex=True, sharey=True)
        # i=0
        # for month in self.months:
        #     try:
        #         axs[i].loglog(self.diameter_midpoints, normalized_psd[str(month)])
        #         axs[i].set_title('2022'+'-'+str(month))

        #         i+=1
        #     except:
        #         pass
        # axs[0].set_ylabel('normalized dn/dlogdp')
        # axs[int(np.round(num_months/2))].set_xlabel('Diameter (nm)')

        # plt.show()

    def plot_psd_timeseries(self, data):
        """
        Plots a colormap of the psd over time.
        
        Only works for the 16 bin structure

        Inputs:
        - data: df of network mean data
        
        Returns: none

        Output: colormap plot
        """

        # Replace values equal to 0 with NaN
        data.replace(0, np.nan, inplace=True) 

        # compute dn/dlogdp lavlues and reformat for plotting
        dndlogdp_matrix = []
        # rehape to [[bin0time0, bin0time1, ...',
        #               bin1time0, bin1time1, ...], ...]
        for i, bin in enumerate(self.standard_bins):
            # compute dn/dlogdp for each bin
            row = np.array((data[bin]/self.dlogdp[i]).tolist())
            # remove values that are too small    
            dndlogdp_matrix.append(row)
        
        dndlogdp_matrix = np.array(dndlogdp_matrix)
        
        # set up meshgrid
        diameters = self.diameter_midpoints
        times = data['DateTime'].tolist()
        #times, diameters = np.meshgrid(data['DateTime'].tolist(), diameters)

        # plot contour plot with log-y axis
        contour_levels = np.logspace(np.log10(np.nanmin(dndlogdp_matrix)), np.log10(np.nanmax(dndlogdp_matrix)), 500)
        plt.contourf(times, diameters, dndlogdp_matrix, norm=colors.LogNorm(), levels=contour_levels, cmap='inferno')
        plt.yscale('log')

        # make sure colorbar shows
        cbar = plt.colorbar()

        # make cbar labels
        cbar_ticks = np.logspace(np.ceil(np.log10(np.nanmin(dndlogdp_matrix))), np.floor(np.log10(np.nanmax(dndlogdp_matrix))), 6)
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels(['$10^{{{}}}$'.format(int(np.log10(tick))) for tick in cbar_ticks])

        # make label for colorbar
        cbar.ax.set_ylabel('dN/dlogD$_p$')

        # y-axis label
        plt.ylabel('D$_p$ (nm)')

        plt.gca().xaxis.set_major_locator(ticker.AutoLocator())


        plt.show()
 
    def plot_monthly_bin_average(self, data, bin_names):
        """
        Plots the average value of the specified bins for each month.

        Inputs:
        - data: df of network mean
        - bin_names: list of bin names

        Output: plot

        Returns: none
        """

        # make groups into months 
        data['DateTime'] = pd.to_datetime(data['DateTime'])

        data['Time'] = data['DateTime'].dt.time
        data['Month'] = data['DateTime'].dt.month
        data['Year'] = data['DateTime'].dt.year
        data['Day'] = data['DateTime'].dt.day

        # number of months in data
        num_months = data['Month'].nunique()

        
        daily_averages = {}
        for bin in bin_names:
            daily_averages[bin] = data.groupby(['Year', 'Month', 'Time'])[bin].mean()

        # compute the average value of each month for each bin and save to dict
        bin_averages = {}
        for bin in bin_names:
            bin_averages[bin] = {}
            for year in self.years:
                for month in self.months:
                    try:
                        avg = np.mean(daily_averages[bin][year][month])
                        bin_averages[bin][f'{year}-{month}'] = avg
                    except:
                        pass
        
        # plot data
        for bin in bin_names:
            plt.semilogy(list(bin_averages[bin].values()), label=bin)
        plt.xlabel('Date')
        plt.ylabel('cm$^{-3}$')
        # make custom ticks
        custom_ticks = [i for i in range(num_months)] 
        custom_labels = list(bin_averages[bin].keys()) # use last bin because all same
        plt.xticks(custom_ticks, custom_labels)
        plt.legend()
        plt.show()
                
  



