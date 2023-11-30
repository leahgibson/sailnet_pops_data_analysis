"""
Analysis of the network mean timeseries.
"""

# import Python packages
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Set the font size for different plot elements
plt.rcParams.update({
    'font.size': 24,               # Font size for general text
    'axes.titlesize': 24,          # Font size for plot titles
    'axes.labelsize': 18,          # Font size for axis labels
    'xtick.labelsize': 18,         # Font size for x-axis ticks
    'ytick.labelsize': 18,         # Font size for y-axis ticks
    'legend.fontsize': 18,         # Font size for legend
})

class basicVisualization:
    """
    Class for basic visualization of data.
    """

    def __init__(self):
        pass

    def plot_network_timeseries(self, df, bin_name, rolling=None):
        """
        Basic plotting of network mean for specified bin.

        Inputs:
        - df: df of network mean
        - bin_name: name of bin to be plotted
        - rolling: default None or int for number of values to use in a rolling mean

        Output: plot

        Returns: None
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
        for group in year_groups:
            df = group[1]
            # replace all years with 2023
            df['DateTime'] = df['DateTime'].apply(lambda x: x.replace(year=2023))
            # now remove year
            #df['DateTime'] = pd.to_datetime(df['DateTime'].dt.strftime('%m-%d %H:%M:%S'))
            

            ax.plot(group[1]['DateTime'], group[1][bin_name], label=str(group[0]))
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
        plt.show()




class temporalAnalysis:
    """
    Class for more in depth analysis of temporal trends in the network mean.
    """

    def __init__(self):
        self.years = [2021, 2022, 2023]
        self.months = [1,2,3,4,5,6,7,8,9,10,11,12]
        self.standard_bins = ['b' + str(i) for i in range(16)]
        

    def plot_monthly_diurnal(self, data, bin_names):
        """
        Plots the average diurnal cycle for each month of 2022 by 
        averaging over each dat in the month.

        Note: there must be more than one month present for plot to work.
        
        Inputs:
        - data: df of network mean
        - bin_name: list of name of bins to analyze
            if data is cov data, use bin_name='' #### FILL THIS IN ###
        
        Outputs: plot
        
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
        for bin in bin_names:
            i=0
            for year in self.years:
                for month in self.months:
                    try:
                        axs[i].plot(daily_averages[bin][year][month].values, label=bin)
                        axs[i].set_title(str(year)+'-'+str(month))
                        axs[i].set_xticks([0, 23, 47])
                        axs[i].set_xticklabels(['00:00', '12:00', '24:00'])

                        if i==0:
                            axs[i].legend()
                        
                        i+=1
                    except:
                        pass

        
                
        plt.show()

    
    def plot_monthly_psd(self, data):
        """
        Plots the average monthly psd and the average normalized psd for each month.

        Note: only works for 16 bins and for 2022 year.
        
        Inputs:
        - data: df of network mean

        Outputs: twp diffrent plots

        Reutns: noone
        """

        dlogdp = [0.03645458169, 0.03940255269, 0.04033092159, 0.03849895488,
                    0.03655010672, 0.04559350564, 0.08261548653, 0.06631586816,
                    0.15575785, 0.1008071129, 0.1428650493, 0.1524763279,
                    0.07769393472, 0.1571866015, 0.1130751916, 0.0867054262]
        
        diameter_midpoints = [149, 163, 178, 195, 213, 234, 272, 322, 422, 561, 748,
                            1054, 1358, 1802, 2440, 3062]
        
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
            for month in self.months:
                if str(month) in psd_dict:
                    pass
                else:
                    psd_dict[str(month)] = []
                    normalized_psd[str(month)] = []
                try:
                    # compute monthly average of bin
                    monthly_avg = np.mean(daily_avgs[2022][month])
                    # convert to dndlogdp
                    dndlogdp_val = monthly_avg/dlogdp[i]
                    psd_dict[str(month)].append(dndlogdp_val)

                    # compute normalized average
                    monthly_total = np.mean(total_averages[2022][month])
                    normalized_bin = monthly_avg/monthly_total
                    normalized_dndlogdp = normalized_bin/dlogdp[i]
                    normalized_psd[str(month)].append(normalized_dndlogdp)
        
                except:
                    pass
        
         # plot the particle dize distributions
        fig, axs = plt.subplots(nrows=1, ncols=num_months, sharex=True, sharey=True)
        i=0
        for month in self.months:
            try:
                axs[i].loglog(diameter_midpoints, psd_dict[str(month)])
                axs[i].set_title('2022'+'-'+str(month))

                i+=1
            except:
                pass
        axs[0].set_ylabel('dn/dlogdp')
        axs[int(np.round(num_months/2))].set_xlabel('Diameter (nm)')

        
        plt.show()

        # normalized psd plot
        fig, axs = plt.subplots(nrows=1, ncols=num_months, sharex=True, sharey=True)
        i=0
        for month in self.months:
            try:
                axs[i].loglog(diameter_midpoints, normalized_psd[str(month)])
                axs[i].set_title('2022'+'-'+str(month))

                i+=1
            except:
                pass
        axs[0].set_ylabel('normalized dn/dlogdp')
        axs[int(np.round(num_months/2))].set_xlabel('Diameter (nm)')

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
                





