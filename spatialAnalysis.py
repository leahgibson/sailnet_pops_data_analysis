"""
Used for all spatial analysis.
"""

# Python packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


class spatialVariability:
    """
    Class for the analysis of variability in the data, such as
    - coefficient of variation
    - sudo variograms (coined this term myself)
    """

    def __init__(self):
        pass

    def coefficient_of_variation(self, dict_of_data, bin_names):
        """
        Treats the site data at time t as a data set and computes the coefficient of variation.
        Plots the timeseries of these computations to see how CV changes over time.

        Inputs:
        - dict_of_data: dictionary of site data
        - bin_names: list of bin names to use when computing CV

        Outputs: single plot with different lines for different bins

        Returns: df of the CV
        """

         # compute cov for each bin and plot on subplot
        
        cv_df = pd.DataFrame()

        for bin in bin_names:
            analysis_df = pd.DataFrame()
            sites = []
            for site, df in dict_of_data.items():
                sites.append(site)
                analysis_df[site] = df[bin]        
            analysis_df['DateTime'] = df['DateTime']
            analysis_df = analysis_df.set_index('DateTime')

            # compute cov
            cov = analysis_df.std(axis=1)/analysis_df.mean(axis=1)
            

            # add to column in df
            cv_df[bin] = cov.to_frame(name=bin)

            # plot
            plt.plot(cov, label=bin)
          
        plt.gca().xaxis.set_major_locator(ticker.AutoLocator())
        plt.legend()
        plt.show()

        # make datetime index an actual column
        cv_df = cv_df.reset_index()
    
        return cv_df

    def sudo_variogram(self, dict_of_data, bin_name, distance_type):
        """
        Plots the average percent difference between pairs of sites as a function
        of either their vertical difference, distance between them, or 
        normalized euclidean distance.

        Plots the result and computes the Pearson r correlation.
        
        Inputs:
        - dict_of_data: dictionary of site data
        - bin_name: name of bin header to use in comparison
        - distance_type: str to represent the distance used, pick from:
            - 'horizontal'
            - 'vertical'
            - 'euclidean' 
        
        Output: figure

        Returns: none
        """

        ### to do: copy code form otherplace and copy supporting functions


class networkDesign:
    """
    Class for the analysis of the network as a whole. Features include:
    - representation error and analysis
    """

    def __init__(self):
        pass
