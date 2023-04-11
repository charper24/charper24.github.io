import pandas as pd
import plotly.express as px
import numpy as np
import math
import requests
from urllib.request import urlopen
import json
import os
import matplotlib.pyplot as plt
import csv


censusVar = 'DP03_0120PE'

class DataDownloader:
    """
    The DataDownloader class is responsible for downloading FIPS codes and county center coordinates data.
    """
    def __init__(self):
        self.fipsDf = None
        self.countyCenters = None
        
    def download_fips_data(self):
        """
        Downloads FIPS codes data from a remote source and stores it as a pandas DataFrame.
        """
        url = "https://raw.githubusercontent.com/ChuckConnell/articles/master/fips2county.tsv"
        self.fipsDf = pd.read_csv(url, sep='\t', header='infer', dtype=str, encoding='latin-1')
        
    def download_county_centers(self):
        """
        Downloads county center coordinates from a remote source and stores it as a pandas DataFrame.
        """
        url = "https://www.cse.unl.edu/~charper/155t/embeddedEthics/countyGeoCenter_latLong.txt"
        self.countyCenters = pd.read_csv(url, sep='\t', dtype=str)
        self.countyCenters = self.countyCenters[['FIPS', 'Latitude', 'Longitude']]
        self.countyCenters = self.countyCenters.rename(columns={'FIPS': 'fips_id'})

class CountyData:
    """
    The CountyData class is responsible for handling county data operations.
    """
    def __init__(self, data_downloader):
        """
        Initializes a CountyData object with a DataDownloader object.
        
        Args:
            data_downloader (DataDownloader): A DataDownloader object to access FIPS codes and county center coordinates.
        """
        self.data_downloader = data_downloader
        
    def countyNameStateNameToFips(self, countyName, stateName):
        """
        Returns the FIPS code for a given county and state name.
        
        Args:
            countyName (str): The name of the county.
            stateName (str): The name of the state.
        
        Returns:
            str: The FIPS code of the county if found, otherwise "-1".
        """
        state_fips = self.data_downloader.fipsDf[self.data_downloader.fipsDf["StateName"] == stateName]
        county_fips = state_fips[state_fips["CountyName"] == countyName]
        fip_id = county_fips["CountyFIPS"]
        if fip_id.size > 0:
            return county_fips["CountyFIPS"].values[0]
        else:
            print("County Not Found")
            return "-1"

    def addCenterLatLonToCensusData(self, censusData):
        """
        Adds center latitude and longitude data to the given census data.
        
        Args:
            censusData (pd.DataFrame): A pandas DataFrame containing census data.
        
        Returns:
            pd.DataFrame: The updated census data with center latitude and longitude columns.
        """
        neFipIds = [f"31{str((i * 2 - 1)).zfill(3)}" for i in range(1, 94)]
        countyCenters = self.data_downloader.countyCenters[self.data_downloader.countyCenters['fips_id'].isin(neFipIds)]
        censusData = pd.merge(censusData, countyCenters, on='fips_id', how='outer')
        return censusData

class CountyPlotter:
    """
    The CountyPlotter class is responsible for plotting county data on a map.
    """
    def __init__(self):
        self.counties = None
        
    def download_geojson(self):
        """
        Downloads GeoJSON data for US counties and stores it as a dictionary.
        """
        url = 'https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json'
        with urlopen(url) as response:
            self.counties = json.load(response)
            
    def plotCountyData_ne(self, nebraskaCountyData):
        """
        Plots Nebraska county data on a map using Plotly.
        
        Args:
            nebraskaCountyData (pd.DataFrame): A pandas DataFrame containing Nebraska county data.
        """
        fig = px.choropleth(nebraskaCountyData, geojson=self.counties, locations='fips_id', color=censusVar,
                            color_continuous_scale="Viridis",
                            range_color=(0, 30),
                            scope="usa",
                            labels={censusVar: 'Percent in Poverty'})
        fig.update_layout(title_text="Title", margin={"r": 0, "t": 0, "l": 0, "b": 0}, dragmode=False,
                          geo=dict(
                              projection_scale=6.5,
                              center=dict(lat=41.4, lon=-99.9)
                          ))
        fig.show()

class CensusData:
    """
    The CensusData class is responsible for handling census data operations.
    """
    def __init__(self, county_data):
        """
        Initializes a CensusData object with a CountyData object.
        
        Args:
            county_data (CountyData): A CountyData object to access county information.
        """
        self.county_data = county_data
        
    def getCensusPovertyDataByYear_ne(self, year):
        """
        Retrieves census poverty data for Nebraska counties for a specific year.
        
        Args:
            year (int): The year for which census poverty data is required.
        
        Returns:
            pd.DataFrame: A pandas DataFrame containing the retrieved census poverty data.
        """
        file_path = os.path.join("data", f"census_data_{censusVar}_{year}.csv")
        
        with open(file_path, 'r') as csvfile:
            data = [row for row in csv.reader(csvfile)]

        censusData = pd.DataFrame(data[1:], columns=data[0])

        fips_ids = [row["state"] + row["county"] for _, row in censusData.iterrows()]
        censusData['fips_id'] = fips_ids
        censusData[censusVar] = censusData[censusVar].astype(str).astype(float)
        censusData = censusData.sort_values(by=[censusVar])
        censusData = self.county_data.addCenterLatLonToCensusData(censusData)
        censusData['Latitude'] = pd.to_numeric(censusData['Latitude'])
        censusData['Longitude'] = pd.to_numeric(censusData['Longitude'])
        censusData[censusVar] = pd.to_numeric(censusData[censusVar])
        censusData[["countyName", "stateName"]] = censusData["NAME"].str.split(', ', expand=True)
        censusData.drop('NAME', inplace=True, axis=1)
        return censusData

class SamplingMethods:
    """
    The SamplingMethods class is responsible for various sampling methods on a given DataFrame.
    """
    def __init__(self):
        pass
        
    def getHalfCounties_random(self, dataFrame):
        """
        Randomly selects half of the counties from a given DataFrame.
        
        Args:
            dataFrame (pd.DataFrame): A pandas DataFrame containing county data.
        
        Returns:
            pd.DataFrame: A pandas DataFrame containing the randomly selected half of counties.
        """
        numCounties = dataFrame.shape[0]
        randomIndicesToSelect = np.random.choice(np.arange(0, numCounties), int(numCounties / 2), replace=False)
        randomlySelectedCountiesData = dataFrame.iloc[randomIndicesToSelect]
        return randomlySelectedCountiesData

    def getHalf_highestPovCounties(self, dataFrameSource, dataFrameTarget):
        """
        Selects half of the highest poverty counties from the source DataFrame and filters the target DataFrame based on these counties.
        
        Args:
            dataFrameSource (pd.DataFrame): A pandas DataFrame containing source county data.
            dataFrameTarget (pd.DataFrame): A pandas DataFrame containing target county data.
        
        Returns:
            pd.DataFrame: A pandas DataFrame containing the filtered target county data based on half of the highest poverty counties.
        """
        numCountiesToSelect = int(dataFrameSource.shape[0] / 2)
        dataFrameSource = dataFrameSource.sort_values(by=censusVar)
        highestPovertyCounties_fips_ids = dataFrameSource.iloc[numCountiesToSelect:]['fips_id'].to_list()
        dataFrameTarget = dataFrameTarget[dataFrameTarget['fips_id'].isin(highestPovertyCounties_fips_ids)]
        return dataFrameTarget

    def get25PercLowestPov_25PercHighestPov(self, dataFrameSource, dataFrameTarget):
        """
        Selects 25% of the lowest and highest poverty counties from the source DataFrame and filters the target DataFrame based on these counties.
        
        Args:
            dataFrameSource (pd.DataFrame): A pandas DataFrame containing source county data.
            dataFrameTarget (pd.DataFrame): A pandas DataFrame containing target county data.
        
        Returns:
            pd.DataFrame: A pandas DataFrame containing the filtered target county data based on 25% of the lowest and highest poverty counties.
        """
        dataFrameSource = dataFrameSource.sort_values(by=censusVar)
        numValues_25_percent = int(dataFrameSource.shape[0] / 4)
        lowPoverty = dataFrameSource.iloc[0:numValues_25_percent]
        highPoverty = dataFrameSource.iloc[dataFrameSource.shape[0] - numValues_25_percent:]
        combined_lowHigh = pd.concat([lowPoverty, highPoverty], ignore_index=True)
        combined_lowHigh_fips_ids = combined_lowHigh['fips_id'].to_list()
        dataFrameTarget = dataFrameTarget[dataFrameTarget['fips_id'].isin(combined_lowHigh_fips_ids)]
        return dataFrameTarget

class IDWInterpolation:
    """
    The IDWInterpolation class is responsible for performing Inverse Distance Weighting (IDW) interpolation on given data.
    """
    def __init__(self):
        pass

    @staticmethod
    def standard_idw(lon, lat, longs, lats, d_values, id_power, s_radious):
        """
        Performs standard IDW interpolation for a given location and a set of known data points.
        
        Args:
            lon (float): Longitude of the location to interpolate.
            lat (float): Latitude of the location to interpolate.
            longs (list): List of longitudes of known data points.
            lats (list): List of latitudes of known data points.
            d_values (list): List of data values corresponding to the known data points.
            id_power (float): The power parameter for IDW.
            s_radious (int): The number of nearest neighbors to use for interpolation.
        
        Returns:
            float: The interpolated value at the given location.
        """
        calc_arr = np.zeros(shape=(len(longs), 6))
        calc_arr[:, 0] = longs
        calc_arr[:, 1] = lats
        calc_arr[:, 3] = d_values
        calc_arr[:, 4] = 1 / (np.sqrt((calc_arr[:, 0] - lon) ** 2 + (calc_arr[:, 1] - lat) ** 2) ** id_power + 1)
        calc_arr = calc_arr[np.argsort(calc_arr[:, 4])][-s_radious:, :]
        calc_arr[:, 5] = calc_arr[:, 3] * calc_arr[:, 4]
        idw = calc_arr[:, 5].sum() / calc_arr[:, 4].sum()
        return idw

    def interpolate(self, censusData, sample, power, numNeighbors):
        """
        Interpolates missing data for a set of counties using IDW interpolation.
        
        Args:
            censusData (pd.DataFrame): A pandas DataFrame containing all county data.
            sample (pd.DataFrame): A pandas DataFrame containing a subset of county data with known values.
            power (float): The power parameter for IDW interpolation.
            numNeighbors (int): The number of nearest neighbors to use for interpolation.
        
        Returns:
            pd.DataFrame: A pandas DataFrame containing the interpolated values for the missing counties.
        """
        countiesToInterp = self.getMissingCounties(censusData["fips_id"], sample['fips_id'].to_list())

        countiesToInterp_dict = {id: self.getLatLongByFipsId(censusData, id) for id in countiesToInterp}

        knownLatitudes = sample["Latitude"].values
        knownLongitudes = sample["Longitude"].values
        percentInPovData = sample[censusVar].values

        estimatedPovData_dict = {id: self.standard_idw(sampleLong, sampleLat, knownLongitudes, knownLatitudes, percentInPovData, power, numNeighbors)
                                 for id, (sampleLat, sampleLong) in countiesToInterp_dict.items()}

        estimates_df = pd.DataFrame(estimatedPovData_dict.items(), columns=['fips_id', censusVar])
        return estimates_df

    @staticmethod
    def getLatLongByFipsId(censusData, fips_id):
        """
        Retrieves the latitude and longitude for a county given its FIPS code.
        
        Args:
            censusData (pd.DataFrame): A pandas DataFrame containing county data with FIPS codes.
            fips_id (str): The FIPS code of the county.
        
        Returns:
            tuple: A tuple containing the latitude and longitude of the county if found, otherwise (nan, nan).
        """
        latitudeDf = censusData[censusData['fips_id'] == fips_id]["Latitude"]
        longitudeDf = censusData[censusData['fips_id'] == fips_id]["Longitude"]
        if latitudeDf.shape[0] > 0:
            return latitudeDf.iloc[0], longitudeDf.iloc[0]
        else:
            return float('nan'), float('nan')

    @staticmethod
    def getMissingCounties(allCounties, countiesWithData):
        """
        Returns a list of FIPS codes for counties that are missing data.
        
        Args:
            allCounties (list): A list of FIPS codes for all counties.
            countiesWithData (list): A list of FIPS codes for counties with data.
        
        Returns:
            list: A list of FIPS codes for counties that are missing data.
        """
        return [id for id in allCounties if id not in countiesWithData]

    @staticmethod
    def combineDataWithInterp(df1, df2):
        """
        Combines two DataFrames containing county data and interpolated values.
        
        Args:
            df1 (pd.DataFrame): A pandas DataFrame containing county data with known values.
            df2 (pd.DataFrame): A pandas DataFrame containing county data with interpolated values.
        
        Returns:
            pd.DataFrame: A pandas DataFrame containing the combined county data.
        """
        sample_data_withInterp = df1[['fips_id', censusVar]].copy(deep=True)
        sample_data_withInterp = pd.concat([sample_data_withInterp, df2], ignore_index=True)
        return sample_data_withInterp

    @staticmethod
    def getRealvaluesGivenInterpolated(censusData, sample_interp):
        """
        Retrieves the actual data values for a set of interpolated counties.
        
        Args:
            censusData (pd.DataFrame): A pandas DataFrame containing all county data.
            sample_interp (pd.DataFrame): A pandas DataFrame containing a set of interpolated county data.
        
        Returns:
            pd.DataFrame: A pandas DataFrame containing the actual data values for the interpolated counties.
        """
        sample_actual = sample_interp.copy(deep=True)
        sample_actual = sample_actual.drop(censusVar, axis=1)
        sample_actual = censusData[censusData['fips_id'].isin(sample_actual['fips_id'].to_list())][['fips_id', censusVar]]

        sample_interp = sample_interp.rename(columns={censusVar: censusVar + "_est"})
        sample_interp_withActual = pd.merge(sample_interp, sample_actual, on="fips_id")

        return sample_interp_withActual

    def plotSampleWithInterp(self, sample, sample_interp, county_plotter):
        """
        Plots a map of counties using a given sample of county data and interpolated values.
        
        Args:
            sample (pd.DataFrame): A pandas DataFrame containing a subset of county data with known values.
            sample_interp (pd.DataFrame): A pandas DataFrame containing county data with interpolated values.
            county_plotter (CountyPlotter): A CountyPlotter object to plot the map.
        """
        sample_data_withInterp = self.combineDataWithInterp(sample, sample_interp)
        county_plotter.plotCountyData_ne(sample_data_withInterp)


class ErrorCalculator:
    """
    The ErrorCalculator class is responsible for calculating various error metrics on interpolated data.
    """
    def __init__(self):
        pass

    @staticmethod
    def getAvgError(dataFrame):
        """
        Calculates the average error between actual and estimated values in the given DataFrame.
        
        Args:
            dataFrame (pd.DataFrame): A pandas DataFrame containing actual and estimated values.
        
        Returns:
            float: The average error.
        """
        errors = [(row[censusVar + "_est"] - row[censusVar]) for _, row in dataFrame.iterrows()]
        errors = np.array(errors)
        return errors.mean()

    @staticmethod
    def getAvgAbsError(dataFrame):
        """
        Calculates the average absolute error between actual and estimated values in the given DataFrame.
        
        Args:
            dataFrame (pd.DataFrame): A pandas DataFrame containing actual and estimated values.
        
        Returns:
            float: The average absolute error.
        """
        errors = [abs(row[censusVar + "_est"] - row[censusVar]) for _, row in dataFrame.iterrows()]
        errors = np.array(errors)
        return errors.mean()

    @staticmethod
    def getMeanSquaredError(dataFrame):
        """
        Calculates the mean squared error between actual and estimated values in the given DataFrame.
        
        Args:
            dataFrame (pd.DataFrame): A pandas DataFrame containing actual and estimated values.
        
        Returns:
            float: The mean squared error.
        """
        errors = [((row[censusVar + "_est"] - row[censusVar]) ** 2) for _, row in dataFrame.iterrows()]
        errors = np.array(errors)
        return errors.mean()

    @staticmethod
    def getRootMeanSquaredError(dataFrame):
        """
        Calculates the root mean squared error between actual and estimated values in the givenDataFrame.

        Args:
            dataFrame (pd.DataFrame): A pandas DataFrame containing actual and estimated values.
        
        Returns:
            float: The root mean squared error.
        """
        return math.sqrt(ErrorCalculator.getMeanSquaredError(dataFrame))

    @staticmethod
    def percentPredictedWithinErrorBound(dataFrame, percentBound):
        """
        Calculates the percentage of predictions that fall within a specified error bound.
        
        Args:
            dataFrame (pd.DataFrame): A pandas DataFrame containing actual and estimated values.
            percentBound (float): The error bound as a percentage.
        
        Returns:
            str: The percentage of predictions within the error bound, formatted as a string with 3 decimal places.
        """
        errors = [1 if abs(row[censusVar + "_est"] - row[censusVar]) < percentBound else 0 for _, row in dataFrame.iterrows()]
        errors = np.array(errors)
        return str(int(errors.mean() * 100000) / 1000) + "%"

    @staticmethod
    def getErrors(dataFrame, percentBound=3, printErrors=False):
        """
        Calculates various error metrics for a given DataFrame.
        
        Args:
            dataFrame (pd.DataFrame): A pandas DataFrame containing actual and estimated values.
            percentBound (float, optional): The error bound as a percentage. Defaults to 3.
            printErrors (bool, optional): Whether to print the calculated errors. Defaults to False.
        
        Returns:
            dict: A dictionary containing the calculated error metrics.
        """
        errors = dict()
        errors["Average Error"] = ErrorCalculator.getAvgError(dataFrame)
        errors["Average Absolute Error"] = ErrorCalculator.getAvgAbsError(dataFrame)
        errors["Mean Squared Error"] = ErrorCalculator.getMeanSquaredError(dataFrame)
        errors["Root Mean Squared Error"] = ErrorCalculator.getRootMeanSquaredError(dataFrame)
        errors["Percent Under Error Threshold"] = ErrorCalculator.percentPredictedWithinErrorBound(dataFrame, percentBound)
        if printErrors:
            print("Average Error: " + str(ErrorCalculator.getAvgError(dataFrame)))
            print("Average Absolute Error: " + str(ErrorCalculator.getAvgAbsError(dataFrame)))
            print("Mean Squared Error: " + str(ErrorCalculator.getMeanSquaredError(dataFrame)))
            print("Root Mean Squared Error: " + str(ErrorCalculator.getRootMeanSquaredError(dataFrame)))
            print("Percent Predicted With Smaller than a " + str(percentBound) + "% error: " + ErrorCalculator.percentPredictedWithinErrorBound(dataFrame, percentBound))
            print("\n")
        return errors

    @staticmethod
    def getErrorsByQuartile(dataFrame, percentBound=3, printErrors=False):
        """
        Calculates error metrics for each quartile of the given DataFrame.
        
        Args:
            dataFrame (pd.DataFrame): A pandas DataFrame containing actual and estimated values.
            percentBound (float, optional): The error bound as a percentage. Defaults to 3.
            printErrors (bool, optional): Whether to print the calculated errors. Defaults to False.
        
        Returns:
            dict: A dictionary containing the calculated error metrics for each quartile.
        """
        dataFrame.sort_values(censusVar)
        q = dataFrame.quantile([0.00, 0.25, 0.50, 0.75, 1.00], numeric_only=True)
        col = censusVar
        quartileErrors = dict()

        q1 = dataFrame[((dataFrame[col] >= q[col][0.00]) & (dataFrame[col] < q[col][0.25]))]
        q2 = dataFrame[((dataFrame[col] >= q[col][0.25]) & (dataFrame[col] < q[col][0.50]))]
        q3 = dataFrame[((dataFrame[col] >= q[col][0.50]) & (dataFrame[col] < q[col][0.75]))]
        q4 = dataFrame[((dataFrame[col] >= q[col][0.75]) & (dataFrame[col] <= q[col][1.00]))]

        q1_errors = ErrorCalculator.getErrors(q1, percentBound)
        q2_errors = ErrorCalculator.getErrors(q2, percentBound)
        q3_errors = ErrorCalculator.getErrors(q3, percentBound)
        q4_errors = ErrorCalculator.getErrors(q4, percentBound)
        quartileErrors["Quartile 1 Errors"] = q1_errors
        quartileErrors["Quartile 2 Errors"] = q2_errors
        quartileErrors["Quartile 3 Errors"] = q3_errors
        quartileErrors["Quartile 4 Errors"] = q4_errors

        if printErrors:
            print("ERROR FOR QUARTILE #1 -- " + str(q1.iloc[0][censusVar]) + "% poverty to " + str(q1.iloc[q1.shape[0] - 1][censusVar]) + "% poverty:")
            print("\nERROR FOR QUARTILE #2 -- " + str(q2.iloc[0][censusVar]) + "% poverty to " + str(q2.iloc[q2.shape[0] - 1][censusVar]) + "% poverty:")
            print("\nERROR FOR QUARTILE #3 -- " + str(q3.iloc[0][censusVar]) + "% poverty to " + str(q3.iloc[q3.shape[0] - 1][censusVar]) + "% poverty:")
            print("\nERROR FOR QUARTILE #4 -- " + str(q4.iloc[0][censusVar]) + "% poverty to " + str(q4.iloc[q4.shape[0] - 1][censusVar]) + "% poverty:")

        # Calculate poverty_ranges
        poverty_ranges = [
            f"{q1[censusVar].min():.1f}% - {q1[censusVar].max():.1f}%",
            f"{q2[censusVar].min():.1f}% - {q2[censusVar].max():.1f}%",
            f"{q3[censusVar].min():.1f}% - {q3[censusVar].max():.1f}%",
            f"{q4[censusVar].min():.1f}% - {q4[censusVar].max():.1f}%"
        ]

        return {'quartile_errors': quartileErrors, 'poverty_ranges': poverty_ranges}



class ErrorVisualizer:

    @staticmethod
    def plot_error_barchart(quartile_errors, poverty_ranges):
        quartiles = list(quartile_errors.keys())
        avg_abs_errors = [errors['Average Absolute Error'] for errors in quartile_errors.values()]
        rmse_errors = [errors['Root Mean Squared Error'] for errors in quartile_errors.values()]
        percentages = [float(errors['Percent Under Error Threshold'][:-1]) for errors in quartile_errors.values()]

        bar_width = 0.3
        index = np.arange(len(quartiles))

        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.bar(index - bar_width, avg_abs_errors, width=bar_width, color='b', label='Average Absolute Error')
        ax1.bar(index, rmse_errors, width=bar_width, color='g', label='Root Mean Squared Error')
        ax1.set_ylabel('Error Value')
        ax1.set_title('Error Metrics by Quartile')
        ax1.set_xticks(index)
        ax1.set_xticklabels(poverty_ranges)  # Update x-axis labels with poverty percent ranges
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()
        percentage_bars = ax2.bar(index + bar_width, percentages, width=bar_width, color='r', label='Percent Under Error Threshold')
        ax2.set_ylabel('Percentage')
        ax2.legend(loc='upper right')

        # Add percentage labels on top of bars
        for bar in percentage_bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.1f}%", ha='center', va='bottom', fontsize=10)

        fig.tight_layout()
        plt.show()