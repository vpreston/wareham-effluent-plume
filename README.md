# wareham-effluent-plume

This repository contains cleaned data and analysis from chemical sampling that occured in the Wareham River, MA July 2017. A (JetYak vehicle)[https://www.whoi.edu/oceanus/feature/the-jetyak] was deployed with sensing equipment including a CTD, a Greenhouse Gas Analyzer (GGA), oxygen sensor (optode), and optical nitrate sensor. Salinity, temperature, methane, carbon dioxide, oxygen, and nitrate were target signals of interest for the deployment. The vehicle was further equipped with a (pixhawk)[https://pixhawk.org/] controller for navigation by radio control.

For completeness, the regimes used to clean the raw data is provided, however, the raw data files are not provided in this respository. To review a copy of the raw data files, please contact the repository owner.

## Using this Repository
The following is necessary to be available on your machine to use this repository:
	* ipython notebook; anaconda distribution of jupyter is recommended, but not necessary
	* numpy
	* scipy
	* pandas
	* descartes
	* shapely
	* matplotlib

## Understanding the Data
The data is largely stored in CSV files, and processed using pandas dataframes in ipython notebooks. The dataframes are organized by instrument and property, and ordered using chronological time. Simple interpolation is used to align measurements from the multiple sensors to a single timeline. Inline comments in the notebooks will clarify any specific aspects of the data presented which may not be immediately clear.

## Notebooks
### Raw Data Parsing
In this ipython notebook, the raw data is initially explored. 

### Data Cleaning
This notebook presents the data cleaning regime that is used on the data which is further analyzed by the Visualization and Animation notebooks. 

### Data Visualization
The cleaned data is visualized in a number of ways: spatially, temporally, property vs. property. Please see in-line comments for brief qualitative remarks on observations in the data. 

### Animations
In this notebook, a more intensive extrapolation regime is used to create a scalar field of information across the entirety of the river geometry. This is then subsequently animated in time based upon 5 discrete transects conducted of the river.