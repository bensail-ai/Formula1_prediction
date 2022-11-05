# Formula 1 Qualifying Prediction

Welcome to the project on predicting Formula 1 qualifying performance prediction as part of Brain Station's Data Science Diploma program. 

### Brain Station Capstone Project
Author : Ben Palmer - [LinkedIn](https://www.linkedin.com/in/benpalmer470/)
Date: November 2022

## Project Overview:
The aim of the project is to investigate whether machine learning can accurately predict Formula1 qualifying position based on characteristics of the driver, car and circuit.

Formula1 is the pinnacle of motorsport where drivers and teams compete against each other over multiple races around the world, to be crowned world champion. For each race the drivers start the race from a standing start in the order based on their results in Qualifying the previous day. The teams and drivers look to maximise their position in Qualifying to give themselves the best chance of performing well in the race. Therefore, if teams have the knowledge of what features of the car and driver are likely to maximise performance for a given track that could be advantageous for them. In addition if Teams can predict there performance based on their car and driver combination it can help them focus on the areas they need to improve on. 

## Important Project Setup:

If you wish to run this project yourself, you can setup your the conda environment used to run this project by running this bash script:
<insert bash script>

Alternativily create your own conda environment using the requirements.txt file. 

### Data Sources:

The data comes from 3 main sources: 
- [Ergast API](http://ergast.com/mrd/)database historical result data from 1950 to present day
- Weather data from Visual Crossings API -[Visual Crossings API](https://www.visualcrossing.com/)
- [F1 Live](https://www.formula1.com/en/f1-live.html) python package which acts as a link to the [F1 Live](https://www.formula1.com/en/f1-live.html) telemetry data from 2018 to present day 

### Data location:

All the data needed to run this project is contained inside [./data/](./data/):
- [./data/raw](./data/raw/) : contains the snapshot of Ergast Relational database downloaded on the 24th September 2022 as .csv files
- [./data/raw/weather_data/](./data/raw/weather_data/) : contains the weather data from the visual crossing api
- [./data/clean/](./data/clean/) : contains the cleaned and aggregated Ergast Database in a wide format. Plus the raw telemetry data downloaded from FastF1 packaged for 10 GrandPrix's over 2018 and 2019
- [./data/model_data](./data/model_data/) : contains the complete dataset for modeling, the aggregated telemetry features from all the GrandPrix's 2018 to present. Plus summary dataframes of the modelling results as .pkl files

### Notebooks:
**The notebooks are labelled in the order they should be run 1 to 9.**
- [1_Ben_Palmer_Formula1_Ergast_database_combination](./1_Ben_Palmer_Formula1_Ergast_database_combination.ipynb) : Downloads, cleans and merges the relational Ergast database
- [2_Ben_Palmer_Formula1_EDA_on_ergast_legacy_database](./2_Ben_Palmer_Formula1_EDA_on_ergast_legacy_database.ipynb) : Exploratory Data Analysis on the legacy results Ergast database 
- [3_Ben_Palmer_Formula1_fastf1_telemetrydata_download_cleaning](./3_Ben_Palmer_Formula1_fastf1_telemetrydata_download_cleaning.ipynb) : Download F1 live telemetry data for 10 GrandPrix' over 2018 and 2019, plus clean the dataframe
- [4_Ben_Palmer_Formula1_telemetrydata_eda_and_aggregation](./4_Ben_Palmer_Formula1_telemetrydata_eda_and_aggregation.ipynb) : Run Feature aggregation on the sample telemetry dataset
- [5_Ben_Palmer_Forumla1_Feature_Importance](./5_Ben_Palmer_Forumla1_Feature_Importance.ipynb) : Investigate feature importance on the sample telemetry dataset. Are the features genereated good predictors? Can the dimension space be reduced?
- [6_Ben_Palmer_Formula1_Initial_Modelling](./6_Ben_Palmer_Formula1_Initial_Modelling.ipynb) : Download and run feature aggregation pipeline on complete telemetry dataset 2018 to present. EDA and feature importance investigation on the complete dataset.
- [7_Ben_Palmer_Formula1_classification_modelling_Random_Forest](./7_Ben_Palmer_Formula1_classification_modelling_Random_Forest.ipynb) : Building a Random Forest Classifier 
- [8_Ben_Palmer_Formula1_Regression_models](./8_Ben_Palmer_Formula1_Regression_models.ipynb) : Developing the regression models to predict the target variables
- [9_Ben_Palmer_Formula1_predicting_last_races_2022_season](./9_Ben_Palmer_Formula1_predicting_last_races_2022_season.ipynb) : Project summary and testing the result on the last 4 races of 2022 which is a new unseen test dataset.

### Scripts:
In here are the script files that complete:
- [./scripts/data_cleaning.py](./scripts/data_cleaning.py) : data cleaning of ergast database
- [./scripts/ds_ultils.py](./scripts/ds_ultils.py) : data science helper functions
- [./scripts/f1_ultils.py](./scripts/f1_ultils.py) : f1 telemetry helper functions to calculate feature aggregations
- [./scripts/fastf1_data_download.py](./scripts/fastf1_data_download.py) : f1 telemetry functions to download and run feature aggregations from the fastf1 package
- [./scripts/model_ultils.py](./scripts/model_ultils.py) : helper functions to prepare data for modelling
- [./scripts/Weather_data_collection.py](./scripts/Weather_data_collection.py) : script to get weather data from api
- [./scripts/weather_funcs.py](./scripts/weather_funcs.py) : functions that combin with weather data collection script to get weather data from api and web scraping
- [./scripts/plotstyle.mplstyle](./scripts/plotstyle.mplstyle) : color style for plots
### Model files:
This folder contains all the save .pkl model files from modelling for future use

### Images:
Collection of images saved from the visualisations created in this project

## Further help:

If you have any questions about this project or any issues running the project please reach out on benpalmer470@gmail.com
