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


### Scripts:


### Model files:


### Images:


If you have any questions about this project or any issues running the project please reach out on benpalmer470@gmail.com
