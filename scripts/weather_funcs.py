import urllib.request
import urllib.error
import sys
import json
import ssl
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import requests

def weather_api_call(Location,StartDate):

    context = ssl._create_unverified_context()

    # This is the core of our weather query URL
    BaseURL = 'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/'

    ApiKey='CD9PVKSU42B7LXYDT3R7S3863'
    #UnitGroup sets the units of the output - us or metric
    UnitGroup='metric'

    ContentType="json"
    Include="hours"
    #basic query including location
    ApiQuery=BaseURL + Location

    #append the start and end date if present
    if (len(StartDate)):
        ApiQuery+="/"+StartDate
        

    #Url is completed. Now add query parameters (could be passed as GET or POST)
    ApiQuery+="?"

    #append each parameter as necessary
    if (len(UnitGroup)):
        ApiQuery+="&unitGroup="+UnitGroup

    if (len(ContentType)):
        ApiQuery+="&contentType="+ContentType

    if (len(Include)):
        ApiQuery+="&include="+Include

    ApiQuery+="&key="+ApiKey

    print(' - Running query URL: ', ApiQuery)
    print()

    try: 
        CSVBytes = urllib.request.urlopen(ApiQuery,context=context)
        data = CSVBytes.read()
        weatherData = json.loads(data.decode('utf-8'))
        frames=[]
        for day in weatherData['days']:
            print('day',day['datetime'])
            day_date = day['datetime']
            time= []
            temp =[]          
            precip=[]
            humidity=[]
            conditions=[]
        
            for hour in day['hours']:
                time.append(hour['datetime'])                  
                temp.append(hour['temp'])
                precip.append(hour['precip'])         
                humidity.append(hour['humidity'])
                conditions.append(hour['conditions'])
            df=pd.DataFrame({'time':time,'temp':temp,
            'precip':precip, 'humidity':humidity,'conditions':conditions
            })
            df['day']=day_date    
            
            frames.append(df)
        dfweather=pd.concat(frames).reset_index(drop=True)
        #print('data',dfweather.groupby(['day']).mean())
        return dfweather

    except urllib.error.HTTPError  as e:
        ErrorInfo= e.read().decode() 
        print('Error code: ', e.code, ErrorInfo)
        #sys.exit()
    except  urllib.error.URLError as e:
        #ErrorInfo= e.read().decode() 
        print('Error code: ', e.reason)
        #sys.exit()


def weather_api_call_day(Location,StartDate):

    context = ssl._create_unverified_context()

    # This is the core of our weather query URL
    BaseURL = 'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/'

    ApiKey='CD9PVKSU42B7LXYDT3R7S3863'
    #UnitGroup sets the units of the output - us or metric
    UnitGroup='metric'

    ContentType="json"
    Include="days"
    #basic query including location
    ApiQuery=BaseURL + Location

    #append the start and end date if present
    if (len(StartDate)):
        ApiQuery+="/"+StartDate
        

    #Url is completed. Now add query parameters (could be passed as GET or POST)
    ApiQuery+="?"

    #append each parameter as necessary
    if (len(UnitGroup)):
        ApiQuery+="&unitGroup="+UnitGroup

    if (len(ContentType)):
        ApiQuery+="&contentType="+ContentType

    if (len(Include)):
        ApiQuery+="&include="+Include

    ApiQuery+="&key="+ApiKey

    print(' - Running query URL: ', ApiQuery)
    print()

    try: 
        CSVBytes = urllib.request.urlopen(ApiQuery,context=context)
        data = CSVBytes.read()
        weatherData = json.loads(data.decode('utf-8'))
        frames=[]    
        temp =[]   
        precip=[]
        humidity=[]
        conditions=[]
    
        for day in weatherData['days']:
            print('day',day['datetime'])
                            
            
            temp.append(day['temp'])
            precip.append(day['precip'])
            
            humidity.append(day['humidity'])
            conditions.append(day['conditions'])
            df=pd.DataFrame({'temp':temp,
            'precip':precip, 
            'humidity':humidity,'conditions':conditions,
            })               
            
            frames.append(df)
        dfweather=pd.concat(frames).reset_index(drop=True)
        #print('data',dfweather.groupby(['day']).mean())
        return pd.Series([dfweather['temp'].values[0],dfweather['precip'].values[0], dfweather['humidity'].values[0],  dfweather['conditions'].values[0]])

    except urllib.error.HTTPError  as e:
        ErrorInfo= e.read().decode() 
        print('Error code: ', e.code, ErrorInfo)
        #sys.exit()
    except  urllib.error.URLError as e:
        #ErrorInfo= e.read().decode() 
        print('Error code: ', e.reason)
        #sys.exit()




def get_weather_data(Lat,Lng,StartDate,time): 


    Location=str(Lat)+','+str(Lng)    
    
    if str(time) != 'nan':
        df=weather_api_call(Location,StartDate)
        hour=int(time.split(":")[0])
        if hour < 11:
            starttime=f"0{hour-1}:00:00"
        else:
            starttime=f"{hour}:00:00"
        if hour+4 < 10:
            endtime= f"0{hour+4}:00:00"
        else:
            endtime= f"{hour+4}:00:00" 
        
        df = df[(df['time'] > starttime) & (df['time'] < endtime)]

        
        if len(df['conditions'].mode()) > 1:
            conditions = [x for x in df['conditions'].values]   
           
        else:
            conditions = df['conditions'].mode().values[0]     
        
        weather_data=pd.Series([df['temp'].mean(),df['precip'].mean(), df['humidity'].mean(), conditions])
    else:
        weather_data = weather_api_call_day(Location,StartDate)   

    return weather_data
    


def get_weather_data_quali(Lat,Lng,qualiDate,qualitime,raceDate,racetime):

    Location=str(Lat)+','+str(Lng)    

    if str(qualiDate) != 'nan':
        StartDate = qualiDate
        time=qualitime
    else:
        StartDate = str(pd.to_datetime(raceDate) -pd.DateOffset(1)).split()[0]
        time=racetime

    if str(time) != 'nan':
        df=weather_api_call(Location,StartDate)
        hour=int(time.split(":")[0])
        if hour < 11:
            starttime=f"0{hour-1}:00:00"
        else:
            starttime=f"{hour}:00:00"
        if hour+4 < 10:
            endtime= f"0{hour+4}:00:00"
        else:
            endtime= f"{hour+4}:00:00" 
        
        df = df[(df['time'] > starttime) & (df['time'] < endtime)]

        
        if len(df['conditions'].mode()) > 1:
            conditions = [x for x in df['conditions'].values]   
           
        else:
            conditions = df['conditions'].mode().values[0]     
        
        weather_data=pd.Series([df['temp'].mean(),df['precip'].mean(), df['humidity'].mean(), conditions])
    else:
        weather_data = weather_api_call_day(Location,StartDate)   

    return weather_data








def get_url_weather(url):

    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    tables = soup.find_all('tr')
    temp= None
    for t in tables:
        if len(t.get_text().split()) >0:
            if t.get_text().split()[0] == 'Weather':
                temp = t.get_text()
    if temp == None:
        try:
            driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
            #driver = webdriver.Chrome()
            driver.get(url)

            # isnpired by verenico nigro medium article
            button = driver.find_element(By.LINK_TEXT,'Italiano')
            button.click()
            table = driver.find_element(By.XPATH, "/html/body/div[3]/div[3]/div[5]/div[1]/table[1]/tbody/tr[9]")
            temp =table.text
        except:
            pass
    return temp
    
    