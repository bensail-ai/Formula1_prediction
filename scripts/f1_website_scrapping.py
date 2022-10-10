#%%
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import requests
import json
import pandas as pd
#%%

# url = 'https://www.formula1.com/en/results.html'

# driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

# driver.get(url)
# xpath="/html/body/div[1]/main/article/div/div[2]/form/div[1]/select"
# #xpath="//select[(@class='resultsarchive-filter-form-select') and (@name = 'year')"
# year = driver.find_elements(By.XPATH,xpath)

# options = [x for x in year[0].find_elements(By.TAG_NAME,"option")]
# for element in options:
#     if element == '1995':
#         element.click()
#         break
# xpath="/html/body/div[2]/main/article/div/div[2]/form/div[3]/select"
# race = driver.find_element(By.XPATH,xpath)
# print(race)
# options = [x for x in race.find_elements(By.TAG_NAME,"option")]

# for element in options:
#     print(element)



# %%
url= 'https://www.formula1.com/en/results.html/1995/races/637/australia/qualifying-0.html'
page = requests.get(url)
df_q3 = pd.read_html(page.content)[0]

url= 'https://www.formula1.com/en/results.html/1995/races/637/australia/qualifying-1.html'
page = requests.get(url)
df_q1 = pd.read_html(page.content)[0]
url= 'https://www.formula1.com/en/results.html/1995/races/637/australia/qualifying-2.html'
page = requests.get(url)
df_q2 = pd.read_html(page.content)[0]
# %%
df_q3.drop(columns=['Unnamed: 0','Unnamed: 7'],axis=0,inplace=True)
df_q2.drop(columns=['Unnamed: 0','Unnamed: 7'],axis=0,inplace=True)
df_q1.drop(columns=['Unnamed: 0','Unnamed: 7'],axis=0,inplace=True)