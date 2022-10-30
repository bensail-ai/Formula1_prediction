import requests
import pandas as pd
#%%
def _get_json_content_from_url(url, *args, timeout: int = 15, **kwargs):
    """Returns JSON content from requestsm URL"""
    return requests.get(url, *args, timeout=timeout, **kwargs).json()


def get_current_year():
    # Requesting the season list and get the last one to initialize CURR_YEAR
    CURR_YEAR = int(
        _get_json_content_from_url(
            "http://ergast.com/api/f1/seasons.json?limit=200"
        )["MRData"]["SeasonTable"]["Seasons"][-1]["season"]
    )
    return CURR_YEAR

#%%
class QualifyingResults:
    """Class which contains the methods which provide the details qualifying_results()"""
    def __init__(self, results):
        self.results = results

    def get_positions(self):
        """Returns a list of driver positions"""
        return [i["position"] for i in self.results]

    def get_names(self):
        """Returns a list of driver names"""
        return [" ".join([i["Driver"]["givenName"],
                          i["Driver"]["familyName"]]) for i in self.results]

    def get_driver_numbers(self):
        """Returns a list of driver numbers"""
        return [i["number"] for i in self.results]

    def get_constructors(self):
        """Returns the list of name of the constructors"""
        return [i["Constructor"]["name"] for i in self.results]

    def get_q1_times(self):
        """Returns a list of Q1 timings"""
        r_q1 = []
        for i in self.results:
            if "Q1" in i:
                r_q1.append(i["Q1"])
        return r_q1

    def get_q2_times(self):
        """Returns a list of Q2 timings"""
        r_q2 = []
        for i in self.results:
            if "Q2" in i:
                r_q2.append(i["Q2"])
            else:
                r_q2.append('')
        return r_q2

    def get_q3_times(self):
        """Returns a list of Q3 timings"""
        r_q3 = []
        for i in self.results:
            if "Q3" in i:
                r_q3.append(i["Q3"])
            else:
                r_q3.append('')
        return r_q3



#%%
def qualifying_results(year: int, race_round: int,current_year=False):
    """Returns the driver name , driver position, driver number, constructor name , the 3 Q times"""
    if year < 2003 or year > CURR_YEAR:
        raise ValueError(
            f"Only years between 2003 and {CURR_YEAR} are considered as valid value for year"
        )
    if current_year == True:
        year = get_current_year()
    
    json_data = _get_json_content_from_url(
        f"https://ergast.com/api/f1/{year}/{race_round}/qualifying.json"
    )
    schedule_json = json_data["MRData"]["RaceTable"]["Races"][0]["QualifyingResults"]
    r_obj = QualifyingResults(schedule_json)
    driver_positions = r_obj.get_positions()
    driver_names = r_obj.get_names()
    driver_numbers = r_obj.get_driver_numbers()
    constructor_names = r_obj.get_constructors()
    q1_times = r_obj.get_q1_times()
    q2_times = r_obj.get_q2_times()
    q3_times = r_obj.get_q3_times()
    return pd.DataFrame(
        zip(
            driver_positions,
            driver_names,
            driver_numbers,
            constructor_names,
            q1_times,
            q2_times,
            q3_times
        ),
        columns=[
            "Position",
            "DriverName",
            "DriverNumber",
            "Constructor",
            "Q1",
            "Q2",
            "Q3"
        ],
    )
# %%
