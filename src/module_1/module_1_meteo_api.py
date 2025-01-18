""" This is a dummy example """
import openmeteo_requests
import requests_cache
import pandas as pd 
from retry_requests import retry
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
import seaborn as sns

#Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo= openmeteo_requests.Client(session = retry_session)

API_URL = "https://archive-api.open-meteo.com/v1/archive"

general_params = {
    "start_date": "2010-01-01",
    "end_date": "2020-12-31",
    "daily": "temperature_2m_mean,precipitation_sum,wind_speed_10m_max"
}
cities = {
    "Madrid": {"latitude": 40.416775,"longitude": -3.703790},
    "London": {"latitude": 51.507351,"longitude": -0.12775},
    "Rio": {"latitude": -22.906847,"longitude": -43.172896} 
}

def get_data_meteo_api(city:str):
    if city not in cities:
        raise ValueError(f"City '{city}' not found in this project")

    params= {**general_params, **cities[city]}
    responses = openmeteo.weather_api(API_URL, params= params)
    response = responses [0]
    print(f"The information shown from now on is about the city {city}")
    return response

def process_response(response, city:str):
    print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation {response.Elevation()} m asl")
    print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

    daily = response.Daily()
    daily_temperature_2m_mean = daily.Variables(0).ValuesAsNumpy()
    precipitation_sum = daily.Variables(1).ValuesAsNumpy()
    wind_speed_10m_max = daily.Variables(2).ValuesAsNumpy()

    daily_data = {"date": pd.date_range(
	start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
	end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
	freq = pd.Timedelta(seconds = daily.Interval()),
	inclusive = "left"
    )}

    daily_data["temperature_2m_mean"] = daily_temperature_2m_mean
    daily_data["precipitation_sum"] = precipitation_sum
    daily_data["wind_speed_10m_max"] = wind_speed_10m_max

    daily_dataframe = pd.DataFrame(data = daily_data)
    print(f"This is the daily dataframe os the city {city}\n", daily_dataframe)
    return daily_dataframe


def daily_data_to_monthly_data(df: pd.DataFrame, city:str):

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month


    df_monthly = df.groupby(['year', 'month']).mean(numeric_only=True).reset_index()
    
    df_monthly['date'] = pd.to_datetime(df_monthly[['year', 'month']].assign(day=1))

    df_monthly = df_monthly.drop(columns=['year', 'month'])

    df_monthly = df_monthly[['date', 'temperature_2m_mean', 'precipitation_sum', 'wind_speed_10m_max']]

    print("This is the monthly dataframe of the city {city}\n", df_monthly)
    return df_monthly
    

def visualize_evolution(df: pd.DataFrame, city:str):
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12,12))
    plt.subplots_adjust(hspace=0.5)

    date_format = mdates.DateFormatter('%Y-%m')

    for i, col in enumerate(df.columns[1:]):
        axes[i].plot(df['date'], df[col], label=col, color='blue')
        axes[i].set_ylabel('Values')
        axes[i].set_title(f"{col} evolution for {city}")
        axes[i].grid(True)
        axes[i].legend()

        axes[i].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,7]))
        axes[i].xaxis.set_major_formatter(date_format)

        plt.setp(axes[i].xaxis.get_majorticklabels(), rotation=45, ha="right")
    
    axes[-1].set_xlabel('Date')
    plt.show()

def visualize_histogram(df: pd.DataFrame, city: str):
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 12))
    plt.subplots_adjust(hspace=0.5)

    date_format = mdates.DateFormatter('%Y-%m')

    for i, col in enumerate(df.columns[1:]):  
        axes[i].hist(df['date'], bins=pd.date_range(df['date'].min(), df['date'].max(), freq='M'), weights=df[col], color='blue', edgecolor='black', alpha=0.7)
        axes[i].set_ylabel('Frequency')
        axes[i].set_title(f"{col} distribution for {city}")
        axes[i].grid(True)
        axes[i].legend([col])

        axes[i].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,7])) 
        axes[i].xaxis.set_major_formatter(date_format)  
        plt.setp(axes[i].xaxis.get_majorticklabels(), rotation=45, ha="right")

    axes[-1].set_xlabel(f'{col} Values')  # Solo en el último gráfico
    plt.show()

def main():
    city="Madrid"
    response=get_data_meteo_api(city)
    df=process_response(response, city)
    df_monthly=daily_data_to_monthly_data(df, city)
    visualize_evolution(df_monthly, city)
    visualize_histogram(df_monthly,city)
        
        

if __name__ == "__main__":
    main()
