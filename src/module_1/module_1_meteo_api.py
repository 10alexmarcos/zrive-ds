import time
import requests
import logging
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from jsonschema import validate, ValidationError

logger = logging.getLogger(__name__)  # uses file name as name of the log
logger.level = logging.INFO

API_URL = "https://archive-api.open-meteo.com/v1/archive"

GENERAL_PARAMS = {
    "start_date": "2010-01-01",
    "end_date": "2020-12-31",
    "daily": "temperature_2m_mean,precipitation_sum,wind_speed_10m_max",
}

CITIES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.12775},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "latitude": {"type": "number"},
        "longitude": {"type": "number"},
        "generationtime_ms": {"type": "number"},
        "utc_offset_seconds": {"type": "number"},
        "timezone": {"type": "string"},
        "timezone_abbreviation": {"type": "string"},
        "elevation": {"type": "number"},
        "daily_units": {
            "type": "object",
            "properties": {
                "time": {"type": "string"},
                "temperature_2m_mean": {"type": "string"},
                "precipitation_sum": {"type": "string"},
                "wind_speed_10m_max": {"type": "string"},
            },
        },
        "daily": {
            "type": "object",
            "properties": {
                "time": {"type": "array", "items": {"type": "string"}},
                "temperature_2m_mean": {"type": "array", "items": {"type": "number"}},
                "precipitation_sum": {"type": "array", "items": {"type": "number"}},
                "wind_speed_10m_max": {"type": "array", "items": {"type": "number"}},
            },
            "required": [
                "time",
                "temperature_2m_mean",
                "precipitation_sum",
                "wind_speed_10m_max",
            ],
        },
    },
    "required": ["latitude", "longitude", "daily"],
}


def make_api_call(url: str, params: dict, retries: int = 5, cooldown: int = 2):
    for attempt in range(retries):
        response = requests.get(url, params=params)

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            logging.warning(f"Rate limit hit, retrying in {cooldown} seconds...")
            time.sleep(cooldown)
            cooldown *= 2
        else:
            logging.error(
                f"Error fetching data (status code{response.status_code}): {response.text}"
            )
            time.sleep(cooldown)
            cooldown *= 2

    logging.error(f"Failed to fetch data after {retries} attempts")
    raise Exception(f"Failed to fetch data after {retries} attempts")


def get_data_meteo_api(city: str, api_call: callable = make_api_call):
    if city not in CITIES:
        raise ValueError(f"City '{city}' not found in this project")

    params = {**GENERAL_PARAMS, **CITIES[city]}
    response_json = api_call(API_URL, params=params)

    print(f"You have selected the city: {city}")
    return response_json


def validate_response(response):
    try:
        validate(instance=response, schema=RESPONSE_SCHEMA)
        print("Schema validation passed.")
    except ValidationError as e:
        print(f"Response validation failed: {e}")
        raise


def process_response(response: dict, city: str) -> pd.DataFrame:
    print(f"Coordinates of {city}: {response['latitude']}°N {response['longitude']}°E")

    daily = response["daily"]
    temperature_2m_mean = daily["temperature_2m_mean"]
    precipitation_sum = daily["precipitation_sum"]
    wind_speed_10m_max = daily["wind_speed_10m_max"]

    daily_data = {
        "date": pd.date_range(
            start=pd.to_datetime(daily["time"][0], format="%Y-%m-%d"),
            end=pd.to_datetime(daily["time"][-1], format="%Y-%m-%d"),
            freq="D",
        )
    }

    daily_data["temperature_2m_mean(ºC)"] = temperature_2m_mean
    daily_data["precipitation_sum(mm)"] = precipitation_sum
    daily_data["wind_speed_10m_max(km/h)"] = wind_speed_10m_max

    daily_dataframe = pd.DataFrame(data=daily_data)
    print(f"This is the daily dataframe for the city {city}\n", daily_dataframe)
    return daily_dataframe


def daily_data_to_monthly_data(df: pd.DataFrame, city: str) -> pd.DataFrame:
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    # print(df.dtypes), used to be sure that I can use .mean(numeric_only)
    df_monthly = df.groupby(["year", "month"]).mean(numeric_only=True).reset_index()
    df_monthly["date"] = pd.to_datetime(df_monthly[["year", "month"]].assign(day=1))
    df_monthly = df_monthly.drop(columns=["year", "month"])

    df_monthly = df_monthly[
        [
            "date",
            "temperature_2m_mean(ºC)",
            "precipitation_sum(mm)",
            "wind_speed_10m_max(km/h)",
        ]
    ]

    print(f"This is the monthly dataframe for the city {city}\n", df_monthly)
    return df_monthly


def visualize_evolution(df: pd.DataFrame, city: str) -> None:
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(13, 13))
    plt.subplots_adjust(hspace=0.5)

    date_format = mdates.DateFormatter("%Y-%m")

    for i, col in enumerate(df.columns[1:]):
        axes[i].plot(df["date"], df[col], label=col, color="blue")
        axes[i].set_ylabel("Values")
        axes[i].set_title(f"{col} evolution for {city}")
        axes[i].grid(True)
        axes[i].legend()

        axes[i].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
        axes[i].xaxis.set_major_formatter(date_format)

        plt.setp(axes[i].xaxis.get_majorticklabels(), rotation=45, ha="right")

    axes[-1].set_xlabel("Date")
    filename=f"src/module_1/{city}_evolution.png"
    plt.savefig(filename, bbox_inches="tight")
    plt.show()


def main():
    city = "Rio"
    response = get_data_meteo_api(city)
    validate_response(response)
    df = process_response(response, city)
    df_monthly = daily_data_to_monthly_data(df, city)
    visualize_evolution(df_monthly, city)


if __name__ == "__main__":
    main()
