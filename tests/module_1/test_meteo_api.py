import pytest
import pandas as pd
from datetime import datetime

from src.module_1.module_1_meteo_api import validate_response


mock_response = {
    "latitude": 40.416775,
    "longitude": -3.703790,
    "generation_time_ms": 0.123,
    "utc_offset_seconds": 3600,
    "timezone": "Europe/Madrid",
    "timezone_abbreviation": "CET",
    "elevation": 667.0,
    "daily_units": {
        "time": "ISO8601",
        "temperature_2m_mean": "ºC",
        "precipitation_sum": "mm",
        "wind_speed_10m_max":"km/h",
    },
    "daily": {
        "time": ["2020-01-01", "2020-01-02"],
        "temperature_2m_mean": [10.5, 11.0],
        "precipitation_sum": [0.0, 5.0],
        "wind_speed_10m_max": [15.0, 20.0],
    },    
}


mock_daily_data = {
    "date": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
    "temperature_2m_mean(ºC)": [10.5, 11.0],
    "precipitation_sum(mm)": [0.0, 5.0],
    "wind_speed_10m_max(km/h)": [15.0, 20.0],
}

mock_monthly_data = {
    "date": [datetime(2020, 1, 1)],
    "temperature_2m_mean(ºC)": [10.75],
    "precipitation_sum(mm)": [2.5],
    "wind_speed_10m_max(km/h)": [17.5],
}


def test_validate_response():
    validate_response(mock_response)
    with pytest.raises(Exception):
        validate_response({"invalid": "response"})
