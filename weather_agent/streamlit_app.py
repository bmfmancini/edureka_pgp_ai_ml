#!/usr/bin/env python3
"""Streamlit UI for checking city weather via wttr.in."""

from __future__ import annotations

import requests
import streamlit as st


@st.cache_data(ttl=300)
def fetch_weather(city: str) -> dict:
    """Fetch weather data for a city from wttr.in."""
    response = requests.get(
        f"https://wttr.in/{city}",
        params={"format": "j1"},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def parse_weather(data: dict) -> dict:
    """Parse key weather fields from wttr.in response payload."""
    current = data.get("current_condition", [])
    if not current:
        raise ValueError("No current weather data returned.")

    current_data = current[0]

    area_list = data.get("nearest_area", [])
    area = area_list[0].get("areaName", [{"value": "Unknown"}])[0]["value"] if area_list else "Unknown"
    country = area_list[0].get("country", [{"value": "Unknown"}])[0]["value"] if area_list else "Unknown"

    weather_desc = current_data.get("weatherDesc", [{"value": "Unknown"}])

    return {
        "area": area,
        "country": country,
        "temperature_c": current_data.get("temp_C", "N/A"),
        "feels_like_c": current_data.get("FeelsLikeC", "N/A"),
        "humidity": current_data.get("humidity", "N/A"),
        "wind_kmph": current_data.get("windspeedKmph", "N/A"),
        "condition": weather_desc[0].get("value", "Unknown"),
    }


def main() -> None:
    st.set_page_config(
        page_title="Weather UI",
        page_icon=":sun_behind_cloud:",
        layout="centered",
    )

    st.title("Weather UI")
    st.caption("Powered by wttr.in")

    city = st.text_input("City name", placeholder="e.g. Toronto")

    if st.button("Check weather", type="primary"):
        city = city.strip()
        if not city:
            st.warning("Enter a city name.")
            return

        with st.spinner("Fetching latest weather..."):
            try:
                data = fetch_weather(city)
                weather = parse_weather(data)
            except requests.exceptions.RequestException as exc:
                st.error(f"Network/API error: {exc}")
                return
            except (KeyError, IndexError, ValueError) as exc:
                st.error(f"Unexpected weather response: {exc}")
                return

        st.subheader(f"{weather['area']}, {weather['country']}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Temperature", f"{weather['temperature_c']} degC")
        col2.metric("Feels Like", f"{weather['feels_like_c']} degC")
        col3.metric("Humidity", f"{weather['humidity']}%")

        col4, col5 = st.columns(2)
        col4.metric("Condition", weather["condition"])
        col5.metric("Wind", f"{weather['wind_kmph']} km/h")


if __name__ == "__main__":
    main()
