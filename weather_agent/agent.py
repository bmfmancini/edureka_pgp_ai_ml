#!/usr/bin/env python3
"""Simple weather assistant script converted from a notebook."""

import asyncio
import os
from pathlib import Path
from typing import Annotated

import requests
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core.models import ModelCapabilities
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient

def _load_env_file(file_path: str | None = None) -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ."""
    if file_path is None:
        file_path = str(Path(__file__).resolve().parent / ".env")

    if not os.path.exists(file_path):
        return

    try:
        with open(file_path, "r", encoding="utf-8") as env_file:
            for raw_line in env_file:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key:
                    os.environ[key] = value
    except OSError:
        # Ignore file-read issues and rely on existing environment variables.
        return


# Load variables from .env (if present) before reading keys.
_load_env_file()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")


def get_weather(city: Annotated[str, "city name"]) -> str:
    """Get weather for a city using wttr.in."""
    try:
        city = city.strip().title()

        response = requests.get(
            f"https://wttr.in/{city}",
            params={"format": "j1"},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        current = data.get("current_condition", [])
        if not current:
            return f"Could not fetch weather for: {city}"

        current_data = current[0]
        temp = current_data.get("temp_C", "N/A")
        feel_like = current_data.get("FeelsLikeC", "N/A")
        wind = current_data.get("windspeedKmph", "N/A")
        condition = current_data.get("weatherDesc", [{"value": "Unknown"}])[0]["value"]
        humidity = current_data.get("humidity", "N/A")

        return (
            f"\nweather in {city}:\n"
            f"\nTemperature: {temp} degC\n"
            f"Feels like: {feel_like} degC\n"
            f"Wind speed: {wind} km/h\n"
            f"Condition: {condition}\n"
            f"Humidity: {humidity}%\n"
        )
    except requests.exceptions.RequestException as exc:
        return f"Error: {exc}"


def create_weather_agent() -> AssistantAgent:
    groq_capabilities = ModelCapabilities(
        vision=False,
        function_calling=True,
        json_output=True,
    )

    model_client = OpenAIChatCompletionClient(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
        model_capabilities=groq_capabilities,
    )

    return AssistantAgent(
        name="weather_agent",
        model_client=model_client,
        tools=[FunctionTool(get_weather, description="Get weather for a city")],
        system_message=(
            "You are a weather assistant.\n"
            "When user asks about weather for a particular city, ALWAYS use the get_weather tool.\n"
            "Then explain the results clearly and precisely."
        ),
    )


async def run() -> None:
    if not GROQ_API_KEY:
        raise RuntimeError("Missing GROQ_API_KEY environment variable")

    weather_agent = create_weather_agent()
    print("==== Weather assistant ====\n")

    while True:
        city = input("Enter city name (or type 'exit'): ").strip()
        if city.lower() == "exit":
            print("Goodbye!")
            break
        if not city:
            continue

        task = f"What is the weather in {city}?"
        termination = MaxMessageTermination(max_messages=4)
        team = RoundRobinGroupChat([weather_agent], termination_condition=termination)
        results = await team.run(task=task)

        for message in results.messages:
            if message.source == "weather_agent":
                print(f"Assistant: {message.content}")


if __name__ == "__main__":
    asyncio.run(run())
