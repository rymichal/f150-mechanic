import random
from langchain.tools import tool


@tool
def get_weather(location: str) -> str:
    """
    Get the current weather for a given location.

    Args:
        location: The city name or city name with state/country (e.g., "Grand Rapids, Michigan" or "London, UK")

    Returns:
        A string describing the current weather conditions
    """
    # Fake weather data - simulates OpenWeatherMap API response
    weather_conditions = [
        "clear sky",
        "few clouds",
        "scattered clouds",
        "broken clouds",
        "light rain",
        "moderate rain",
        "overcast clouds",
        "mist",
        "partly cloudy"
    ]

    # Generate realistic fake data
    temp = random.randint(45, 85)
    feels_like = temp + random.randint(-5, 5)
    humidity = random.randint(30, 90)
    description = random.choice(weather_conditions)

    # Clean up the location name for display
    city_name = location.split(',')[0].strip().title()

    weather_report = (
        f"Weather in {city_name}:\n"
        f"Temperature: {temp}°F (feels like {feels_like}°F)\n"
        f"Conditions: {description.capitalize()}\n"
        f"Humidity: {humidity}%"
    )

    return weather_report
