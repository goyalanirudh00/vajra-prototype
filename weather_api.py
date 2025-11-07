"""
Weather API integration and anomaly detection module.
Uses OpenWeatherMap API (free tier) to fetch real-time weather data
and detect anomalies based on current conditions.
"""

import requests
import os
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta

# Try to import streamlit, but handle if not available
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


def get_openweather_api_key() -> Optional[str]:
    """Get OpenWeatherMap API key from Streamlit secrets or environment variables."""
    try:
        # Try Streamlit secrets first (if available)
        if STREAMLIT_AVAILABLE and hasattr(st, 'secrets'):
            try:
                key = st.secrets.get("OPENWEATHER_API_KEY")
                if key and key != "your-openweather-api-key-here":
                    return key
            except:
                pass
            try:
                key = st.secrets["OPENWEATHER_API_KEY"]
                if key and key != "your-openweather-api-key-here":
                    return key
            except:
                pass
        
        # Try environment variable
        key = os.getenv("OPENWEATHER_API_KEY")
        if key and key != "your-openweather-api-key-here":
            return key
    except Exception:
        pass
    
    return None


def geocode_location(location: str, api_key: str) -> Optional[Dict]:
    """
    Geocode a location name to get coordinates using OpenWeatherMap Geocoding API.
    Returns: {lat, lon, name, country} or None
    """
    try:
        url = "http://api.openweathermap.org/geo/1.0/direct"
        params = {
            "q": location,
            "limit": 1,
            "appid": api_key
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data and len(data) > 0:
            return {
                "lat": data[0]["lat"],
                "lon": data[0]["lon"],
                "name": data[0].get("name", location),
                "country": data[0].get("country", ""),
                "state": data[0].get("state", "")
            }
    except Exception as e:
        print(f"Geocoding error for {location}: {e}")
    
    return None


def get_current_weather(lat: float, lon: float, api_key: str) -> Optional[Dict]:
    """
    Get current weather data from OpenWeatherMap API.
    Returns: {temp, feels_like, humidity, pressure, wind_speed, wind_gust, 
              rain_1h, snow_1h, weather_main, weather_description, visibility}
    """
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": api_key,
            "units": "metric"  # Use metric units
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        return {
            "temp": data["main"]["temp"],
            "feels_like": data["main"]["feels_like"],
            "humidity": data["main"]["humidity"],
            "pressure": data["main"]["pressure"],
            "wind_speed": data["wind"].get("speed", 0),  # m/s
            "wind_gust": data["wind"].get("gust", 0),  # m/s
            "rain_1h": data.get("rain", {}).get("1h", 0),  # mm
            "snow_1h": data.get("snow", {}).get("1h", 0),  # mm
            "weather_main": data["weather"][0]["main"],
            "weather_description": data["weather"][0]["description"],
            "visibility": data.get("visibility", 10000) / 1000,  # Convert to km
            "clouds": data.get("clouds", {}).get("all", 0)
        }
    except Exception as e:
        print(f"Weather API error: {e}")
    
    return None


def get_forecast(lat: float, lon: float, api_key: str, days: int = 5) -> Optional[Dict]:
    """
    Get weather forecast from OpenWeatherMap API.
    Returns: List of forecast data for the next few days.
    """
    try:
        url = "https://api.openweathermap.org/data/2.5/forecast"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": api_key,
            "units": "metric",
            "cnt": min(days * 8, 40)  # 8 forecasts per day, max 40
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        forecasts = []
        for item in data.get("list", []):
            forecasts.append({
                "dt": item["dt"],
                "dt_txt": item["dt_txt"],
                "temp": item["main"]["temp"],
                "feels_like": item["main"]["feels_like"],
                "humidity": item["main"]["humidity"],
                "pressure": item["main"]["pressure"],
                "wind_speed": item["wind"].get("speed", 0),
                "wind_gust": item["wind"].get("gust", 0),
                "rain_3h": item.get("rain", {}).get("3h", 0),
                "snow_3h": item.get("snow", {}).get("3h", 0),
                "weather_main": item["weather"][0]["main"],
                "weather_description": item["weather"][0]["description"],
                "clouds": item.get("clouds", {}).get("all", 0)
            })
        
        return forecasts
    except Exception as e:
        print(f"Forecast API error: {e}")
    
    return None


def detect_anomaly(current_weather: Dict, forecast: Optional[Dict] = None) -> Tuple[bool, Dict]:
    """
    Enhanced anomaly detection model based on weather thresholds and forecast consistency.
    
    Anomaly indicators:
    - High wind speed (> 15 m/s or ~54 km/h)
    - Heavy rain (> 10 mm/hour)
    - Extreme temperatures (very hot > 40°C or very cold < -10°C)
    - Low visibility (< 1 km)
    - Severe weather conditions (Thunderstorm, Heavy Rain, etc.)
    
    Confidence calculation factors:
    - Number and severity of indicators
    - Forecast consistency (multiple forecast periods showing similar conditions)
    - Data quality (complete vs partial data)
    - Historical context (extreme values vs moderate)
    
    Returns: (is_anomaly: bool, anomaly_details: Dict)
    """
    anomaly_details = {
        "severity": "none",
        "type": None,
        "confidence": 0.0,
        "indicators": []
    }
    
    if not current_weather:
        return False, anomaly_details
    
    indicators = []
    severity_score = 0
    confidence_factors = []
    
    # Wind speed anomaly (15 m/s = 54 km/h, 20 m/s = 72 km/h)
    wind_speed_ms = current_weather.get("wind_speed", 0)
    wind_gust_ms = current_weather.get("wind_gust", 0) or wind_speed_ms
    wind_speed_kmh = wind_speed_ms * 3.6
    
    if wind_gust_ms > 20:  # > 72 km/h - severe
        indicators.append(f"Severe wind gusts: {wind_gust_ms * 3.6:.0f} km/h")
        severity_score += 3
        confidence_factors.append(0.15)  # High confidence for extreme values
        anomaly_details["type"] = "High Wind"
    elif wind_speed_ms > 15:  # > 54 km/h - moderate
        indicators.append(f"High wind speed: {wind_speed_kmh:.0f} km/h")
        severity_score += 2
        confidence_factors.append(0.10)
        if not anomaly_details["type"]:
            anomaly_details["type"] = "High Wind"
    
    # Rainfall anomaly (10 mm/h = heavy, 25 mm/h = very heavy)
    rain_1h = current_weather.get("rain_1h", 0)
    if rain_1h > 25:  # Very heavy rain
        indicators.append(f"Very heavy rainfall: {rain_1h:.0f} mm/hour")
        severity_score += 3
        confidence_factors.append(0.15)
        anomaly_details["type"] = "Heavy Rainfall"
    elif rain_1h > 10:  # Heavy rain
        indicators.append(f"Heavy rainfall: {rain_1h:.0f} mm/hour")
        severity_score += 2
        confidence_factors.append(0.10)
        if not anomaly_details["type"]:
            anomaly_details["type"] = "Heavy Rainfall"
    
    # Temperature extremes
    temp = current_weather.get("temp", 20)
    if temp > 40:
        indicators.append(f"Extreme heat: {temp:.0f}°C")
        severity_score += 2
        confidence_factors.append(0.12)
        if not anomaly_details["type"]:
            anomaly_details["type"] = "Heat Wave"
    elif temp < -10:
        indicators.append(f"Extreme cold: {temp:.0f}°C")
        severity_score += 2
        confidence_factors.append(0.12)
        if not anomaly_details["type"]:
            anomaly_details["type"] = "Cold Wave"
    
    # Visibility anomaly
    visibility_km = current_weather.get("visibility", 10)
    if visibility_km < 1:
        indicators.append(f"Very low visibility: {visibility_km:.1f} km")
        severity_score += 2
        confidence_factors.append(0.12)
        if not anomaly_details["type"]:
            anomaly_details["type"] = "Low Visibility"
    elif visibility_km < 3:
        indicators.append(f"Reduced visibility: {visibility_km:.1f} km")
        severity_score += 1
        confidence_factors.append(0.05)
    
    # Weather condition severity
    weather_main = current_weather.get("weather_main", "").lower()
    if weather_main in ["thunderstorm", "extreme"]:
        indicators.append(f"Severe weather: {current_weather.get('weather_description', '')}")
        severity_score += 3
        confidence_factors.append(0.15)
        if not anomaly_details["type"]:
            anomaly_details["type"] = "Severe Storm"
    elif weather_main in ["heavy rain", "squall"]:
        indicators.append(f"Severe weather: {current_weather.get('weather_description', '')}")
        severity_score += 2
        confidence_factors.append(0.10)
        if not anomaly_details["type"]:
            anomaly_details["type"] = "Heavy Rain"
    
    # Check forecast for upcoming severe weather and consistency
    forecast_consistency = 0
    forecast_severe_count = 0
    if forecast and len(forecast) > 0:
        # Check next 24-48 hours for severe conditions
        for fc in forecast[:8]:  # Next 24 hours (8 forecasts)
            fc_wind = fc.get("wind_speed", 0) or fc.get("wind_gust", 0)
            fc_rain = fc.get("rain_3h", 0) / 3  # Convert to per hour
            fc_weather = fc.get("weather_main", "").lower()
            
            if fc_wind > 15 or fc_rain > 10 or fc_weather in ["thunderstorm", "extreme"]:
                forecast_severe_count += 1
                if forecast_severe_count == 1:
                    indicators.append(f"Severe conditions forecasted in next 24 hours")
                severity_score += 1
        
        # Forecast consistency: more forecast periods showing severe = higher confidence
        if forecast_severe_count >= 4:
            forecast_consistency = 0.15  # Very consistent
        elif forecast_severe_count >= 2:
            forecast_consistency = 0.10  # Moderately consistent
        elif forecast_severe_count >= 1:
            forecast_consistency = 0.05  # Some consistency
    
    # Determine if anomaly exists
    is_anomaly = severity_score >= 2
    
    if is_anomaly:
        # Base confidence on severity score
        if severity_score >= 5:
            anomaly_details["severity"] = "severe"
            base_confidence = 0.75
        elif severity_score >= 3:
            anomaly_details["severity"] = "moderate"
            base_confidence = 0.65
        else:
            anomaly_details["severity"] = "mild"
            base_confidence = 0.55
        
        # Add confidence from multiple indicators (more indicators = higher confidence)
        indicator_bonus = min(sum(confidence_factors), 0.20)  # Cap at 20%
        
        # Add forecast consistency bonus
        final_confidence = base_confidence + indicator_bonus + forecast_consistency
        
        # Cap confidence between 0.55 and 0.95
        anomaly_details["confidence"] = min(max(final_confidence, 0.55), 0.95)
        
        # Round to 2 decimal places for cleaner display
        anomaly_details["confidence"] = round(anomaly_details["confidence"], 2)
        
        anomaly_details["indicators"] = indicators
    
    return is_anomaly, anomaly_details


def fetch_weather_data(location: str) -> Optional[Dict]:
    """
    Main function to fetch weather data for a location.
    Returns: {
        "location": {...},
        "current": {...},
        "forecast": [...],
        "anomaly": {...}
    } or None
    """
    api_key = get_openweather_api_key()
    if not api_key:
        return None
    
    # Geocode location
    location_data = geocode_location(location, api_key)
    if not location_data:
        return None
    
    # Get current weather
    current_weather = get_current_weather(
        location_data["lat"],
        location_data["lon"],
        api_key
    )
    if not current_weather:
        return None
    
    # Get forecast
    forecast = get_forecast(
        location_data["lat"],
        location_data["lon"],
        api_key,
        days=5
    )
    
    # Detect anomaly
    is_anomaly, anomaly_details = detect_anomaly(current_weather, forecast)
    
    return {
        "location": location_data,
        "current": current_weather,
        "forecast": forecast or [],
        "anomaly": {
            "detected": is_anomaly,
            **anomaly_details
        }
    }

