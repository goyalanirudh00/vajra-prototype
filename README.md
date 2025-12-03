# Vajra Prototype

Weather & Disruption Intelligence Layer for Food Delivery Operations

A forecasting platform that demonstrates anomaly-aware demand forecasting using AI-powered predictions.

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up API keys (optional - app works with mock data if not provided):
   - Create `.streamlit/secrets.toml` or set environment variables:
     - `OPENAI_API_KEY` - Get from https://platform.openai.com/api-keys
     - `OPENWEATHER_API_KEY` - Get from https://openweathermap.org/api
     - `GOOGLE_PLACES_API_KEY` - Get from Google Cloud Console

3. Run the app:
```bash
streamlit run app.py
```

4. Open `http://localhost:8501` in your browser

## Features

- AI-powered SKU generation using GPT-4o-mini
- Real-time weather integration with anomaly detection
- Location-specific weather anomaly predictions
- Anomaly-aware demand forecasting
- Cost analysis with accuracy and savings metrics
- Actionable, location-specific recommendations

## Usage

1. Enter a retailer name (e.g., "Target", "Whole Foods")
2. Enter an area/city/zip code (e.g., "San Jose", "94102")
3. Click "Evaluate Forecast"
4. View results in three tabs: Anomaly Alerts, Forecast Analysis, and Recommendations

## Documentation

Detailed guides are in the [`docs/`](./docs/) folder:

- [Development Guide](./docs/DEV_GUIDE.md) - Local development with Docker
- [Secrets Management](./docs/SECRETS_GUIDE.md) - Managing API keys
- [Container Deployment](./docs/CONTAINER_DEPLOYMENT.md) - Docker deployment
- [Deployment Options](./docs/DEPLOYMENT.md) - Streamlit Cloud and other platforms
- [Production Checklist](./docs/PRODUCTION_CHECKLIST.md) - Production readiness
- [Metrics System](./docs/METRICS.md) - Metrics tracking

## Deployment

**Streamlit Cloud**: Push to GitHub and deploy on [share.streamlit.io](https://share.streamlit.io)

**Docker**: See [Container Deployment Guide](./docs/CONTAINER_DEPLOYMENT.md) for Railway, Render, Fly.io, etc.

**Local Development**: See [Development Guide](./docs/DEV_GUIDE.md) for Docker setup

## Technical Details

- Uses OpenAI GPT-4o-mini for SKU generation and anomaly predictions
- API key lookup order: Streamlit secrets → environment variables → mock data fallback
- Responses are cached to minimize API calls
- SQLite database for metrics tracking
