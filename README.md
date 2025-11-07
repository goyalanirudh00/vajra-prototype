# Vajra Prototype

**Weather & Disruption Intelligence Layer for Food Delivery Operations**

A futuristic-elegant, tech-minimal forecasting platform that demonstrates anomaly-aware demand forecasting using AI-powered predictions.

## 🚀 Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up OpenAI API key (required for LLM features):
   - Option 1: Create `.streamlit/secrets.toml` file with:
     ```toml
     OPENAI_API_KEY = "your-api-key-here"
     ```
   - Option 2: Set environment variable:
     ```bash
     export OPENAI_API_KEY="your-api-key-here"
     ```
   - Note: If no API key is provided, the app will use fallback mock data

3. Run the app:
```bash
streamlit run app.py
```

4. Open your browser to the URL shown (typically `http://localhost:8501`)

## ✨ Features

- **AI-Powered SKU Generation**: GPT-4o-mini generates realistic, retailer-specific SKUs
- **Location-Specific Anomalies**: Highly personalized weather anomalies with local references (neighborhoods, ZIP codes, landmarks)
- **Anomaly-Aware Forecasting**: Vajra forecast (intelligent) vs. generic moving average
- **Cost Analysis**: Tail-loss multipliers with accuracy and savings metrics
- **Actionable Recommendations**: Context-aware, locality-specific recommendations
- **Brand Design**: Futuristic-elegant UI with Diamond Storm color palette
- **Three-Tab Interface**: Anomaly Alerts, Forecast Analysis, and Recommendations

## 🎨 Design

- **Color Palette**: Electric Violet (#7F5AF0), Deep Purple (#6438B7), Crystal Blue-Gray (#7A88A1)
- **Typography**: Space Grotesk (headers), Inter (body), Orbitron (logo)
- **Aesthetic**: Clean, minimal, modern with geometric precision

## 📖 Usage

1. Enter a **Retailer Name** (e.g., "Target", "Whole Foods", "Walmart")
2. Enter an **Area / City / Pin Code** (e.g., "San Jose", "Chicago", "94102")
3. Click **Evaluate Forecast** button
4. View results in three tabs:
   - **Anomaly Alerts**: Detected weather anomalies with locality-specific details
   - **Forecast Analysis**: Interactive chart and accuracy/savings comparison
   - **Recommendations**: Actionable, location-specific recommendations

## 🌐 Deployment

### Streamlit Cloud (Recommended)

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app"
5. Select your repository and set:
   - **Main file path**: `app.py`
   - **Python version**: 3.9+
6. Add your OpenAI API key in the "Secrets" section:
   ```toml
   OPENAI_API_KEY = "your-api-key-here"
   ```
7. Click "Deploy"

### Local Deployment

See Quick Start section above.

## 🔮 Future Extensions

- **Multi-store Comparison**: Compare forecasts across multiple locations
- **Live Data Integration**: Connect to real-time demand and weather APIs
- **Advanced Forecasting**: Machine learning models for demand prediction
- **Historical Analysis**: Compare forecasts against historical anomalies

## LLM Integration

The app now uses **OpenAI GPT-4o-mini** for intelligent predictions:

- **SKU Generation**: Generates realistic, retailer-specific SKUs based on product mix
- **Weather Anomaly Prediction**: Predicts location-specific weather anomalies for September-November
- **Recommendations**: Provides actionable, context-aware recommendations for each SKU-anomaly combination

### API Key Setup

The app requires an OpenAI API key. It will:
1. Check Streamlit secrets (`.streamlit/secrets.toml`)
2. Fall back to `OPENAI_API_KEY` environment variable
3. Use mock data if no API key is found

### Cost Efficiency

- Uses **GPT-4o-mini** model (cost-effective, high quality)
- Responses are cached to minimize API calls
- Automatic fallback to mock data if API fails


