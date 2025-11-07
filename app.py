"""
Vajra Streamlit Prototype
Weather & Disruption Intelligence Layer for Food Delivery Operations
Demonstrates anomaly-aware forecasting vs. generic forecasts
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import List, Dict, Optional
import json
import os
import time
from datetime import datetime, timedelta
from metrics import (
    init_database, track_button_click, track_page_load, track_evaluation_performance,
    get_button_click_count, get_average_page_load_time, get_evaluation_stats,
    get_recent_evaluations, track_time
)
from weather_api import fetch_weather_data

# OpenAI API setup
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ============================================================================
# 1. CONFIGURATION & STYLING
# ============================================================================

def apply_custom_styling():
    """Apply Vajra brand design: Futuristic-elegant, tech-minimal, mythic-geometric"""
    st.markdown("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600;700&family=Orbitron:wght@400;600;700;900&display=swap" rel="stylesheet">
    
    <style>
    /* Import brand fonts - using web-safe alternatives with similar characteristics */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600;700&family=Orbitron:wght@400;600;700;900&display=swap');
    
    /* Brand Color Variables - Diamond Storm Palette */
    :root {
        --electric-violet: #7F5AF0;
        --deep-purple: #6438B7;
        --crystal-blue-gray: #7A88A1;
        --metallic-silver: #DADADA;
        --deep-charcoal: #0F0F0F;
        --white-smoke: #F7F7F7;
    }
    
    /* Base App Styling - Futuristic Minimal */
    .stApp {
        background: linear-gradient(180deg, #FFFFFF 0%, #F7F7F7 100%);
        color: #0F0F0F;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Logo/Wordmark - Geometric Precision */
    h1 {
        font-family: 'Orbitron', 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 2.5rem;
        letter-spacing: -0.02em;
        color: #0F0F0F;
        background: linear-gradient(135deg, #7F5AF0 0%, #6438B7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    
    /* Headers - Architectural Authority */
    h2, h3 {
        font-family: 'Space Grotesk', 'Inter', sans-serif;
        font-weight: 600;
        color: #0F0F0F;
        letter-spacing: -0.01em;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Body Text - Neutral Readability */
    body, p, div, span, .stMarkdown {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        font-weight: 400;
        color: #0F0F0F;
        line-height: 1.6;
    }
    
    /* Numbers and Data - Clean Display */
    [data-testid="stMetricValue"], .metric-value, .number-display {
        font-family: 'Space Grotesk', monospace;
        font-weight: 500;
        color: #7F5AF0;
        font-size: 2rem;
        letter-spacing: -0.02em;
    }
    
    [data-testid="stMetricLabel"] {
        font-family: 'Inter', sans-serif;
        font-weight: 400;
        color: #7A88A1;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Sidebar - Crystalline Minimal */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #FFFFFF 0%, #F7F7F7 100%);
        border-right: 1px solid #DADADA;
    }
    
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        color: #0F0F0F;
    }
    
    /* Input Fields - Low-Poly Precision */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > div,
    .stSlider > div > div {
        border: 1px solid #DADADA;
        border-radius: 4px;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        background-color: #FFFFFF;
    }
    
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > div:focus {
        border-color: #7F5AF0;
        box-shadow: 0 0 0 3px rgba(127, 90, 240, 0.1);
    }
    
    .stSelectbox > div > div:hover {
        border-color: #7F5AF0;
    }
    
    /* Buttons - Divine Modernism */
    .stButton > button {
        background: linear-gradient(135deg, #7F5AF0 0%, #6438B7 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 4px;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 500;
        padding: 0.75rem 1.5rem;
        transition: all 0.2s ease;
        width: 100%;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #6438B7 0%, #7F5AF0 100%) !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(127, 90, 240, 0.3);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Override secondary button styling to use purple */
    .stButton > button[kind="secondary"] {
        background: linear-gradient(135deg, #7F5AF0 0%, #6438B7 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
    }
    
    /* Cards - Sacred Geometry */
    .stCard, .card {
        background-color: #FFFFFF;
        border: 1px solid #DADADA;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(15, 15, 15, 0.05);
    }
    
    /* Recommendation Cards - Crystalline Accent */
    .recommendation-card {
        background: linear-gradient(90deg, #F7F7F7 0%, #FFFFFF 100%);
        border-left: 3px solid #7F5AF0;
        padding: 1rem 1.5rem;
        margin: 0.75rem 0;
        border-radius: 4px;
        font-family: 'Inter', sans-serif;
        transition: all 0.2s ease;
    }
    
    .recommendation-card:hover {
        border-left-color: #6438B7;
        box-shadow: 0 2px 8px rgba(127, 90, 240, 0.1);
    }
    
    /* Tables - Tech-Minimal Precision */
    .analysis-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1.5rem 0;
        font-family: 'Inter', sans-serif;
        background-color: #FFFFFF;
    }
    
    .analysis-table th {
        background: linear-gradient(135deg, #7F5AF0 0%, #6438B7 100%);
        color: #FFFFFF;
        padding: 1rem 1.5rem;
        text-align: left;
        font-weight: 600;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        border: none;
    }
    
    .analysis-table td {
        padding: 1rem 1.5rem;
        border-bottom: 1px solid #DADADA;
        color: #0F0F0F;
        font-family: 'Inter', sans-serif;
    }
    
    .analysis-table tr:hover {
        background-color: #F7F7F7;
    }
    
    .analysis-table tr:last-child td {
        border-bottom: none;
    }
    
    /* Accent Colors */
    .accent-violet {
        color: #7F5AF0;
    }
    
    .accent-purple {
        color: #6438B7;
    }
    
    .accent-blue-gray {
        color: #7A88A1;
    }
    
    /* Plotly Chart Styling */
    .js-plotly-plot {
        background-color: transparent;
    }
    
    /* Info/Warning/Success Messages */
    .stInfo, .stSuccess, .stWarning, .stError {
        border-radius: 4px;
        border-left: 3px solid;
        font-family: 'Inter', sans-serif;
    }
    
    .stInfo {
        border-left-color: #7A88A1;
    }
    
    .stSuccess {
        border-left-color: #7F5AF0;
    }
    
    .stWarning {
        border-left-color: #6438B7;
    }
    
    .stError {
        border-left-color: #0F0F0F;
    }
    
    /* Selectbox Styling */
    .stSelectbox label {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        color: #0F0F0F;
        font-size: 0.875rem;
        margin-bottom: 0.5rem;
    }
    
    /* Text Input Styling */
    .stTextInput label {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        color: #0F0F0F;
        font-size: 0.875rem;
        margin-bottom: 0.5rem;
    }
    
    /* Divider Styling */
    hr {
        border: none;
        border-top: 1px solid #DADADA;
        margin: 2rem 0;
    }
    
    /* Caption Styling */
    .stCaption {
        font-family: 'Inter', sans-serif;
        font-weight: 400;
        color: #7A88A1;
        font-size: 0.75rem;
    }
    
    /* Subheader Styling */
    .stSubheader {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        color: #0F0F0F;
        letter-spacing: -0.01em;
    }
    
    /* Tab Styling - Brand Design */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        margin-bottom: 1.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 500;
        color: #7A88A1;
        padding: 0.75rem 1.5rem;
        border-radius: 4px 4px 0 0;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #7F5AF0;
        background-color: rgba(127, 90, 240, 0.05);
    }
    
    .stTabs [aria-selected="true"] {
        color: #7F5AF0;
        background-color: rgba(127, 90, 240, 0.1);
        border-bottom: 2px solid #7F5AF0;
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        padding: 1.5rem 0;
    }
    
    /* Custom Loading Animation - Clean Modern Minimal */
    @keyframes vajra-pulse {
        0%, 100% {
            opacity: 0.4;
            transform: scale(0.95);
        }
        50% {
            opacity: 1;
            transform: scale(1.05);
        }
    }
    
    @keyframes vajra-gradient {
        0% {
            background-position: 0% 50%;
        }
        50% {
            background-position: 100% 50%;
        }
        100% {
            background-position: 0% 50%;
        }
    }
    
    @keyframes vajra-glow {
        0%, 100% {
            box-shadow: 0 0 20px rgba(127, 90, 240, 0.3);
        }
        50% {
            box-shadow: 0 0 40px rgba(127, 90, 240, 0.6), 0 0 60px rgba(100, 56, 183, 0.4);
        }
    }
    
    .vajra-loader {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 4rem 2rem;
        background: linear-gradient(135deg, rgba(127, 90, 240, 0.08) 0%, rgba(100, 56, 183, 0.08) 100%);
        border-radius: 20px;
        border: 2px solid rgba(127, 90, 240, 0.3);
        margin: 2rem 0;
        font-family: 'Space Grotesk', sans-serif;
        animation: vajra-glow 2s ease-in-out infinite;
    }
    
    .vajra-loader-spinner {
        width: 80px;
        height: 80px;
        position: relative;
        margin-bottom: 2rem;
    }
    
    .vajra-loader-spinner::before,
    .vajra-loader-spinner::after {
        content: '';
        position: absolute;
        border-radius: 50%;
        border: 4px solid transparent;
    }
    
    .vajra-loader-spinner::before {
        width: 80px;
        height: 80px;
        border-top: 4px solid #7F5AF0;
        border-right: 4px solid #7F5AF0;
        border-bottom: 4px solid transparent;
        border-left: 4px solid transparent;
        animation: vajra-spin 1.2s cubic-bezier(0.68, -0.55, 0.27, 1.55) infinite;
    }
    
    .vajra-loader-spinner::after {
        width: 50px;
        height: 50px;
        top: 15px;
        left: 15px;
        border-bottom: 4px solid #6438B7;
        border-left: 4px solid #6438B7;
        border-top: 4px solid transparent;
        border-right: 4px solid transparent;
        animation: vajra-spin 0.9s cubic-bezier(0.68, -0.55, 0.27, 1.55) infinite reverse;
    }
    
    @keyframes vajra-spin {
        0% {
            transform: rotate(0deg);
        }
        100% {
            transform: rotate(360deg);
        }
    }
    
    .vajra-loader-text {
        font-size: 1.1rem;
        font-weight: 600;
        background: linear-gradient(135deg, #7F5AF0 0%, #6438B7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: 0.03em;
        animation: vajra-pulse 2s ease-in-out infinite;
    }
    
    .vajra-loader-dots {
        display: inline-block;
        width: 30px;
        text-align: left;
    }
    
    .vajra-loader-dots::after {
        content: '...';
        animation: vajra-dots 1.2s steps(4, end) infinite;
    }
    
    @keyframes vajra-dots {
        0%, 20% {
            content: '.';
        }
        40% {
            content: '..';
        }
        60%, 100% {
            content: '...';
        }
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# 2. SIDEBAR CONTROLS
# ============================================================================

def render_sidebar() -> Dict:
    """Render sidebar with all user inputs"""
    st.sidebar.title("Vajra Controls")
    st.sidebar.markdown("---")
    
    retailer = st.sidebar.text_input(
        "Retailer Name",
        value="",
        placeholder="Enter retailer name",
        help="Examples: Walmart, Target, Amazon Fresh"
    )
    
    area = st.sidebar.text_input(
        "Area / City / Pin Code",
        value="",
        placeholder="Enter area, city, or pin code",
        help="Examples: New York, Mumbai, 10001"
    )
    
    st.sidebar.markdown("---")
    
    # Evaluate button with brand styling (using secondary to avoid red primary color)
    evaluate_button = st.sidebar.button(
        "Evaluate Forecast",
        type="secondary",
        use_container_width=True,
        help="Generate forecasts and analysis"
    )
    
    # Fixed values - not shown to user
    seed = 42
    timeline_months = 12
    reveal_month = 12
    use_mocked_llm = True
    
    return {
        "retailer": retailer,
        "area": area,
        "seed": seed,
        "timeline_months": timeline_months,
        "reveal_month": reveal_month,
        "use_mocked_llm": use_mocked_llm,
        "evaluate_button": evaluate_button
    }

# ============================================================================
# 3. LLM API FUNCTIONS
# ============================================================================

def get_openai_client() -> Optional[object]:
    """Initialize OpenAI client with API key from secrets or environment"""
    if not OPENAI_AVAILABLE:
        return None
    
    api_key = None
    # Try Streamlit secrets first
    try:
        # Use proper Streamlit secrets access - try multiple methods
        if hasattr(st.secrets, "OPENAI_API_KEY"):
            api_key = st.secrets.OPENAI_API_KEY
            print(f"✅ Found API key via st.secrets.OPENAI_API_KEY")
        elif "OPENAI_API_KEY" in st.secrets:
            api_key = st.secrets["OPENAI_API_KEY"]
            print(f"✅ Found API key via st.secrets['OPENAI_API_KEY']")
        elif hasattr(st.secrets, "get"):
            api_key = st.secrets.get("OPENAI_API_KEY")
            if api_key:
                print(f"✅ Found API key via st.secrets.get()")
    except AttributeError as e:
        # st.secrets might not be available
        print(f"⚠️ AttributeError accessing secrets: {e}")
        pass
    except Exception as e:
        # Log the error for debugging
        print(f"⚠️ Error accessing Streamlit secrets: {e}")
        pass
    
    # Fallback to environment variable
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            print(f"✅ Found API key in environment variable")
    
    if not api_key:
        print("❌ No API key found in secrets or environment")
        return None
    
    # Validate API key format
    if not api_key.startswith("sk-"):
        print(f"⚠️ API key doesn't start with 'sk-', might be invalid")
    
    print(f"✅ API key found: {api_key[:20]}...")
    
    try:
        return OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {str(e)}")
        return None

def llm_call(prompt: str, system_prompt: str = None, model: str = "gpt-4o-mini", max_tokens: int = 1000, use_cache: bool = True) -> Optional[str]:
    """
    Make an LLM API call with error handling and fallback.
    Uses GPT-4o-mini by default, but can use gpt-5-nano if available.
    """
    client = get_openai_client()
    if not client:
        return None
    
    # Create cache key based on prompt content for caching
    if use_cache:
        cache_key = f"{model}_{hash(prompt + str(system_prompt))}"
        if hasattr(st.session_state, 'llm_cache') and cache_key in st.session_state.llm_cache:
            return st.session_state.llm_cache[cache_key]
    
    # Initialize cache if needed
    if not hasattr(st.session_state, 'llm_cache'):
        st.session_state.llm_cache = {}
    
    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7
        )
        
        result = response.choices[0].message.content.strip()
        
        # Cache the result
        if use_cache:
            st.session_state.llm_cache[cache_key] = result
        
        return result
    except Exception as e:
        # Show error for debugging - log full error
        error_msg = str(e)
        error_type = type(e).__name__
        
        # Log to console for debugging
        import traceback
        print(f"LLM API Error ({error_type}): {error_msg}")
        print(traceback.format_exc())
        
        # Show user-friendly error (minimal emojis)
        if "billing" in error_msg.lower() or "account is not active" in error_msg.lower():
            st.error(f"**Billing Issue**: Your OpenAI account is not active. Please check your billing details at https://platform.openai.com/account/billing")
        elif "quota" in error_msg.lower() or "insufficient" in error_msg.lower():
            st.error(f"**Quota Issue**: You've exceeded your current quota or have no credits. Please add credits at https://platform.openai.com/account/billing")
        elif "api_key" in error_msg.lower() or "authentication" in error_msg.lower() or "invalid" in error_msg.lower():
            st.error(f"**API Key Error**: {error_msg}")
        elif "rate limit" in error_msg.lower():
            st.warning(f"Rate limit exceeded. Please try again in a moment.")
        elif "model" in error_msg.lower() and "not found" in error_msg.lower():
            st.error(f"**Model Error**: {error_msg}")
        else:
            st.error(f"**LLM API call failed**: {error_msg[:200]}")
        return None

def llm_top_skus(retailer: str) -> List[Dict]:
    """
    Generate top 5 SKUs for a retailer using LLM.
    Returns realistic SKUs based on retailer type with proper categorization.
    """
    system_prompt = """You are an expert in retail operations and food delivery. 
Generate highly specific, realistic SKUs (Stock Keeping Units) for food delivery operations. 
Each SKU must have: name, category, and flags for perishable, frozen, and organic attributes.
Categories include: produce, protein, dairy, bakery, pantry, household, shelf-stable, seafood.
Return ONLY valid JSON array format, no other text."""
    
    user_prompt = f"""Generate exactly 5 top-selling, highly specific SKUs for {retailer} food delivery operations.

For each SKU, provide:
- name: Specific product name with brand, size, or variant details (e.g., "365 Organic Whole Milk 1 Gallon", "Perdue Premium Chicken Breast 5lb Family Pack", "Dave's Killer Bread 21 Whole Grains 27oz")
- category: One of: produce, protein, dairy, bakery, pantry, household, shelf-stable, seafood
- perishable: true/false
- frozen: true/false  
- organic: true/false

Make SKUs highly specific with:
- Brand names when relevant
- Exact sizes/quantities
- Product variants (organic, premium, family pack, etc.)
- Specific product lines

Consider {retailer}'s typical product mix, customer base, and regional preferences. Return as JSON array only, no markdown or explanation.

Example format:
[
  {{"name": "365 Organic Whole Milk 1 Gallon", "category": "dairy", "perishable": true, "frozen": false, "organic": true}},
  {{"name": "Perdue Premium Chicken Breast 5lb Family Pack", "category": "protein", "perishable": true, "frozen": false, "organic": false}}
]"""
    
    # Try gpt-5-nano first, fallback to gpt-4o-mini
    response = llm_call(user_prompt, system_prompt, model="gpt-4o-mini", max_tokens=800)
    
    if response:
        try:
            # Try to extract JSON from response (handle markdown code blocks)
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            
            skus = json.loads(response)
            # Validate and ensure we have exactly 5 SKUs
            if isinstance(skus, list) and len(skus) >= 5:
                # Ensure all required fields
                validated_skus = []
                for sku in skus[:5]:
                    if all(key in sku for key in ["name", "category", "perishable", "frozen", "organic"]):
                        validated_skus.append({
                            "name": sku["name"],
                            "category": sku["category"],
                            "perishable": bool(sku["perishable"]),
                            "frozen": bool(sku["frozen"]),
                            "organic": bool(sku["organic"])
                        })
                if len(validated_skus) == 5:
                    return validated_skus
        except json.JSONDecodeError:
            pass
    
    # Fallback to mock data
    if retailer.strip():
        st.info(f"Using fallback SKU data for {retailer}. LLM call failed or API key not configured.")
    return mock_llm_top_skus(retailer)

def llm_anomaly(area: str) -> Dict:
    """
    Generate weather anomaly prediction for a location using real-time weather data and LLM.
    Uses OpenWeatherMap API to fetch current weather conditions and detect anomalies.
    """
    # Fetch real-time weather data
    weather_data = None
    try:
        weather_data = fetch_weather_data(area)
    except Exception as e:
        print(f"Error fetching weather data: {e}")
    
    system_prompt = """You are a weather and disruption intelligence expert for food delivery operations.
Analyze real-time weather data to predict specific weather anomalies that could impact food delivery.
Use the provided weather data to make accurate, data-driven predictions.
CRITICAL: Use SPECIFIC, DESCRIPTIVE weather event names (e.g., "Torrential Monsoon Downpour", "Severe Wind Gusts", "Coastal Storm Surge", "Blizzard Conditions", "Flash Flood Event", "Dust Storm", "Heat Wave", "Freezing Rain", "Hailstorm", "Tropical Cyclone") - NOT generic terms like "Seasonal Weather Event" or "Weather Anomaly".
Make the event name region-appropriate and use vivid, descriptive vocabulary that clearly indicates the type and severity of the weather event.
Include specific detection methods and detailed impact descriptions based on actual weather conditions.
Return ONLY valid JSON format, no other text."""
    
    # Build user prompt with real weather data
    if weather_data and weather_data.get("anomaly", {}).get("detected"):
        # Anomaly detected - use real data
        current = weather_data["current"]
        anomaly_info = weather_data["anomaly"]
        location_info = weather_data["location"]
        
        # Calculate current month (for chart positioning)
        current_month = datetime.now().month
        # Ensure it's in Sept-Nov range for chart visibility, or use current month if in range
        if current_month < 9:
            target_month = 9  # September
        elif current_month > 11:
            target_month = 11  # November
        else:
            target_month = current_month
        
        # Format weather details
        wind_speed_kmh = current.get("wind_speed", 0) * 3.6
        wind_gust_kmh = (current.get("wind_gust", 0) or current.get("wind_speed", 0)) * 3.6
        rain_mm = current.get("rain_1h", 0)
        temp = current.get("temp", 20)
        visibility_km = current.get("visibility", 10)
        weather_desc = current.get("weather_description", "")
        
        # Get current date and calculate timeline
        current_date = datetime.now()
        date_str = current_date.strftime("%B %d")
        next_date_str = (current_date + timedelta(days=1)).strftime("%B %d")
        
        user_prompt = f"""Analyze the real-time weather data for {area} and create a detailed, personalized anomaly alert.

REAL-TIME WEATHER DATA:
- Current Temperature: {temp:.1f}°C
- Wind Speed: {wind_speed_kmh:.1f} km/h (gusts up to {wind_gust_kmh:.1f} km/h)
- Rainfall: {rain_mm:.1f} mm/hour
- Visibility: {visibility_km:.1f} km
- Weather Condition: {weather_desc}
- Anomaly Detected: {anomaly_info.get('type', 'Unknown')} ({anomaly_info.get('severity', 'unknown')} severity)
- Detection Confidence: {anomaly_info.get('confidence', 0) * 100:.0f}%
- Location: {location_info.get('name', area)}, {location_info.get('state', '')}, {location_info.get('country', '')}

CRITICAL: Make this feel personalized and delightful to someone from {area}:
- Use specific neighborhoods, streets, landmarks, or local areas within {area}
- Reference local infrastructure (bridges, highways, specific routes known in {area})
- Include ZIP codes or postal codes specific to {area}
- Use local terminology and area names that someone from {area} would recognize
- Base ALL details on the ACTUAL weather data provided above

Return JSON with:
- type: SPECIFIC, DESCRIPTIVE weather event name with timeline. Use vivid, region-appropriate terminology:
  * For rain: "Torrential Monsoon Downpour", "Heavy Rainfall Deluge", "Flash Flood Event", "Persistent Drizzle"
  * For wind: "Severe Wind Gusts", "Gale Force Winds", "Dust Storm", "Tornado Watch", "Cyclonic Winds"
  * For snow: "Blizzard Conditions", "Heavy Snowfall", "Ice Storm", "Freezing Rain Event"
  * For heat: "Heat Wave", "Scorching Temperatures", "Extreme Heat Advisory"
  * For coastal: "Coastal Storm Surge", "High Tide Warning", "Tsunami Alert" (if applicable)
  * For storms: "Thunderstorm System", "Severe Thunderstorm", "Tropical Cyclone", "Hurricane Remnants"
  Examples: "Torrential Monsoon Downpour - {date_str} to {next_date_str}", "Severe Wind Gusts - {date_str}", "Blizzard Conditions - {date_str}"
  DO NOT use generic terms like "Seasonal Weather Event", "Weather Anomaly", or "Weather System"
- description: EXACTLY 3 complete sentences separated by periods. Each sentence is one bullet point:
  
  1. SOURCE & TIMELINE: Start with "Detected via [source of analysis]" (e.g., "OpenWeatherMap real-time monitoring", "NOAA satellite imagery", "Weather station network analysis"), include the TENTATIVE TIMELINE (dates/duration when anomaly will occur), and end with "with [EXACT confidence level]% confidence" (CRITICAL: use EXACTLY {anomaly_info.get('confidence', 0) * 100:.0f}% - do not round or change this number)
     Example: "Detected via OpenWeatherMap real-time monitoring and weather station network analysis, {date_str} to {next_date_str} event expected, with {anomaly_info.get('confidence', 0) * 100:.0f}% confidence"
  
  2. TECHNICAL DETAILS & AFFECTED AREAS: Include ACTUAL weather values (wind speed: {wind_speed_kmh:.0f} km/h, gusts: {wind_gust_kmh:.0f} km/h, rainfall: {rain_mm:.0f} mm/hour, visibility: {visibility_km:.1f} km) and SPECIFIC areas that could be affected - mention actual streets, neighborhoods, ZIP codes, landmarks, or routes in {area}. Explain HOW these areas will be affected (e.g., "flooding on [specific street]", "wind damage in [neighborhood]", "reduced visibility on [highway/route]")
     Example: "{date_str} event bringing {wind_gust_kmh:.0f} km/h wind gusts and {rain_mm:.0f} mm/hour rainfall, particularly affecting [specific streets/neighborhoods in {area}], with flooding anticipated on [specific routes] and reduced visibility on [specific highways]"
  
  3. IMPACT & WHY: Explain HOW and WHY this affects delivery operations in clear, understandable terms. Reference the specific technical details mentioned above and explain the delivery impact (delays, route closures, warehouse restrictions, etc.). CRITICAL: Include a specific percentage (between 8-25%) for deliveries affected or capacity reduction (e.g., "causing 12% capacity reduction", "affecting 15% of deliveries", "resulting in 18% delivery delays")
     Example: "Delivery delays of [X] hours expected in affected areas due to [specific reason from technical details], with warehouse access restrictions on [specific routes] causing [X]% capacity reduction during peak hours"
  
  Use clear, accurate language without excessive jargon. Use the REAL weather values provided. Make it feel personalized and delightful to {area}.
- base_impact_pct: Demand impact based on severity (-20 to -35 for mild, -30 to -45 for moderate, -40 to -55 for severe)
- base_month_index: {target_month} (current month or adjusted for chart visibility)
- direction: "cliff" for drops, "spike" for increases

Return ONLY JSON, no markdown."""
    else:
        # No anomaly detected or weather data unavailable - use forecast or historical pattern
        if weather_data:
            current = weather_data["current"]
            location_info = weather_data["location"]
            forecast = weather_data.get("forecast", [])
            
            # Check forecast for upcoming severe weather
            upcoming_severe = False
            forecast_details = ""
            severe_forecast_item = None
            if forecast:
                for fc in forecast[:8]:  # Next 24 hours
                    fc_wind = (fc.get("wind_speed", 0) or fc.get("wind_gust", 0)) * 3.6
                    fc_rain = fc.get("rain_3h", 0) / 3
                    if fc_wind > 54 or fc_rain > 10:
                        upcoming_severe = True
                        severe_forecast_item = fc
                        forecast_date = datetime.fromtimestamp(fc.get("dt", 0)).strftime("%B %d")
                        forecast_details = f"Forecast shows {fc_wind:.0f} km/h winds and {fc_rain:.0f} mm/h rainfall on {forecast_date}"
                        break
            
            current_month = datetime.now().month
            if current_month < 9:
                target_month = 9
            elif current_month > 11:
                target_month = 11
            else:
                target_month = current_month
            
            if upcoming_severe:
                forecast_date_obj = datetime.fromtimestamp(severe_forecast_item.get("dt", 0)) if severe_forecast_item else datetime.now()
                forecast_date_str = forecast_date_obj.strftime("%B %d")
                forecast_next_date_str = (forecast_date_obj + timedelta(days=1)).strftime("%B %d")
                
                user_prompt = f"""Analyze weather forecast for {area} and predict upcoming anomaly.

CURRENT WEATHER: {current.get('weather_description', 'Clear')}, {current.get('temp', 20):.1f}°C
FORECAST: {forecast_details}

Return JSON with:
- type: SPECIFIC, DESCRIPTIVE weather event name with timeline. Use vivid, region-appropriate terminology:
  * For rain: "Torrential Monsoon Downpour", "Heavy Rainfall Deluge", "Flash Flood Event"
  * For wind: "Severe Wind Gusts", "Gale Force Winds", "Dust Storm", "Cyclonic Winds"
  * For snow: "Blizzard Conditions", "Heavy Snowfall", "Ice Storm"
  * For heat: "Heat Wave", "Scorching Temperatures"
  * For storms: "Thunderstorm System", "Severe Thunderstorm", "Tropical Cyclone"
  Examples: "Torrential Monsoon Downpour - {forecast_date_str} to {forecast_next_date_str}", "Severe Wind Gusts - {forecast_date_str}"
  DO NOT use generic terms like "Severe Weather System" or "Weather Anomaly"
- description: EXACTLY 3 complete sentences separated by periods:
  1. SOURCE & TIMELINE: "Detected via [source]" (e.g., "OpenWeatherMap 5-day forecast analysis", "NOAA extended forecast"), include TENTATIVE TIMELINE ({forecast_date_str} to {forecast_next_date_str}), and confidence level (70-80%)
  2. TECHNICAL DETAILS & AFFECTED AREAS: Include forecast values from {forecast_details} and SPECIFIC streets, neighborhoods, ZIP codes, landmarks in {area} that could be affected. Explain HOW (flooding, wind damage, visibility issues)
  3. IMPACT & WHY: Explain HOW and WHY this affects delivery operations based on the technical details
  
Make it personalized to {area} with specific neighborhoods and ZIP codes.
- base_impact_pct: -30 to -45
- base_month_index: {target_month}
- direction: "cliff"

Return ONLY JSON, no markdown."""
            else:
                # No severe weather - use historical pattern for Sept-Nov
                # Use a typical date in the target month
                if target_month == 9:
                    anomaly_date = f"September {15 + (datetime.now().day % 15)}"
                elif target_month == 10:
                    anomaly_date = f"October {15 + (datetime.now().day % 15)}"
                else:
                    anomaly_date = f"November {15 + (datetime.now().day % 15)}"
                
                user_prompt = f"""Based on historical patterns for {area}, predict a likely weather anomaly for September-November period.

Location: {location_info.get('name', area)}, {location_info.get('state', '')}, {location_info.get('country', '')}
Current conditions: {current.get('weather_description', 'Clear')}, {current.get('temp', 20):.1f}°C

Return JSON with:
- type: SPECIFIC, DESCRIPTIVE weather event name with timeline. Use vivid, region-appropriate terminology based on typical weather patterns for this location:
  * For monsoon regions: "Torrential Monsoon Downpour", "Heavy Monsoon Rains", "Monsoon Deluge"
  * For temperate regions: "Severe Wind Gusts", "Heavy Rainfall", "Thunderstorm System"
  * For cold regions: "Blizzard Conditions", "Heavy Snowfall", "Ice Storm", "Freezing Rain"
  * For hot regions: "Heat Wave", "Scorching Temperatures", "Dust Storm"
  * For coastal regions: "Coastal Storm Surge", "High Tide Warning", "Coastal Flooding"
  Examples: "Torrential Monsoon Downpour - {anomaly_date}", "Severe Wind Gusts - {anomaly_date}", "Blizzard Conditions - {anomaly_date}"
  DO NOT use generic terms like "Monsoon System", "Seasonal Storm", "Seasonal Weather Event", or "Weather Anomaly"
- description: EXACTLY 3 complete sentences separated by periods:
  1. SOURCE & TIMELINE: "Detected via [source]" (e.g., "historical weather pattern analysis", "seasonal climate models"), include TENTATIVE TIMELINE ({anomaly_date}), and confidence level (60-75%)
  2. TECHNICAL DETAILS & AFFECTED AREAS: Include realistic weather values based on typical patterns for this location and SPECIFIC streets, neighborhoods, ZIP codes, landmarks in {area} that could be affected. Explain HOW (flooding, wind damage, visibility issues)
  3. IMPACT & WHY: Explain HOW and WHY this affects delivery operations based on the technical details
  
Make it personalized to {area} with specific neighborhoods and ZIP codes.
- base_impact_pct: -30 to -45
- base_month_index: {target_month}
- direction: "cliff"

Return ONLY JSON, no markdown."""
        else:
            # Fallback: no weather data available
            current_month = datetime.now().month
            if current_month < 9:
                target_month = 9
                anomaly_date = f"September {15 + (datetime.now().day % 15)}"
            elif current_month > 11:
                target_month = 11
                anomaly_date = f"November {15 + (datetime.now().day % 15)}"
            else:
                target_month = current_month
                if target_month == 9:
                    anomaly_date = f"September {15 + (datetime.now().day % 15)}"
                elif target_month == 10:
                    anomaly_date = f"October {15 + (datetime.now().day % 15)}"
                else:
                    anomaly_date = f"November {15 + (datetime.now().day % 15)}"
            
            user_prompt = f"""Predict a realistic weather anomaly for {area} based on typical weather patterns.

Weather data unavailable. Use historical patterns for this location.

Return JSON with:
- type: SPECIFIC, DESCRIPTIVE weather event name with timeline. Use vivid, region-appropriate terminology based on typical weather patterns for this location:
  * For monsoon regions: "Torrential Monsoon Downpour", "Heavy Monsoon Rains", "Monsoon Deluge"
  * For temperate regions: "Severe Wind Gusts", "Heavy Rainfall", "Thunderstorm System"
  * For cold regions: "Blizzard Conditions", "Heavy Snowfall", "Ice Storm", "Freezing Rain"
  * For hot regions: "Heat Wave", "Scorching Temperatures", "Dust Storm"
  * For coastal regions: "Coastal Storm Surge", "High Tide Warning", "Coastal Flooding"
  Examples: "Torrential Monsoon Downpour - {anomaly_date}", "Severe Wind Gusts - {anomaly_date}", "Blizzard Conditions - {anomaly_date}"
  DO NOT use generic terms like "Seasonal Weather Event", "Weather Anomaly", or "Weather System"
- description: EXACTLY 3 complete sentences separated by periods:
  1. SOURCE & TIMELINE: "Detected via [source]" (e.g., "historical weather pattern analysis"), include TENTATIVE TIMELINE ({anomaly_date}), and confidence level (55-70%)
  2. TECHNICAL DETAILS & AFFECTED AREAS: Include realistic weather values and SPECIFIC streets, neighborhoods, ZIP codes, landmarks in {area} that could be affected. Explain HOW (flooding, wind damage, visibility issues)
  3. IMPACT & WHY: Explain HOW and WHY this affects delivery operations based on the technical details
  
Make it personalized to {area} with specific neighborhoods and ZIP codes.
- base_impact_pct: -30 to -45
- base_month_index: {target_month}
- direction: "cliff"

Return ONLY JSON, no markdown."""
    
    response = llm_call(user_prompt, system_prompt, model="gpt-4o-mini", max_tokens=500)
    
    if response:
        try:
            # Extract JSON from response
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            
            anomaly = json.loads(response)
            
            # Validate and ensure month is 9-11
            if all(key in anomaly for key in ["type", "description", "base_impact_pct", "base_month_index", "direction"]):
                # Ensure description is a string, not a list
                if isinstance(anomaly["description"], list):
                    anomaly["description"] = ". ".join(anomaly["description"])
                elif not isinstance(anomaly["description"], str):
                    anomaly["description"] = str(anomaly["description"])
                
                # Preserve confidence and severity from weather API if available
                if weather_data and weather_data.get("anomaly", {}).get("detected"):
                    anomaly["confidence"] = weather_data["anomaly"].get("confidence", 0.75)
                    anomaly["severity"] = weather_data["anomaly"].get("severity", "medium")
                    anomaly["anomaly"] = weather_data["anomaly"]  # Preserve full anomaly data
                
                # Ensure month is in Sept-Nov range
                month = int(anomaly["base_month_index"])
                if month < 9 or month > 11:
                    # Adjust to closest valid month
                    month = 10  # Default to October
                anomaly["base_month_index"] = month
                anomaly["base_impact_pct"] = float(anomaly["base_impact_pct"])
                return anomaly
        except (json.JSONDecodeError, ValueError, KeyError):
            pass
    
    # Fallback to mock data
    if area.strip():
        st.info(f"Using fallback anomaly data for {area}. LLM call failed or API key not configured.")
    return mock_llm_anomaly(area)

def llm_recommendations(sku: Dict, anomaly: Dict) -> List[str]:
    """
    Generate 3 actionable recommendations based on SKU and anomaly using LLM.
    """
    system_prompt = """You are an operations expert for food delivery companies.
Generate extremely specific, actionable recommendations for managing inventory and operations during weather anomalies.
Recommendations must be highly detailed, measurable, and directly address the anomaly's impact on the specific SKU type.
Include specific quantities, timeframes, locations, and metrics.
Return ONLY a JSON array of exactly 3 recommendation strings, no other text."""
    
    sku_details = f"""
SKU: {sku.get('name', 'Unknown')}
Category: {sku.get('category', 'Unknown')}
Perishable: {sku.get('perishable', False)}
Frozen: {sku.get('frozen', False)}
Organic: {sku.get('organic', False)}
"""
    
    anomaly_details = f"""
Anomaly Type: {anomaly.get('type', 'Unknown')}
Description: {anomaly.get('description', 'Unknown')}
Impact: {anomaly.get('impact_pct', 0)}%
Direction: {anomaly.get('direction', 'cliff')}
"""
    
    user_prompt = f"""Provide exactly 3 top recommendations to protect against the cited anomaly.

{sku_details}

{anomaly_details}

CRITICAL: Make recommendations feel personalized and delightful:
- Reference specific neighborhoods, ZIP codes, or areas mentioned in the anomaly description
- Mention local landmarks, routes, or infrastructure from the anomaly description
- Use exact location names, ZIP codes, or area references from the anomaly description
- Ensure recommendations are logically consistent with the anomaly's impact and technical details

Each recommendation must:
- Be one complete, concise sentence
- Start with an action verb (e.g., "Increase", "Redirect", "Pre-position")
- Be specific with numbers, percentages, timeframes, and exact locations from the anomaly
- Address SKU characteristics (perishable, frozen, organic, etc.) when relevant
- Be logically consistent with the anomaly's impact (e.g., if anomaly mentions 4-6 hour delays, recommendations should address that)
- Feel actionable and practical

Return as JSON array of exactly 3 recommendation strings. Each string is one complete sentence.

Example format:
["Increase inventory by 25% in ZIP codes 380015 and 380051 by September 20 to account for 4-6 hour delivery delays", "Redirect deliveries from SG Highway to alternative routes through Satellite and Bopal areas during September 22-24", "Pre-position perishable items in temperature-controlled storage facilities near Vastrapur to minimize spoilage during warehouse access restrictions"]

Return ONLY JSON array, no markdown."""
    
    response = llm_call(user_prompt, system_prompt, model="gpt-4o-mini", max_tokens=400)
    
    if response:
        try:
            # Extract JSON from response
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            
            recommendations = json.loads(response)
            if isinstance(recommendations, list):
                # Ensure we have exactly 3 recommendations
                if len(recommendations) >= 3:
                    return recommendations[:3]
                elif len(recommendations) > 0:
                    # If we have fewer than 3, pad with generic ones
                    while len(recommendations) < 3:
                        recommendations.append("Review operational capacity and adjust delivery schedules accordingly.")
                    return recommendations[:3]
        except json.JSONDecodeError:
            pass
    
    # Fallback to mock data
    return mock_llm_recommendations(sku, anomaly)

# ============================================================================
# 4. FALLBACK MOCK FUNCTIONS (used when LLM unavailable)
# ============================================================================

@st.cache_data
def mock_llm_top_skus(retailer: str) -> List[Dict]:
    """Fallback function when LLM is unavailable - returns generic SKUs"""
    # Generic fallback SKUs without retailer-specific logic
    return [
        {"name": "Organic Whole Milk 1 Gallon", "category": "dairy", "perishable": True, "frozen": False, "organic": True},
        {"name": "Premium Chicken Breast 5lb Family Pack", "category": "protein", "perishable": True, "frozen": False, "organic": False},
        {"name": "Fresh Organic Kale 1lb Bundle", "category": "produce", "perishable": True, "frozen": False, "organic": True},
        {"name": "Artisan Whole Grain Bread 27oz", "category": "bakery", "perishable": True, "frozen": False, "organic": False},
        {"name": "Large Grade A Eggs 18 Count", "category": "dairy", "perishable": True, "frozen": False, "organic": False},
    ]

@st.cache_data
def mock_llm_anomaly(area: str) -> Dict:
    """Fallback function when LLM is unavailable - returns generic weather anomaly"""
    # Generic fallback anomaly without location-specific logic
    np.random.seed(hash(area) % 1000)
    anomaly_month = np.random.choice([9, 10, 11])
    
    return {
        "type": "Severe Weather System - October 15-18",
        "description": f"Detected via NOAA weather monitoring and historical pattern analysis (82% confidence). Severe weather system affecting {area} on October 15-18 with 5-7 inches of precipitation and 40-50 mph winds. Specific impacts: Route closures in downtown and suburban zones, 3-4 hour delivery delays, warehouse access restrictions, supply chain disruptions affecting perishable goods transport. Expected to reduce delivery capacity by 38% during peak hours.",
        "base_impact_pct": -38.0,
        "base_month_index": anomaly_month,
        "direction": "cliff"
    }

def generate_sku_anomaly_variations(base_anomaly: Dict, num_skus: int, seed: int) -> List[Dict]:
    """Creates per-SKU anomaly variations with impact differences, keeping timing in Sept-Nov"""
    np.random.seed(seed)
    variations = []
    
    for i in range(num_skus):
        # Impact variation: ±5-10% from base
        impact_variation = np.random.uniform(-10, 10)
        impact_pct = base_anomaly["base_impact_pct"] + impact_variation
        
        # Timing variation: ±1 month from base, but keep within Sept-Nov (9-11)
        timing_variation = np.random.randint(-1, 2)
        month_index = base_anomaly["base_month_index"] + timing_variation
        # Ensure it stays in Sept-Nov range
        month_index = max(9, min(11, month_index))
        
        variation = {
            "type": base_anomaly["type"],
            "description": base_anomaly["description"],
            "impact_pct": impact_pct,
            "month_index": month_index,
            "direction": base_anomaly.get("direction", "cliff")
        }
        variations.append(variation)
    
    return variations

def mock_llm_recommendations(sku: Dict, anomaly: Dict) -> List[str]:
    """Fallback function when LLM is unavailable - generates exactly 3 context-aware recommendations"""
    recommendations = []
    anomaly_type = anomaly.get("type", "").lower()
    is_perishable = sku.get("perishable", False)
    is_frozen = sku.get("frozen", False)
    
    # Context-aware recommendations - always return exactly 3
    if "heat" in anomaly_type:
        if is_perishable:
            recommendations.append("Pre-cool delivery vans to maintain cold chain integrity")
            recommendations.append("Reduce exposure time for temperature-sensitive SKUs")
            recommendations.append("Increase cold chain capacity by 15% during peak heat hours")
        else:
            recommendations.append("Monitor warehouse temperature controls")
            recommendations.append("Prioritize early morning deliveries")
            recommendations.append("Increase buffer stock for high-velocity items")
    
    elif "snow" in anomaly_type or "storm" in anomaly_type or "rain" in anomaly_type:
        if is_frozen:
            recommendations.append("Pre-position inventory in high-demand ZIPs before storm")
            recommendations.append("Increase buffer stock by 20% for frozen essentials")
            recommendations.append("Throttle low-priority routes; prioritize high-velocity SKUs")
        else:
            recommendations.append("Increase buffer stock for essentials by 15%")
            recommendations.append("Pre-position couriers in strategic locations")
            recommendations.append("Activate surge pricing for high-demand areas")
    
    elif "flood" in anomaly_type:
        recommendations.append("Reroute deliveries away from flood-prone areas")
        recommendations.append("Increase buffer stock by 10% for all SKUs")
        recommendations.append("Prioritize high-velocity perishables; delay non-essentials")
    
    elif "hurricane" in anomaly_type:
        recommendations.append("Pre-position emergency inventory 48 hours before landfall")
        recommendations.append("Increase buffer stock for essentials by 25%")
        recommendations.append("Throttle low-priority SKUs; prioritize high-velocity items")
    
    else:  # Supply chain delay
        recommendations.append("Increase buffer stock by 10% for affected SKUs")
        recommendations.append("Identify alternative suppliers for critical items")
        recommendations.append("Prioritize high-velocity SKUs; delay low-priority restocking")
    
    # Ensure we have exactly 3 recommendations
    while len(recommendations) < 3:
        recommendations.append("Monitor demand patterns and adjust inventory levels accordingly")
    
    return recommendations[:3]

# ============================================================================
# 4. DATA GENERATION
# ============================================================================

@st.cache_data
def generate_demand_series(sku: Dict, anomaly: Dict, timeline_months: int, seed: int) -> pd.DataFrame:
    """Generates time series with seasonality, noise, and anomaly injection"""
    np.random.seed(seed + hash(sku["name"]) % 1000)
    
    months = np.arange(1, timeline_months + 1)
    base_demand = np.random.uniform(1000, 5000)
    
    # Light seasonality (sine wave with holiday peaks)
    seasonality = 1.0 + 0.15 * np.sin(2 * np.pi * (months - 3) / 12)  # Peak in month 3 (March)
    holiday_boost = np.where((months == 11) | (months == 12), 1.2, 1.0)  # Nov/Dec boost
    
    # Random walk noise
    noise = np.cumsum(np.random.normal(0, 0.05, timeline_months))
    noise = 1.0 + noise
    
    # Base demand with seasonality and noise
    base_series = base_demand * seasonality * holiday_boost * noise
    
    # Anomaly injection
    anomaly_month = anomaly["month_index"]
    impact_pct = anomaly["impact_pct"] / 100.0
    direction = anomaly.get("direction", "cliff")
    
    anomaly_impact = np.zeros(timeline_months)
    
    if direction == "cliff":
        # Pre-event: gradual decline (2 months before)
        for i in range(max(0, anomaly_month - 3), anomaly_month):
            if i < timeline_months:
                decay = (anomaly_month - i) / 3.0
                anomaly_impact[i] = impact_pct * 0.3 * (1 - decay)
        
        # At-event: sharp drop
        if anomaly_month <= timeline_months:
            anomaly_impact[anomaly_month - 1] = impact_pct
        
        # Post-event: gradual recovery (2-3 months after)
        for i in range(anomaly_month, min(timeline_months, anomaly_month + 3)):
            recovery = (i - anomaly_month + 1) / 3.0
            anomaly_impact[i] = impact_pct * (1 - recovery * 0.7)
    
    else:  # spike
        # Pre-event: gradual increase
        for i in range(max(0, anomaly_month - 3), anomaly_month):
            if i < timeline_months:
                growth = (anomaly_month - i) / 3.0
                anomaly_impact[i] = abs(impact_pct) * 0.3 * (1 - growth)
        
        # At-event: sharp spike
        if anomaly_month <= timeline_months:
            anomaly_impact[anomaly_month - 1] = abs(impact_pct)
        
        # Post-event: gradual decline
        for i in range(anomaly_month, min(timeline_months, anomaly_month + 3)):
            decline = (i - anomaly_month + 1) / 3.0
            anomaly_impact[i] = abs(impact_pct) * (1 - decline * 0.7)
    
    # Apply anomaly impact
    if direction == "cliff":
        true_demand = base_series * (1 + anomaly_impact)
    else:
        true_demand = base_series * (1 + anomaly_impact)
    
    # Ensure non-negative
    true_demand = np.maximum(true_demand, 100)
    
    df = pd.DataFrame({
        "month": months,
        "true_demand": true_demand,
        "anomaly_impact": anomaly_impact
    })
    
    return df

# ============================================================================
# 5. FORECASTING FUNCTIONS
# ============================================================================

@st.cache_data
def generic_forecast(df_sku: pd.DataFrame, reveal_month: int) -> pd.Series:
    """Simple lagged moving average forecast (no anomaly awareness) - generates for all 12 months starting from January"""
    df_filtered = df_sku[df_sku["month"] <= reveal_month].copy()
    
    # Ensure we have all months 1-12
    all_months = pd.Series(range(1, reveal_month + 1), name="month")
    forecast = pd.Series(index=all_months, dtype=float)
    
    # Initialize with first month's value for January
    if len(df_filtered) > 0:
        initial_value = df_filtered[df_filtered["month"] == 1]["true_demand"].values[0] if len(df_filtered[df_filtered["month"] == 1]) > 0 else df_filtered["true_demand"].iloc[0]
    else:
        initial_value = 1000
    
    for month in all_months:
        if month == 1:
            # January: use actual January value (generic doesn't know about future anomalies, so it's accurate early on)
            if len(df_filtered[df_filtered["month"] == 1]) > 0:
                forecast[month] = df_filtered[df_filtered["month"] == 1]["true_demand"].values[0]
            else:
                forecast[month] = initial_value
        elif month == 2:
            # February: average of Jan-Feb
            available = df_filtered[df_filtered["month"] <= 2]["true_demand"]
            forecast[month] = available.mean() if len(available) > 0 else initial_value
        elif month == 3:
            # March: average of Jan-Mar
            available = df_filtered[df_filtered["month"] <= 3]["true_demand"]
            forecast[month] = available.mean() if len(available) > 0 else initial_value
        else:
            # April onwards: 3-month moving average (very accurate for normal months)
            window = df_filtered[df_filtered["month"] < month]["true_demand"].tail(3)
            if len(window) > 0:
                forecast[month] = window.mean()
            else:
                forecast[month] = initial_value
    
    # Fill any missing months with forward fill then backward fill
    forecast = forecast.ffill().bfill()
    
    # Ensure all values are finite (no NaN or inf)
    forecast = forecast.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')
    
    # Final check: if any values are still NaN, fill with mean
    if forecast.isna().any():
        forecast = forecast.fillna(forecast.mean() if not forecast.isna().all() else 1000)
    
    return forecast

@st.cache_data
def vajra_forecast_with_alerts(df_sku: pd.DataFrame, anomaly: Dict, reveal_month: int) -> pd.Series:
    """Anomaly-aware forecasting - Vajra uses intelligence to predict anomalies accurately - generates for all 12 months"""
    df_filtered = df_sku[df_sku["month"] <= reveal_month].copy()
    
    # Ensure we have all months 1-12
    all_months = pd.Series(range(1, reveal_month + 1), name="month")
    forecast = pd.Series(index=all_months, dtype=float)
    
    anomaly_month = anomaly["month_index"]
    impact_pct = anomaly["impact_pct"] / 100.0
    direction = anomaly.get("direction", "cliff")
    
    # Vajra's key advantage: it knows about the anomaly from intelligence layer
    # It should track actual demand much more closely, especially during anomalies
    
    # Use the same base series calculation as true_demand for consistency
    # This ensures Vajra's base matches what was used to generate true_demand
    months = df_filtered["month"].values
    pre_anomaly_data = df_filtered[df_filtered["month"] < max(1, anomaly_month - 2)]
    
    if len(pre_anomaly_data) > 0:
        # Calculate base demand from pre-anomaly period (this matches true_demand's base)
        base_demand = pre_anomaly_data["true_demand"].mean()
        
        # Calculate seasonality pattern from pre-anomaly data
        # Use a simple approach: average the seasonal pattern
        seasonality_factor = 1.0  # Default
        if len(pre_anomaly_data) >= 3:
            # Estimate seasonality from pre-anomaly months
            seasonal_pattern = pre_anomaly_data["true_demand"].values / base_demand
            seasonality_factor = np.mean(seasonal_pattern)
    else:
        base_demand = df_filtered.head(3)["true_demand"].mean()
        seasonality_factor = 1.0
    
    # Get generic forecast for comparison
    generic_fcst = generic_forecast(df_sku, reveal_month)
    
    for month in df_filtered["month"]:
        # Start with base demand adjusted for seasonality
        month_idx = int(month) - 1
        base_fcst = base_demand * seasonality_factor
        
        # Calculate anomaly impact factor (matching the EXACT pattern from generate_demand_series)
        anomaly_impact_factor = 0.0
        
        if direction == "cliff":
            if month < anomaly_month - 2:
                # Pre-anomaly: Generic is more accurate (no anomaly to predict yet)
                # Vajra should be slightly less accurate pre-anomaly to show generic's strength in normal periods
                forecast[month] = generic_fcst.get(month, base_fcst) * 0.98  # Slightly off to give generic edge
                continue
            elif month < anomaly_month:
                # Pre-event: gradual decline (2-3 months before) - Vajra anticipates this
                decay = (anomaly_month - month) / 3.0
                anomaly_impact_factor = impact_pct * 0.3 * (1 - decay)
            elif month == anomaly_month:
                # At-event: sharp drop - Vajra predicts this accurately
                anomaly_impact_factor = impact_pct
            elif month <= anomaly_month + 2:
                # Post-event: gradual recovery - Vajra tracks recovery accurately
                recovery = (month - anomaly_month + 1) / 3.0
                anomaly_impact_factor = impact_pct * (1 - recovery * 0.7)
            else:
                # Post-recovery: Generic is more accurate again (back to normal)
                forecast[month] = generic_fcst.get(month, base_fcst) * 0.98  # Slightly off to give generic edge
                continue
        else:  # spike
            if month < anomaly_month - 2:
                forecast[month] = generic_fcst.get(month, base_fcst) * 0.98  # Slightly off to give generic edge
                continue
            elif month < anomaly_month:
                growth = (anomaly_month - month) / 3.0
                anomaly_impact_factor = abs(impact_pct) * 0.3 * (1 - growth)
            elif month == anomaly_month:
                anomaly_impact_factor = abs(impact_pct)
            elif month <= anomaly_month + 2:
                decline = (month - anomaly_month + 1) / 3.0
                anomaly_impact_factor = abs(impact_pct) * (1 - decline * 0.7)
            else:
                forecast[month] = generic_fcst.get(month, base_fcst) * 0.98  # Slightly off to give generic edge
                continue
        
        # Apply anomaly impact - Vajra knows about it proactively
        # Use the generic forecast as the base (which tracks the underlying pattern)
        # Then apply Vajra's known anomaly adjustment
        generic_base = generic_fcst.get(month, base_fcst)
        forecast[month] = generic_base * (1 + anomaly_impact_factor)
        
        # Vajra's intelligence makes it more accurate - refine the forecast
        # During anomaly period, Vajra's prediction is much closer to actual
        # This simulates Vajra having better intelligence about the anomaly impact
    
    # Ensure all months have values (fill any missing with generic forecast)
    for month in all_months:
        if month not in forecast.index or pd.isna(forecast.get(month)):
            forecast[month] = generic_fcst.get(month, base_fcst) if month in generic_fcst.index else base_fcst
    
    # Ensure the forecast Series is properly indexed with all months
    forecast = forecast.reindex(all_months)
    
    # Fill any remaining NaN values
    forecast = forecast.fillna(method='ffill').fillna(method='bfill')
    if forecast.isna().any():
        # Final fallback: use generic forecast or base value
        for month in all_months:
            if pd.isna(forecast[month]):
                forecast[month] = generic_fcst.get(month, base_fcst) if month in generic_fcst.index else base_fcst
    
    # Replace any inf values with NaN then fill
    forecast = forecast.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')
    if forecast.isna().any():
        forecast = forecast.fillna(forecast.mean() if not forecast.isna().all() else base_fcst)
    
    return forecast

# ============================================================================
# 6. COST & METRICS CALCULATIONS
# ============================================================================

def derive_cost_per_unit(sku: Dict, retailer: str) -> float:
    """Derives cost per unit from SKU type and retailer (deterministic)"""
    retailer_lower = retailer.lower()
    
    # Deterministic base cost based on SKU hash
    sku_hash = hash(sku["name"]) % 1000
    np.random.seed(sku_hash)
    
    # Base cost by category
    if sku.get("perishable", False):
        base_cost = np.random.uniform(15, 30)
    elif sku.get("frozen", False):
        base_cost = np.random.uniform(10, 20)
    else:  # shelf-stable
        base_cost = np.random.uniform(5, 15)
    
    # Organic/premium premium
    if sku.get("organic", False):
        base_cost += np.random.uniform(5, 10)
    
    # Retailer adjustments - removed hardcoded retailers
    # Base cost is determined by SKU characteristics only
    
    return round(base_cost, 2)

def calculate_tail_loss_multiplier(month: int, anomaly_month: int) -> float:
    """Smooth multiplier function with gentle shoulders - 3x peak at anomaly month"""
    distance = abs(month - anomaly_month)
    # Exponential decay: 3.0 × exp(-distance / 1.5)
    multiplier = 3.0 * np.exp(-distance / 1.5)
    # Ensure minimum of 1.0
    return max(1.0, multiplier)

def calculate_forecast_costs(
    df_sku: pd.DataFrame,
    generic_fcst: pd.Series,
    vajra_fcst: pd.Series,
    sku: Dict,
    anomaly: Dict,
    retailer: str,
    reveal_month: int
) -> Dict:
    """Calculate per-SKU forecast costs with tail-loss multipliers and baseline savings"""
    # Filter to revealed months only
    df_filtered = df_sku[df_sku["month"] <= reveal_month].copy()
    
    cost_per_unit = derive_cost_per_unit(sku, retailer)
    anomaly_month = anomaly["month_index"]
    
    # Calculate baseline cost (using naive forecast: mean of all historical data)
    # This represents "no forecasting" scenario
    baseline_forecast = df_filtered["true_demand"].mean()
    baseline_cost = 0.0
    generic_cost = 0.0
    vajra_cost = 0.0
    generic_errors = []
    vajra_errors = []
    
    for _, row in df_filtered.iterrows():
        month = int(row["month"])
        true_demand = row["true_demand"]
        
        # Get forecasts
        gen_fcst = generic_fcst.get(month, true_demand)
        vaj_fcst = vajra_fcst.get(month, true_demand)
        
        # Calculate errors
        baseline_error = abs(true_demand - baseline_forecast)
        gen_error = abs(true_demand - gen_fcst)
        vaj_error = abs(true_demand - vaj_fcst)
        
        generic_errors.append(gen_error)
        vajra_errors.append(vaj_error)
        
        # Calculate tail-loss multiplier
        multiplier = calculate_tail_loss_multiplier(month, anomaly_month)
        
        # Calculate costs
        baseline_cost += baseline_error * cost_per_unit * multiplier
        generic_cost += gen_error * cost_per_unit * multiplier
        vajra_cost += vaj_error * cost_per_unit * multiplier
    
    # Calculate savings relative to baseline
    generic_savings = baseline_cost - generic_cost
    generic_savings_pct = (generic_savings / baseline_cost * 100) if baseline_cost > 0 else 0
    
    vajra_savings = baseline_cost - vajra_cost
    vajra_savings_pct = (vajra_savings / baseline_cost * 100) if baseline_cost > 0 else 0
    
    # Vajra savings vs generic (additional savings)
    additional_savings = generic_cost - vajra_cost
    additional_savings_pct = (additional_savings / generic_cost * 100) if generic_cost > 0 else 0
    
    return {
        "baseline_cost": round(baseline_cost, 2),
        "generic_cost": round(generic_cost, 2),
        "vajra_cost": round(vajra_cost, 2),
        "generic_savings": round(generic_savings, 2),
        "generic_savings_pct": round(generic_savings_pct, 2),
        "vajra_savings": round(vajra_savings, 2),
        "vajra_savings_pct": round(vajra_savings_pct, 2),
        "additional_savings": round(additional_savings, 2),
        "additional_savings_pct": round(additional_savings_pct, 2),
        "error_mae_generic": round(np.mean(generic_errors), 2),
        "error_mae_vajra": round(np.mean(vajra_errors), 2),
        "cost_per_unit": cost_per_unit
    }

def calculate_portfolio_totals(skus_data: Dict) -> Dict:
    """Aggregate costs across all SKUs"""
    total_generic = sum(data["costs"]["generic_cost"] for data in skus_data.values())
    total_vajra = sum(data["costs"]["vajra_cost"] for data in skus_data.values())
    total_savings = total_generic - total_vajra
    avg_savings_pct = (total_savings / total_generic * 100) if total_generic > 0 else 0
    
    return {
        "total_generic_cost": round(total_generic, 2),
        "total_vajra_cost": round(total_vajra, 2),
        "total_savings": round(total_savings, 2),
        "avg_savings_pct": round(avg_savings_pct, 2)
    }

# ============================================================================
# 7. VISUALIZATION COMPONENTS
# ============================================================================

def render_sku_selector(skus: List[Dict]) -> str:
    """Render SKU selector and return selected SKU name"""
    sku_names = [sku["name"] for sku in skus]
    selected = st.selectbox(
        "Select SKU",
        options=sku_names,
        key="sku_selector"
    )
    return selected

def render_demand_chart(
    df_sku: pd.DataFrame,
    generic_fcst: pd.Series,
    vajra_fcst: pd.Series,
    anomaly: Dict,
    reveal_month: int
):
    """Render minimal, aesthetic demand chart"""
    # Filter to revealed months
    df_filtered = df_sku[df_sku["month"] <= reveal_month].copy()
    
    # Month names for labels
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    # Ensure month labels are valid strings
    month_labels = []
    for m in df_filtered["month"]:
        try:
            month_int = int(m)
            if 1 <= month_int <= 12:
                month_labels.append(month_names[month_int - 1])
            else:
                month_labels.append(f"M{month_int}")
        except (ValueError, TypeError):
            month_labels.append("")
    
    fig = go.Figure()
    
    # True Demand - Bar Chart (lightish gray)
    # Ensure all values are finite and valid
    true_demand_clean = df_filtered["true_demand"].replace([np.inf, -np.inf], np.nan).fillna(0)
    month_values = list(df_filtered["month"].astype(float))
    demand_values = list(true_demand_clean.astype(float))
    fig.add_trace(go.Bar(
        x=month_values,
        y=demand_values,
        name="True Demand",
        marker=dict(color="#B0B0B0", opacity=0.7),  # Lightish gray
        showlegend=True,
        hovertemplate="<b>True Demand</b><br>Month: %{x}<br>Value: %{y}<extra></extra>",
    ))
    
    # Generic Forecast - Line Chart (darkish gray)
    gen_filtered = generic_fcst[generic_fcst.index <= reveal_month].copy()
    # Remove NaN values and ensure valid data
    gen_filtered = gen_filtered.dropna()
    # Ensure all values are finite numbers
    gen_filtered = gen_filtered[np.isfinite(gen_filtered)]
    if len(gen_filtered) > 0:
        # Convert to lists to ensure proper data types
        gen_x = list(gen_filtered.index.astype(float))
        gen_y = list(gen_filtered.values.astype(float))
        fig.add_trace(go.Scatter(
            x=gen_x,
            y=gen_y,
            name="Generic Forecast",
            line=dict(color="#4A4A4A", width=2.5, dash="dash"),  # Darkish gray
            mode="lines",
            showlegend=True,
            hovertemplate="<b>Generic Forecast</b><br>Month: %{x}<br>Value: %{y}<extra></extra>",
        ))
    
    # Vajra Forecast - Line Chart (electric-violet)
    vaj_filtered = vajra_fcst[vajra_fcst.index <= reveal_month].copy()
    # Remove NaN values and ensure valid data
    vaj_filtered = vaj_filtered.dropna()
    # Ensure all values are finite numbers
    vaj_filtered = vaj_filtered[np.isfinite(vaj_filtered)]
    if len(vaj_filtered) > 0:
        # Convert to lists to ensure proper data types
        vaj_x = list(vaj_filtered.index.astype(float))
        vaj_y = list(vaj_filtered.values.astype(float))
        fig.add_trace(go.Scatter(
            x=vaj_x,
            y=vaj_y,
            name="Vajra Forecast",
            line=dict(color="#7F5AF0", width=3),  # Electric-violet
            mode="lines",
            showlegend=True,
            hovertemplate="<b>Vajra Forecast</b><br>Month: %{x}<br>Value: %{y}<extra></extra>",
        ))
    
    # Anomaly indicator - muted red
    anomaly_month = anomaly["month_index"]
    if anomaly_month <= reveal_month:
        fig.add_vline(
            x=anomaly_month,
            line_dash="solid",
            line_color="#C85A5A",  # Muted red - not too bright or jarring
            line_width=2.5,
            opacity=0.8,
            annotation_text="Anomaly",
            annotation_position="top",  # Back to top since legend is now on the side
            annotation=dict(
                font=dict(family="Space Grotesk, sans-serif", size=11, color="#C85A5A", weight=600),
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="#C85A5A",
                borderwidth=1,
                borderpad=4
            )
        )
    
    # Minimal, clean styling
    fig.update_layout(
        title=None,  # Remove title for minimal look
        xaxis=dict(
            title=None,  # No axis title - keep minimal
            tickmode="array",
            tickvals=list(df_filtered["month"].values),  # Ensure it's a list of numeric values
            ticktext=month_labels,  # January to December labels
            showgrid=False,
            showline=True,
            linecolor="#DADADA",
            linewidth=1,
            tickfont=dict(family="Space Grotesk, sans-serif", size=11, color="#0F0F0F"),
            showticklabels=True,  # Show month labels on x-axis
            ticks="outside",  # Ensure ticks are visible
            ticklen=5,  # Make tick marks visible
            tickwidth=1,
            type="linear"  # Explicitly set axis type
        ),
        yaxis=dict(
            title=None,  # No axis title - keep minimal
            showgrid=False,  # Remove horizontal grid lines for cleaner look
            showline=True,
            linecolor="#DADADA",
            linewidth=1,
            tickfont=dict(family="Space Grotesk, sans-serif", size=11, color="#0F0F0F"),
            showticklabels=True,  # Keep number labels on y-axis
            ticks="outside",  # Ensure ticks are visible
            ticklen=5,  # Make tick marks visible
            tickwidth=1
        ),
        template="plotly_white",  # Use white template to ensure tick labels show
        plot_bgcolor="rgba(255,255,255,0)",
        paper_bgcolor="rgba(255,255,255,0)",
        font=dict(family="Inter, sans-serif", color="#0F0F0F", size=12),
        hovermode="x unified",
        # Disable default hover template to prevent undefined from showing
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="#DADADA",
            font_size=11,
            font_family="Inter, sans-serif"
        ),
        legend=dict(
            orientation="v",  # Vertical legend on the right side
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,  # Position to the right of the chart
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#DADADA",
            borderwidth=1,
            font=dict(family="Inter, sans-serif", size=11, color="#0F0F0F")
        ),
        margin=dict(l=0, r=120, t=40, b=0),  # Add right margin for legend, top margin for anomaly label
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ============================================================================
# 8. RECOMMENDATIONS DISPLAY
# ============================================================================

def render_anomaly_alerts_page(anomaly: Dict, area: str):
    """Render the simplified Anomaly Alerts page"""
    import html
    from datetime import datetime
    
    # Check if anomaly exists
    if not anomaly or not anomaly.get('type'):
        st.info("✅ No current anomalies. All zones stable.")
        return
    
    # Extract anomaly information
    anomaly_type = anomaly.get('type', 'Unknown Anomaly')
    description = anomaly.get('description', '') or ''
    base_month_index = anomaly.get('base_month_index', 10)
    
    # Extract confidence from description FIRST (to ensure consistency with what's displayed)
    # This is the source of truth since it's what the LLM actually wrote
    import re
    confidence = 0.75  # Default fallback
    if description:
        if isinstance(description, list):
            description_str = ". ".join(str(item) for item in description if item)
        else:
            description_str = str(description)
        
        # Try to extract confidence percentage from description
        # Look for patterns like "with 65% confidence", "65% confidence", "confidence level of 65%", "a confidence level of 65%"
        confidence_match = re.search(r'(?:with\s+)?(?:a\s+)?(?:confidence\s+level\s+of\s+)?(\d{1,3})\s*%(?:\s+confidence)?', description_str, re.IGNORECASE)
        if confidence_match:
            extracted_confidence = int(confidence_match.group(1)) / 100.0
            # Use extracted confidence if it's reasonable (between 0.5 and 1.0)
            if 0.5 <= extracted_confidence <= 1.0:
                confidence = extracted_confidence
        else:
            # Fallback: try to get confidence from nested structure (weather API) or direct
            if 'anomaly' in anomaly and isinstance(anomaly['anomaly'], dict):
                confidence = anomaly['anomaly'].get('confidence', 0.75)
            else:
                confidence = anomaly.get('confidence', 0.75)
    else:
        # No description, use API confidence
        if 'anomaly' in anomaly and isinstance(anomaly['anomaly'], dict):
            confidence = anomaly['anomaly'].get('confidence', 0.75)
        else:
            confidence = anomaly.get('confidence', 0.75)
    
    # Try to get severity from nested structure (weather API) or direct
    severity = 'medium'  # Default
    if 'anomaly' in anomaly and isinstance(anomaly['anomaly'], dict):
        severity = anomaly['anomaly'].get('severity', 'medium')
    else:
        severity = anomaly.get('severity', 'medium')
    
    # Extract timeline from anomaly type or calculate from base_month_index
    current_date = datetime.now()
    month_names = ["", "January", "February", "March", "April", "May", "June", 
                   "July", "August", "September", "October", "November", "December"]
    
    # Try to extract date from anomaly type (e.g., "Heavy Monsoon Rains - November 15")
    import re
    # Look for pattern like "- November 15" or "November 15" at the end
    date_match = re.search(r'[-–]\s*(\w+)\s+(\d{1,2})', anomaly_type)
    if not date_match:
        # Try without dash separator
        date_match = re.search(r'(\w+)\s+(\d{1,2})(?:\s|$)', anomaly_type)
    
    if date_match:
        month_name = date_match.group(1)
        day = int(date_match.group(2))
        # Find month index
        month_index = None
        for i, m in enumerate(month_names):
            if m and m.lower().startswith(month_name.lower()):
                month_index = i
                break
        if month_index:
            timeline_start = f"{month_names[month_index]} {day}"
            timeline_end = f"{month_names[month_index]} {day + 1}"
            timeline_display = f"{timeline_start} - {timeline_end}"
        else:
            # Fallback to base_month_index
            anomaly_month = month_names[base_month_index] if 1 <= base_month_index <= 12 else "October"
            timeline_start = f"{anomaly_month} {day}"
            timeline_end = f"{anomaly_month} {day + 1}"
            timeline_display = f"{timeline_start} - {timeline_end}"
    else:
        # No date in type, use base_month_index
        anomaly_month = month_names[base_month_index] if 1 <= base_month_index <= 12 else "October"
        day = 15 + (current_date.day % 15)
        timeline_start = f"{anomaly_month} {day}"
        timeline_end = f"{anomaly_month} {day + 1}"
        timeline_display = f"{timeline_start} - {timeline_end}"
    
    # Estimate duration based on anomaly type
    duration_hours = 8  # Default
    if 'monsoon' in anomaly_type.lower() or 'rain' in anomaly_type.lower():
        duration_hours = 12
    elif 'wind' in anomaly_type.lower() or 'gust' in anomaly_type.lower():
        duration_hours = 6
    elif 'blizzard' in anomaly_type.lower() or 'snow' in anomaly_type.lower():
        duration_hours = 18
    
    duration_display = f"{duration_hours}h"
    if duration_hours >= 24:
        days = duration_hours // 24
        hours = duration_hours % 24
        duration_display = f"{days}d {hours}h" if hours > 0 else f"{days}d"
    
    # Calculate confidence percentage - use the extracted confidence (from description or API)
    if isinstance(confidence, float) and confidence <= 1.0:
        confidence_pct = int(round(confidence * 100))  # Round to nearest integer
    else:
        confidence_pct = int(confidence) if isinstance(confidence, (int, float)) else 75
    
    # Determine severity level and color
    severity_lower = str(severity).lower()
    if severity_lower in ['severe', 'high']:
        severity_label = "High"
        severity_color = "#EF4444"
    elif severity_lower in ['moderate', 'medium']:
        severity_label = "Medium"
        severity_color = "#FBBF24"
    else:
        severity_label = "Low"
        severity_color = "#10B981"
    
    # Extract deliveries affected from description or calculate from impact
    deliveries_affected = None
    if description:
        if isinstance(description, list):
            description_str = ". ".join(str(item) for item in description if item)
        else:
            description_str = str(description)
        
        # Try to extract deliveries affected percentage from description
        # Look for patterns like "12% capacity reduction", "affecting 15% of deliveries", "12% of deliveries", "causing 18% delivery delays"
        # Also look for patterns like "12% reduction", "15% impact", "affecting 18%"
        deliveries_match = re.search(r'(\d{1,2})\s*%\s*(?:capacity\s+reduction|of\s+deliveries|deliveries\s+affected|delivery\s+capacity|delivery\s+delays|reduction|impact|affecting)', description_str, re.IGNORECASE)
        if deliveries_match:
            deliveries_affected = int(deliveries_match.group(1))
        else:
            # Try to find percentage in context of delivery/impact keywords
            # Look for patterns like "X% ... delivery" or "delivery ... X%" or "impact ... X%"
            delivery_context_pattern = r'(?:delivery|deliveries|capacity|impact|reduction).*?(\d{1,2})\s*%|(\d{1,2})\s*%.*?(?:delivery|deliveries|capacity|impact|reduction)'
            context_match = re.search(delivery_context_pattern, description_str, re.IGNORECASE)
            if context_match:
                pct_val = int(context_match.group(1) or context_match.group(2))
                if 5 <= pct_val <= 30:
                    deliveries_affected = pct_val
            
            # If still not found, try simpler pattern: Look for any percentage in the impact section (usually 3rd bullet point)
            if deliveries_affected is None:
                # Split by sentences and check the last one (impact section)
                sentences = description_str.split('.')
                if len(sentences) >= 3:
                    impact_section = sentences[-1].strip()  # Last sentence
                else:
                    impact_section = description_str
                
                # Look for all percentages in impact context
                impact_percentages = re.findall(r'(\d{1,2})\s*%', impact_section)
                if impact_percentages:
                    # Use the first reasonable percentage (between 5-30%)
                    # Exclude confidence percentages (usually 50-100%)
                    for pct in impact_percentages:
                        pct_val = int(pct)
                        if 5 <= pct_val <= 30:
                            deliveries_affected = pct_val
                            break
    
    # Fallback: calculate from impact percentage if not found in description
    # Use a more varied calculation based on severity and impact to avoid always getting 12%
    if deliveries_affected is None:
        impact_pct = abs(anomaly.get('base_impact_pct', -35))
        severity = anomaly.get('severity', 'medium').lower()
        
        # Adjust multiplier based on severity
        if severity == 'high':
            multiplier = 0.40  # Higher impact for severe anomalies
        elif severity == 'low':
            multiplier = 0.25  # Lower impact for mild anomalies
        else:
            multiplier = 0.35  # Default for medium
        
        # Add variation using multiple factors to ensure we don't always get 12%
        # Use hash of area name + impact_pct + anomaly type to create consistent but varied values
        area_hash = hash(str(area)) if area else 0
        anomaly_type_hash = hash(str(anomaly.get('type', ''))) if anomaly.get('type') else 0
        # Create variation from -4 to +8 using multiple factors
        variation = ((int(impact_pct) % 7) + (abs(area_hash) % 5) + (abs(anomaly_type_hash) % 3) - 3)  # -3 to +12 variation
        base_calc = int(impact_pct * multiplier)
        deliveries_affected = min(max(base_calc + variation, 8), 25)  # Between 8-25%
        
        # Ensure we never get exactly 12% by adding a deterministic offset if needed
        if deliveries_affected == 12:
            # Use a deterministic offset based on area and anomaly type to ensure variation
            offset = (abs(area_hash) + abs(anomaly_type_hash) + int(impact_pct)) % 5  # 0-4
            if offset == 0:
                deliveries_affected = 11  # Slightly lower
            elif offset == 1:
                deliveries_affected = 13  # Slightly higher
            elif offset == 2:
                deliveries_affected = 14  # A bit higher
            elif offset == 3:
                deliveries_affected = 10  # A bit lower
            else:  # offset == 4
                deliveries_affected = 15  # Higher
    
    # Extract areas from description or use provided area
    areas_affected = [area] if area else ["Unknown"]
    import re
    zip_pattern = r'\b\d{5,6}\b'
    zip_codes = re.findall(zip_pattern, description)
    if zip_codes:
        areas_affected = zip_codes[:3]
    
    # Main headline with emoji and timeline
    # Determine appropriate emoji based on anomaly type
    anomaly_lower = anomaly_type.lower()
    if 'rain' in anomaly_lower or 'monsoon' in anomaly_lower or 'flood' in anomaly_lower:
        emoji = "🌧️"
    elif 'wind' in anomaly_lower or 'gust' in anomaly_lower or 'storm' in anomaly_lower:
        emoji = "💨"
    elif 'snow' in anomaly_lower or 'blizzard' in anomaly_lower or 'ice' in anomaly_lower:
        emoji = "❄️"
    elif 'heat' in anomaly_lower or 'hot' in anomaly_lower:
        emoji = "🌡️"
    elif 'thunder' in anomaly_lower or 'lightning' in anomaly_lower:
        emoji = "⛈️"
    elif 'cyclone' in anomaly_lower or 'hurricane' in anomaly_lower or 'tropical' in anomaly_lower:
        emoji = "🌀"
    else:
        emoji = "⚠️"
    
    # Extract just the start date for headline (remove range if present)
    # If anomaly type already has a date, use it; otherwise add start date
    if date_match:
        month_name = date_match.group(1) if date_match else None
        day = int(date_match.group(2)) if date_match else None
        if month_name and day:
            # Find month index
            month_index = None
            for i, m in enumerate(month_names):
                if m and m.lower().startswith(month_name.lower()):
                    month_index = i
                    break
            if month_index:
                start_date_display = f"{month_names[month_index]} {day}"
            else:
                anomaly_month = month_names[base_month_index] if 1 <= base_month_index <= 12 else "October"
                start_date_display = f"{anomaly_month} {day}"
        else:
            anomaly_month = month_names[base_month_index] if 1 <= base_month_index <= 12 else "October"
            day = 15 + (current_date.day % 15)
            start_date_display = f"{anomaly_month} {day}"
    else:
        # No date in type, use base_month_index
        anomaly_month = month_names[base_month_index] if 1 <= base_month_index <= 12 else "October"
        day = 15 + (current_date.day % 15)
        start_date_display = f"{anomaly_month} {day}"
    
    # Create headline with just start date
    # Remove date from anomaly_type if it exists, then add our extracted date
    anomaly_type_clean = re.sub(r'\s*[-–]\s*\w+\s+\d{1,2}.*$', '', anomaly_type).strip()
    headline_with_date = f"{anomaly_type_clean} - {start_date_display}"
    
    st.markdown(f"""
    <div style="margin-bottom: 1.5rem;">
        <h2 style="font-family: 'Space Grotesk', sans-serif; font-weight: 600; color: #0F0F0F; font-size: 1.5rem; margin: 0;">
            {emoji} {html.escape(headline_with_date)}
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Details in a 2x2 grid format
    st.markdown("""
    <div style="background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 8px; padding: 1.5rem; margin-bottom: 1.5rem;">
    """, unsafe_allow_html=True)
    
    # Create 2x2 grid using columns
    col1, col2 = st.columns(2)
    
    with col1:
        # Duration
        st.markdown(f"""
        <div style="margin-bottom: 1.5rem;">
            <div style="font-family: Inter, sans-serif; font-size: 0.75rem; color: #6B7280; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">Duration</div>
            <div style="font-family: Inter, sans-serif; font-size: 1rem; color: #0F0F0F; font-weight: 500;">{html.escape(duration_display)}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence Level
        st.markdown(f"""
        <div>
            <div style="font-family: Inter, sans-serif; font-size: 0.75rem; color: #6B7280; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">Confidence Level</div>
            <div style="font-family: Inter, sans-serif; font-size: 1rem; color: #0F0F0F; font-weight: 500; margin-bottom: 0.5rem;">{confidence_pct}%</div>
            <div style="width: 100%; height: 8px; background: #E5E7EB; border-radius: 4px; overflow: hidden;">
                <div style="width: {confidence_pct}%; height: 100%; background: linear-gradient(90deg, #7C3AED 0%, #9333EA 100%); border-radius: 4px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Severity
        st.markdown(f"""
        <div style="margin-bottom: 1.5rem;">
            <div style="font-family: Inter, sans-serif; font-size: 0.75rem; color: #6B7280; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">Severity</div>
            <div style="font-family: Inter, sans-serif; font-size: 1rem; color: #0F0F0F; font-weight: 500;">{severity_label}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Deliveries Affected
        st.markdown(f"""
        <div>
            <div style="font-family: Inter, sans-serif; font-size: 0.75rem; color: #6B7280; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">Deliveries Affected</div>
            <div style="font-family: Inter, sans-serif; font-size: 1rem; color: #0F0F0F; font-weight: 500; margin-bottom: 0.5rem;">{deliveries_affected}%</div>
            <div style="width: 100%; height: 8px; background: #E5E7EB; border-radius: 4px; overflow: hidden;">
                <div style="width: {deliveries_affected}%; height: 100%; background: linear-gradient(90deg, #7C3AED 0%, #9333EA 100%); border-radius: 4px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Description section - Operational Details
    if description:
        # Parse and display description
        if isinstance(description, list):
            description = ". ".join(str(item) for item in description if item)
        elif not isinstance(description, str):
            description = str(description) if description else ''
        
        import re
        description = description.strip()
        description = re.sub(r'^[•\*]\s*', '', description, flags=re.MULTILINE)
        description = re.sub(r'[ \t]+', ' ', description)
        sentences = re.split(r'\.\s+', description)
        
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                sentence = sentence.rstrip('.')
                if len(sentence) >= 15:
                    cleaned_sentences.append(sentence)
        
        if cleaned_sentences:
            st.markdown("""
            <div style="background: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 8px; padding: 1.5rem; margin-top: 2rem;">
                <h4 style="font-family: Inter, sans-serif; font-weight: 600; color: #0F0F0F; font-size: 0.875rem; margin: 0 0 1rem 0; text-transform: uppercase; letter-spacing: 0.05em;">Operational Details</h4>
            </div>
            """, unsafe_allow_html=True)
            
            bullet_html = '<div style="background: #F9FAFB; border: 1px solid #E5E7EB; border-top: none; border-radius: 0 0 8px 8px; padding: 0 1.5rem 1.5rem 1.5rem; font-family: Inter, sans-serif; color: #374151; line-height: 1.7; font-size: 0.9rem;">'
            for sentence in cleaned_sentences:
                sentence = sentence.strip()
                if sentence:
                    sentence_escaped = html.escape(sentence)
                    bullet_html += f'<p style="margin: 0.75rem 0; padding-left: 1.25rem; position: relative;"><span style="position: absolute; left: 0; color: #7C3AED; font-weight: 600;">•</span><span style="margin-left: 0.5rem;">{sentence_escaped}</span></p>'
            bullet_html += '</div>'
            st.markdown(bullet_html, unsafe_allow_html=True)

def render_recommendations(sku: Dict, anomaly: Dict, recommendations: List[str]):
    """Display 3 actionable recommendations with improved design"""
    import html
    
    # Get the first (most important) recommendation as headline
    headline_rec = ""
    if recommendations and len(recommendations) > 0:
        headline_rec = str(recommendations[0]).strip()
        # Clean up the recommendation text (remove any leading bullets, etc.)
        headline_rec = headline_rec.lstrip('•').lstrip('*').lstrip('-').strip()
        # Ensure it's a complete sentence (if it ends with period, keep it; otherwise add one)
        if headline_rec and not headline_rec.endswith(('.', '!', '?')):
            headline_rec = headline_rec.rstrip('.') + '.'
    
    # If no recommendation available, use default
    if not headline_rec:
        headline_rec = "Review operational recommendations below"
    
    st.markdown("---")
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(127, 90, 240, 0.05) 0%, rgba(100, 56, 183, 0.05) 100%); 
                border-left: 3px solid #6438B7; 
                padding: 1.5rem; 
                border-radius: 8px; 
                margin: 1.5rem 0;">
        <h3 style="font-family: 'Space Grotesk', sans-serif; font-weight: 600; color: #0F0F0F; margin-top: 0; margin-bottom: 1rem; font-size: 1.1rem; line-height: 1.5;">
            {html.escape(headline_rec)}
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Display remaining recommendations (skip the first one since it's the headline)
    remaining_recs = recommendations[1:3] if len(recommendations) > 1 else []
    
    if remaining_recs:
        # Add a title for additional recommendations
        st.markdown("""
        <div style="padding: 0 1.5rem 0.5rem 1.5rem; font-family: 'Inter', sans-serif;">
            <h4 style="font-family: 'Space Grotesk', sans-serif; font-weight: 600; color: #0F0F0F; font-size: 0.875rem; margin: 0; text-transform: uppercase; letter-spacing: 0.05em;">
                Additional Steps
            </h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Display the remaining 2 recommendations as bullet points
        bullet_html = '<div style="padding: 0 1.5rem 1.5rem 1.5rem; font-family: \'Inter\', sans-serif; color: #0F0F0F; line-height: 1.8;">'
        for rec in remaining_recs:
            if rec:
                rec_str = str(rec).strip()
                if rec_str:
                    # Escape HTML to prevent XSS and ensure proper display
                    rec_escaped = html.escape(rec_str)
                    bullet_html += f'<p style="margin: 0.75rem 0; padding-left: 1rem; position: relative;">'
                    bullet_html += f'<span style="position: absolute; left: 0; color: #6438B7;">•</span>'
                    bullet_html += f'<span style="margin-left: 0.5rem;">{rec_escaped}</span>'
                    bullet_html += '</p>'
        bullet_html += '</div>'
        st.markdown(bullet_html, unsafe_allow_html=True)

# ============================================================================
# 9. MAIN APP FLOW
# ============================================================================

def main():
    """Main application flow"""
    # Track page load time
    page_load_start = time.time()
    
    # Initialize metrics database
    init_database()
    
    # Apply styling
    apply_custom_styling()
    
    # Track page load completion
    page_load_time = (time.time() - page_load_start) * 1000
    track_page_load(page_load_time)
    
    # Title
    st.title("Vajra Prototype")
    st.markdown("**Weather & Disruption Intelligence for Food Delivery Operations**")
    
    # Show LLM status (hidden from UI as requested)
    client = get_openai_client()
    # Debug: Show API status (only in sidebar for debugging, no key shown)
    with st.sidebar:
        st.markdown("---")
        # Admin panel access
        if st.button("🔒 Admin Panel", use_container_width=True, help="View metrics and analytics"):
            st.session_state.show_admin = True
            st.rerun()
        
    st.markdown("---")
    
    # Sidebar
    inputs = render_sidebar()
    
    # Initialize session state for evaluation trigger
    if "evaluated" not in st.session_state:
        st.session_state.evaluated = False
    
    # Check if button was clicked
    button_clicked = inputs.get("evaluate_button", False)
    if button_clicked:
        # Track button click
        track_button_click("Evaluate Forecast", inputs["retailer"], inputs["area"])
        st.session_state.evaluated = True
        st.session_state.evaluating = True
        # Initialize metrics tracking for this evaluation
        st.session_state.evaluation_metrics = {
            "retailer": inputs["retailer"],
            "area": inputs["area"],
            "total_time_ms": 0,
            "api_calls_time_ms": 0,
            "chart_data_time_ms": 0,
            "chart_render_time_ms": 0,
            "sku_generation_time_ms": 0,
            "anomaly_generation_time_ms": 0,
            "recommendations_time_ms": 0,
            "forecast_calculation_time_ms": 0,
            "cost_calculation_time_ms": 0
        }
        st.session_state.evaluation_start_time = time.time()
    
    # Check if using default/placeholder values
    is_default = inputs["retailer"].strip() == "" or inputs["area"].strip() == ""
    
    if is_default or not st.session_state.evaluated:
        # Show inviting default page
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(127, 90, 240, 0.05) 0%, rgba(100, 56, 183, 0.05) 100%); 
                    border-left: 4px solid #7F5AF0; 
                    border-radius: 12px; 
                    padding: 2.5rem; 
                    margin: 2rem 0;
                    text-align: center;">
            <h2 style="font-family: 'Space Grotesk', sans-serif; font-weight: 600; color: #0F0F0F; margin-bottom: 1rem; font-size: 1.75rem;">
                Ready to explore intelligent forecasting?
            </h2>
            <p style="font-family: 'Inter', sans-serif; font-size: 1.1rem; color: #374151; line-height: 1.7; margin-bottom: 1.5rem; max-width: 600px; margin-left: auto; margin-right: auto;">
                Enter your retailer name and location in the sidebar to unlock real-time weather intelligence, 
                anomaly detection, and precision demand forecasts tailored to your operations.
            </p>
            <div style="display: flex; align-items: center; justify-content: center; gap: 0.5rem; color: #7F5AF0; font-family: 'Inter', sans-serif; font-weight: 500; font-size: 0.95rem;">
                <span>✨</span>
                <span>Get started in seconds</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Placeholder chart with better design
        st.markdown("""
        <div style="background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 12px; padding: 3rem 2rem; margin: 2rem 0; text-align: center;">
            <div style="font-family: 'Space Grotesk', sans-serif; font-weight: 600; color: #0F0F0F; font-size: 1.25rem; margin-bottom: 1rem;">
                Demand Forecast Comparison
            </div>
            <div style="font-family: 'Inter', sans-serif; color: #7A88A1; font-size: 0.95rem; line-height: 1.6;">
                Interactive forecast visualization will appear here once you enter your retailer and location
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Placeholder table with better design
        st.markdown("""
        <div style="margin: 2rem 0;">
            <div style="font-family: 'Space Grotesk', sans-serif; font-weight: 600; color: #0F0F0F; font-size: 1.25rem; margin-bottom: 1rem;">
                Forecast Analysis
            </div>
            <div style="background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 12px; overflow: hidden;">
                <table style="width: 100%; border-collapse: collapse; font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;">
                    <thead>
                        <tr style="background: linear-gradient(135deg, #7F5AF0 0%, #6438B7 100%); color: #ffffff;">
                            <th style="padding: 1rem 1.5rem; text-align: left; font-weight: 600; font-family: 'Space Grotesk', sans-serif;">Metric</th>
                            <th style="padding: 1rem 1.5rem; text-align: left; font-weight: 600; font-family: 'Space Grotesk', sans-serif;">Generic Forecast</th>
                            <th style="padding: 1rem 1.5rem; text-align: left; font-weight: 600; font-family: 'Space Grotesk', sans-serif;">Vajra Forecast</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr style="border-bottom: 1px solid #E5E7EB;">
                            <td style="padding: 1rem 1.5rem; font-weight: 500; color: #0F0F0F;">Accuracy</td>
                            <td style="padding: 1rem 1.5rem; color: #9CA3AF; font-style: italic;">—</td>
                            <td style="padding: 1rem 1.5rem; color: #9CA3AF; font-style: italic;">—</td>
                        </tr>
                        <tr>
                            <td style="padding: 1rem 1.5rem; font-weight: 500; color: #0F0F0F;">Savings</td>
                            <td style="padding: 1rem 1.5rem; color: #9CA3AF; font-style: italic;">—</td>
                            <td style="padding: 1rem 1.5rem; color: #9CA3AF; font-style: italic;">—</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Show progress bar if evaluating
    progress_placeholder = st.empty()
    progress_bar = None
    status_text = None
    
    if st.session_state.get("evaluating", False):
        with progress_placeholder.container():
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.markdown("""
            <div style="text-align: center; font-family: 'Inter', sans-serif; color: #7F5AF0; font-size: 0.95rem; font-weight: 500; margin-top: 0.5rem;">
                🚀 Starting your forecast analysis...
            </div>
            """, unsafe_allow_html=True)
    
    # Track API calls time
    metrics = st.session_state.get("evaluation_metrics", {})
    api_start = time.time()
    
    # Fun commentary messages
    def update_progress_with_message(progress, message):
        if progress_bar and status_text:
            progress_bar.progress(progress / 100.0)
            status_text.markdown(f"""
            <div style="text-align: center; font-family: 'Inter', sans-serif; color: #7F5AF0; font-size: 0.95rem; font-weight: 500; margin-top: 0.5rem;">
                {message}
            </div>
            """, unsafe_allow_html=True)
    
    # Generate SKUs using LLM (with fallback)
    update_progress_with_message(5, "🛒 Identifying top SKUs for your retailer...")
    with track_time("sku_generation_time_ms", metrics):
        skus = llm_top_skus(inputs["retailer"])
    
    update_progress_with_message(20, "🌦️ Fetching real-time weather intelligence...")
    
    # Generate base anomaly using LLM (with fallback)
    with track_time("anomaly_generation_time_ms", metrics):
        base_anomaly = llm_anomaly(inputs["area"])
    
    update_progress_with_message(35, "⚡ Processing anomaly intelligence...")
    
    # Calculate total API calls time
    metrics["api_calls_time_ms"] = (time.time() - api_start) * 1000
    
    # Generate per-SKU anomaly variations
    update_progress_with_message(40, "🔄 Generating anomaly variations...")
    sku_anomalies = generate_sku_anomaly_variations(
        base_anomaly, len(skus), inputs["seed"]
    )
    
    # Track chart data generation time
    chart_data_start = time.time()
    
    # Process each SKU
    skus_data = {}
    total_skus = len(skus)
    fun_messages = [
        "📦 Crunching numbers for {name}...",
        "🎯 Calculating forecasts for {name}...",
        "💡 Generating insights for {name}...",
        "✨ Polishing data for {name}...",
        "🔍 Analyzing patterns for {name}..."
    ]
    
    for idx, (sku, anomaly) in enumerate(zip(skus, sku_anomalies)):
        # Update progress based on SKU processing
        sku_progress = 40 + (idx / total_skus) * 50
        sku_name = sku.get('name', 'SKU')
        message_template = fun_messages[idx % len(fun_messages)]
        update_progress_with_message(int(sku_progress), message_template.format(name=sku_name))
        
        # Generate demand series
        df_sku = generate_demand_series(
            sku, anomaly, inputs["timeline_months"], inputs["seed"]
        )
        
        # Calculate forecasts
        with track_time("forecast_calculation_time_ms", metrics):
            generic_fcst = generic_forecast(df_sku, inputs["reveal_month"])
            vajra_fcst = vajra_forecast_with_alerts(
                df_sku, anomaly, inputs["reveal_month"]
            )
        
        # Calculate costs
        with track_time("cost_calculation_time_ms", metrics):
            costs = calculate_forecast_costs(
                df_sku, generic_fcst, vajra_fcst, sku, anomaly,
                inputs["retailer"], inputs["reveal_month"]
            )
        
        # Generate recommendations using LLM (with fallback)
        with track_time("recommendations_time_ms", metrics):
            recommendations = llm_recommendations(sku, anomaly)
        
        skus_data[sku["name"]] = {
            "sku": sku,
            "anomaly": anomaly,
            "df": df_sku,
            "generic_fcst": generic_fcst,
            "vajra_fcst": vajra_fcst,
            "costs": costs,
            "recommendations": recommendations
        }
    
    # Calculate total chart data generation time
    metrics["chart_data_time_ms"] = (time.time() - chart_data_start) * 1000
    
    # Final progress update
    update_progress_with_message(95, "🎨 Preparing your dashboard...")
    
    # SKU Selector
    selected_sku_name = render_sku_selector(skus)
    selected_data = skus_data[selected_sku_name]
    
    # Complete progress and clear
    if progress_placeholder and st.session_state.get("evaluating", False):
        update_progress_with_message(100, "✅ Ready! Your forecast is complete.")
        time.sleep(0.5)  # Brief pause to show completion
        progress_placeholder.empty()
        st.session_state.evaluating = False
    
    # Create 3 main tabs: Anomaly Alerts, Recommendations, Forecast Analysis
    tab1, tab2, tab3 = st.tabs(["Anomaly Alerts", "Recommendations", "Forecast Analysis"])
    
    with tab1:
        # Anomaly Alerts Page - Redesigned
        render_anomaly_alerts_page(selected_data['anomaly'], inputs.get('area', ''))
    
    with tab2:
        # Recommendations
        render_recommendations(
            selected_data["sku"],
            selected_data["anomaly"],
            selected_data["recommendations"]
        )
    
    with tab3:
        # Chart in Forecast Analysis tab - track render time
        chart_render_start = time.time()
        render_demand_chart(
            selected_data["df"],
            selected_data["generic_fcst"],
            selected_data["vajra_fcst"],
            selected_data["anomaly"],
            inputs["reveal_month"]
        )
        metrics["chart_render_time_ms"] = (time.time() - chart_render_start) * 1000
        
        # Analysis Table
        st.subheader("Forecast Analysis")
        
        # Calculate accuracy (MAPE-based)
        df_filtered = selected_data["df"][selected_data["df"]["month"] <= inputs["reveal_month"]]
        mean_demand = df_filtered["true_demand"].mean()
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape_generic = (selected_data['costs']['error_mae_generic'] / mean_demand * 100) if mean_demand > 0 else 0
        mape_vajra = (selected_data['costs']['error_mae_vajra'] / mean_demand * 100) if mean_demand > 0 else 0
        
        # Accuracy as percentage (100 - MAPE)
        accuracy_generic = max(0, 100 - mape_generic)
        accuracy_vajra = max(0, 100 - mape_vajra)
        
        # Get savings data
        generic_savings = selected_data['costs']['generic_savings']
        generic_savings_pct = selected_data['costs']['generic_savings_pct']
        vajra_savings = selected_data['costs']['vajra_savings']
        vajra_savings_pct = selected_data['costs']['vajra_savings_pct']
        
        # Display styled table with custom HTML for better aesthetics
        st.markdown(f"""
        <div style="margin: 1.5rem 0;">
            <table style="width: 100%; border-collapse: collapse; font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;">
                <thead>
                    <tr style="background: linear-gradient(135deg, #7F5AF0 0%, #6438B7 100%); color: #ffffff;">
                        <th style="padding: 0.75rem 1rem; text-align: left; font-weight: 600; border-radius: 8px 0 0 0;">Metric</th>
                        <th style="padding: 0.75rem 1rem; text-align: left; font-weight: 600;">Generic Forecast</th>
                        <th style="padding: 0.75rem 1rem; text-align: left; font-weight: 600; border-radius: 0 8px 0 0;">Vajra Forecast</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style="border-bottom: 1px solid #DADADA;">
                        <td style="padding: 0.75rem 1rem; font-weight: 500; color: #0F0F0F;">Accuracy</td>
                        <td style="padding: 0.75rem 1rem; color: #ef4444;">{accuracy_generic:.1f}%</td>
                        <td style="padding: 0.75rem 1rem; color: #7F5AF0; font-weight: 600;">{accuracy_vajra:.1f}%</td>
                    </tr>
                    <tr>
                        <td style="padding: 0.75rem 1rem; font-weight: 500; color: #0F0F0F;">Savings</td>
                        <td style="padding: 0.75rem 1rem; color: #ef4444;">${generic_savings:,.2f} ({generic_savings_pct:.1f}%)</td>
                        <td style="padding: 0.75rem 1rem; color: #7F5AF0; font-weight: 600;">${vajra_savings:,.2f} ({vajra_savings_pct:.1f}%)</td>
                    </tr>
                </tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)
    
    
    # Track total evaluation time and save metrics
    if "evaluation_start_time" in st.session_state:
        metrics["total_time_ms"] = (time.time() - st.session_state.evaluation_start_time) * 1000
        track_evaluation_performance(metrics)
        # Clean up
        del st.session_state.evaluation_start_time
        st.session_state.evaluation_metrics = metrics
    
    # Footer
    st.markdown("---")

def render_admin_panel():
    """Render admin panel for metrics tracking"""
    st.title("Admin Panel")
    st.markdown("**Metrics & Analytics Dashboard**")
    st.markdown("---")
    
    # Get metrics
    button_clicks = get_button_click_count("Evaluate Forecast")
    avg_page_load = get_average_page_load_time()
    eval_stats = get_evaluation_stats()
    recent_evals = get_recent_evaluations(limit=20)
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Evaluations",
            value=button_clicks,
            delta=None
        )
    
    with col2:
        st.metric(
            label="Avg Page Load",
            value=f"{avg_page_load:.1f} ms",
            delta=None
        )
    
    with col3:
        st.metric(
            label="Avg Total Time",
            value=f"{eval_stats['avg_total_time_ms']:.1f} ms",
            delta=None
        )
    
    with col4:
        st.metric(
            label="Avg API Calls",
            value=f"{eval_stats['avg_api_calls_time_ms']:.1f} ms",
            delta=None
        )
    
    st.markdown("---")
    
    # Performance Breakdown
    st.subheader("Performance Breakdown")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Average Times (ms)**")
        perf_data = {
            "Metric": [
                "Total Evaluation",
                "API Calls",
                "Chart Data Generation",
                "Chart Rendering",
                "SKU Generation",
                "Anomaly Generation",
                "Recommendations",
                "Forecast Calculation",
                "Cost Calculation"
            ],
            "Time (ms)": [
                eval_stats['avg_total_time_ms'],
                eval_stats['avg_api_calls_time_ms'],
                eval_stats['avg_chart_data_time_ms'],
                eval_stats['avg_chart_render_time_ms'],
                eval_stats['avg_sku_generation_time_ms'],
                eval_stats['avg_anomaly_generation_time_ms'],
                eval_stats['avg_recommendations_time_ms'],
                eval_stats['avg_forecast_calculation_time_ms'],
                eval_stats['avg_cost_calculation_time_ms']
            ]
        }
        perf_df = pd.DataFrame(perf_data)
        st.dataframe(perf_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("**Performance Chart**")
        if eval_stats['total_evaluations'] > 0:
            chart_data = {
                "API Calls": eval_stats['avg_api_calls_time_ms'],
                "Chart Data": eval_stats['avg_chart_data_time_ms'],
                "Chart Render": eval_stats['avg_chart_render_time_ms']
            }
            
            fig = go.Figure(data=[
                go.Bar(
                    x=list(chart_data.keys()),
                    y=list(chart_data.values()),
                    marker_color="#7F5AF0",
                    text=[f"{v:.1f}ms" for v in chart_data.values()],
                    textposition="outside"
                )
            ])
            fig.update_layout(
                title=None,
                yaxis_title="Time (ms)",
                height=300,
                margin=dict(l=0, r=0, t=0, b=0),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif", color="#0F0F0F", size=12)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No evaluation data yet. Run some evaluations to see performance metrics.")
    
    st.markdown("---")
    
    # Recent Evaluations
    st.subheader("Recent Evaluations")
    if recent_evals:
        recent_df = pd.DataFrame(recent_evals)
        recent_df['timestamp'] = pd.to_datetime(recent_df['timestamp'])
        recent_df = recent_df.sort_values('timestamp', ascending=False)
        recent_df['timestamp'] = recent_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        st.dataframe(
            recent_df[['timestamp', 'retailer', 'area', 'total_time_ms', 'api_calls_time_ms', 'chart_data_time_ms', 'chart_render_time_ms']],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No recent evaluations to display.")

if __name__ == "__main__":
    # Check if admin panel is requested
    if st.session_state.get("show_admin", False):
        apply_custom_styling()
        
        # Check if user is logged in
        if not st.session_state.get("admin_logged_in", False):
            # Show login form
            st.title("🔒 Admin Login")
            st.markdown("---")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                with st.form("admin_login_form"):
                    username = st.text_input("Username", type="default")
                    password = st.text_input("Password", type="password")
                    submit_button = st.form_submit_button("Login", use_container_width=True)
                    
                    if submit_button:
                        if username == "anirudh_admin" and password == "anirudh_password":
                            st.session_state.admin_logged_in = True
                            st.success("Login successful!")
                            st.rerun()
                        else:
                            st.error("Invalid username or password")
            
            # Add back button (centered)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("← Back to Main App", use_container_width=True):
                    st.session_state.show_admin = False
                    st.rerun()
        else:
            # User is logged in, show admin panel
            render_admin_panel()
            # Add logout and back buttons
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("← Back to Main App"):
                    st.session_state.show_admin = False
                    st.session_state.admin_logged_in = False
                    st.rerun()
            with col2:
                if st.button("🚪 Logout"):
                    st.session_state.admin_logged_in = False
                    st.rerun()
    else:
        main()

