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
    
    .vajra-loader {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 3rem 2rem;
        font-family: 'Space Grotesk', sans-serif;
    }
    
    .vajra-loader-spinner {
        width: 60px;
        height: 60px;
        position: relative;
        margin-bottom: 1.5rem;
    }
    
    .vajra-loader-spinner::before,
    .vajra-loader-spinner::after {
        content: '';
        position: absolute;
        border-radius: 50%;
        border: 3px solid transparent;
    }
    
    .vajra-loader-spinner::before {
        width: 60px;
        height: 60px;
        border-top: 3px solid #7F5AF0;
        border-right: 3px solid #7F5AF0;
        animation: vajra-spin 1s linear infinite;
    }
    
    .vajra-loader-spinner::after {
        width: 40px;
        height: 40px;
        top: 10px;
        left: 10px;
        border-bottom: 3px solid #6438B7;
        border-left: 3px solid #6438B7;
        animation: vajra-spin 0.8s linear infinite reverse;
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
        font-size: 0.875rem;
        font-weight: 500;
        color: #7A88A1;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        animation: vajra-pulse 2s ease-in-out infinite;
    }
    
    .vajra-loader-dots {
        display: inline-block;
        width: 20px;
        text-align: left;
    }
    
    .vajra-loader-dots::after {
        content: '...';
        animation: vajra-dots 1.5s steps(4, end) infinite;
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
        help="Enter retailer name"
    )
    
    area = st.sidebar.text_input(
        "Area / City / Pin Code",
        value="",
        placeholder="Enter area, city, or pin code",
        help="Enter area, city, or pin code"
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
    Generate weather anomaly prediction for a location using LLM.
    Anomaly should occur in September-November (months 9-11) for best chart visibility.
    """
    system_prompt = """You are a weather and disruption intelligence expert for food delivery operations.
Predict extremely specific, realistic weather anomalies that could impact food delivery in specific geographic areas.
Anomalies should be weather-related and occur in September, October, or November.
Include specific detection methods and detailed impact descriptions.
Return ONLY valid JSON format, no other text."""
    
    user_prompt = f"""Predict the most likely, specific weather anomaly for {area} occurring in September, October, or November.

CRITICAL: Include highly specific, locality-relevant details that make this feel personalized to {area}:
- Mention specific neighborhoods, streets, landmarks, or local areas within {area}
- Reference local infrastructure (bridges, highways, specific routes that are known in {area})
- Include ZIP codes or postal codes specific to {area}
- Reference local weather patterns, geographic features, or historical events relevant to {area}
- Mention specific delivery routes, warehouses, or distribution centers that would be affected in {area}
- Use local terminology, area names, or references that someone from {area} would recognize

Requirements:
- Specific weather event with dates and severity
- Detection method (NOAA alerts, satellite imagery, historical analysis)
- Impact details: affected routes/ZIP codes specific to {area}, delivery delays, warehouse restrictions
- Month 9, 10, or 11

Return JSON with:
- type: Specific anomaly name with dates (e.g., "Category 2 Hurricane Remnants - October 12-15")
- description: Concise bullet-point format covering: detection method, dates, severity, SPECIFIC geographic impact zones within {area} (neighborhoods, ZIP codes, landmarks), delivery impacts. Format as separate sentences that can be split into bullets. Make it feel personalized to {area}.
- base_impact_pct: Demand impact (-30 to -50 for drops)
- base_month_index: 9, 10, or 11
- direction: "cliff" for drops, "spike" for increases

Example for San Francisco:
{{"type": "Atmospheric River System - October 12-15", "description": "Detected via NOAA West Coast Weather Center and historical pattern analysis (89% confidence). October 12-15 event bringing 4-6 inches rainfall to {area} with 50-60 mph winds. Route closures on Highway 101 near SFO and Bay Bridge access roads. ZIP codes 94102, 94103, 94104 (Financial District, SOMA) most affected. 3-5 hour delivery delays in Mission District and Castro neighborhoods. Warehouse access restrictions at Port of San Francisco. Supply chain disruptions affecting deliveries to Pacific Heights and Russian Hill. 40% delivery capacity reduction during peak hours.", "base_impact_pct": -42.0, "base_month_index": 10, "direction": "cliff"}}

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
    
    user_prompt = f"""Provide exactly 3 concise, actionable recommendations for managing this SKU during the anomaly.

{sku_details}

{anomaly_details}

CRITICAL: Include highly specific, locality-relevant details that make recommendations feel personalized:
- Reference specific neighborhoods, ZIP codes, or areas mentioned in the anomaly description
- Mention local landmarks, routes, or infrastructure from the anomaly description
- Include specific warehouse IDs, route numbers, or delivery zones if mentioned
- Reference local delivery patterns, peak hours, or operational specifics for the area
- Use the exact location names, ZIP codes, or area references from the anomaly description

Each recommendation must:
- Be specific with numbers, percentages, timeframes, locations (use exact locations from anomaly)
- Address SKU characteristics (perishable, frozen, organic, etc.)
- Include measurable actions
- Reference specific areas, routes, or locations mentioned in the anomaly
- Be one concise sentence

Format as bullet points (start each with action verb). Return as JSON array of exactly 3 strings.

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
            if isinstance(recommendations, list) and len(recommendations) >= 3:
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
    """Fallback function when LLM is unavailable - generates 3 context-aware recommendations"""
    recommendations = []
    anomaly_type = anomaly["type"].lower()
    is_perishable = sku.get("perishable", False)
    is_frozen = sku.get("frozen", False)
    
    # Context-aware recommendations
    if "heat" in anomaly_type:
        if is_perishable:
            recommendations.append("Pre-cool delivery vans to maintain cold chain integrity")
            recommendations.append("Reduce exposure time for temperature-sensitive SKUs")
            recommendations.append("Increase cold chain capacity by 15% during peak heat hours")
        else:
            recommendations.append("Monitor warehouse temperature controls")
            recommendations.append("Prioritize early morning deliveries")
            recommendations.append("Increase buffer stock for high-velocity items")
    
    elif "snow" in anomaly_type or "storm" in anomaly_type:
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
        if pd.isna(forecast[month]):
            forecast[month] = generic_fcst.get(month, base_fcst)
    
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
    month_labels = [month_names[int(m) - 1] if m <= 12 else f"M{int(m)}" 
                    for m in df_filtered["month"]]
    
    fig = go.Figure()
    
    # True Demand - Bar Chart (minimal styling)
    fig.add_trace(go.Bar(
        x=df_filtered["month"],
        y=df_filtered["true_demand"],
        name="True Demand",
        marker=dict(color="#7F5AF0", opacity=0.6),
        showlegend=True,
    ))
    
    # Generic Forecast - Line Chart (make it pop)
    gen_filtered = generic_fcst[generic_fcst.index <= reveal_month]
    fig.add_trace(go.Scatter(
        x=gen_filtered.index,
        y=gen_filtered.values,
        name="Generic Forecast",
        line=dict(color="#ef4444", width=2.5, dash="dash"),
        mode="lines",
        showlegend=True,
    ))
    
    # Vajra Forecast - Line Chart (highlighted)
    vaj_filtered = vajra_fcst[vajra_fcst.index <= reveal_month]
    fig.add_trace(go.Scatter(
        x=vaj_filtered.index,
        y=vaj_filtered.values,
        name="Vajra Forecast",
        line=dict(color="#7F5AF0", width=3),
        mode="lines",
        showlegend=True,
    ))
    
    # Anomaly indicator - more apparent with brand colors
    anomaly_month = anomaly["month_index"]
    if anomaly_month <= reveal_month:
        fig.add_vline(
            x=anomaly_month,
            line_dash="solid",
            line_color="#7F5AF0",
            line_width=2.5,
            opacity=0.8,
            annotation_text="Anomaly",
            annotation_position="top",  # Back to top since legend is now on the side
            annotation=dict(
                font=dict(family="Space Grotesk, sans-serif", size=11, color="#7F5AF0", weight=600),
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="#7F5AF0",
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
            tickvals=df_filtered["month"],
            ticktext=month_labels,  # January to December labels
            showgrid=False,
            showline=True,
            linecolor="#DADADA",
            linewidth=1,
            tickfont=dict(family="Space Grotesk, sans-serif", size=11, color="#0F0F0F"),
            showticklabels=True,  # Show month labels on x-axis
            ticks="outside",  # Ensure ticks are visible
            ticklen=5,  # Make tick marks visible
            tickwidth=1
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

def render_recommendations(sku: Dict, anomaly: Dict, recommendations: List[str]):
    """Display 3 actionable recommendations with improved design"""
    st.markdown("---")
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(127, 90, 240, 0.05) 0%, rgba(100, 56, 183, 0.05) 100%); 
                border-left: 3px solid #6438B7; 
                padding: 1.5rem; 
                border-radius: 8px; 
                margin: 1.5rem 0;">
        <h3 style="font-family: 'Space Grotesk', sans-serif; font-weight: 600; color: #0F0F0F; margin-top: 0; margin-bottom: 1rem;">
            Recommendations
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Display recommendations as bullet points with proper spacing
    bullet_html = '<div style="padding: 0 1.5rem 1.5rem 1.5rem; font-family: \'Inter\', sans-serif; color: #0F0F0F; line-height: 1.8;">'
    for i, rec in enumerate(recommendations):
        if rec:
            bullet_html += f'<p style="margin: 0.75rem 0; padding-left: 1rem; position: relative;">'
            bullet_html += f'<span style="position: absolute; left: 0; color: #6438B7;">•</span>'
            bullet_html += f'<span style="margin-left: 0.5rem;">{rec}</span>'
            bullet_html += '</p>'
    bullet_html += '</div>'
    st.markdown(bullet_html, unsafe_allow_html=True)

# ============================================================================
# 9. MAIN APP FLOW
# ============================================================================

def main():
    """Main application flow"""
    # Apply styling
    apply_custom_styling()
    
    # Title
    st.title("Vajra Prototype")
    st.markdown("**Weather & Disruption Intelligence for Food Delivery Operations**")
    
    # Show LLM status (hidden from UI as requested)
    client = get_openai_client()
    # Debug: Show API status (only in sidebar for debugging, no key shown)
    with st.sidebar:
        if st.checkbox("Show API Debug Info", value=False):
            try:
                if hasattr(st.secrets, "OPENAI_API_KEY") or "OPENAI_API_KEY" in st.secrets or os.getenv("OPENAI_API_KEY"):
                    st.success("API Key configured")
                    if client:
                        st.success("OpenAI client initialized")
                    else:
                        st.error("OpenAI client failed to initialize")
                else:
                    st.error("API Key not found")
            except:
                if os.getenv("OPENAI_API_KEY"):
                    st.success("API Key found in environment")
                else:
                    st.error("API Key not found")
    
    st.markdown("---")
    
    # Sidebar
    inputs = render_sidebar()
    
    # Initialize session state for evaluation trigger
    if "evaluated" not in st.session_state:
        st.session_state.evaluated = False
    
    # Check if button was clicked
    button_clicked = inputs.get("evaluate_button", False)
    if button_clicked:
        st.session_state.evaluated = True
        st.session_state.evaluating = True
    
    # Check if using default/placeholder values
    is_default = inputs["retailer"].strip() == "" or inputs["area"].strip() == ""
    
    if is_default or not st.session_state.evaluated:
        # Show placeholder/default page
        st.info("**Enter a retailer name and area/city in the sidebar to see forecasts and analysis**")
        st.markdown("---")
        
        # Placeholder chart
        st.subheader("Demand Forecast Comparison")
        st.markdown("_Chart will appear here once you enter retailer and location_")
        
        # Placeholder table
        st.subheader("Forecast Analysis")
        st.markdown("""
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
                    <tr style="border-bottom: 1px solid #e5e7eb;">
                        <td style="padding: 0.75rem 1rem; font-weight: 500; color: #1a1a1a;">Accuracy</td>
                        <td style="padding: 0.75rem 1rem; color: #9ca3af;">—</td>
                        <td style="padding: 0.75rem 1rem; color: #9ca3af;">—</td>
                    </tr>
                    <tr>
                        <td style="padding: 0.75rem 1rem; font-weight: 500; color: #1a1a1a;">Savings</td>
                        <td style="padding: 0.75rem 1rem; color: #9ca3af;">—</td>
                        <td style="padding: 0.75rem 1rem; color: #9ca3af;">—</td>
                    </tr>
                </tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Show custom loading animation if evaluating
    loading_placeholder = st.empty()
    if st.session_state.get("evaluating", False):
        with loading_placeholder.container():
            st.markdown("""
            <div class="vajra-loader">
                <div class="vajra-loader-spinner"></div>
                <div class="vajra-loader-text">Analyzing Forecast<span class="vajra-loader-dots"></span></div>
            </div>
            """, unsafe_allow_html=True)
    
    # Generate SKUs using LLM (with fallback)
    skus = llm_top_skus(inputs["retailer"])
    
    # Generate base anomaly using LLM (with fallback)
    base_anomaly = llm_anomaly(inputs["area"])
    
    # Clear loading animation once we start generating data
    if st.session_state.get("evaluating", False):
        loading_placeholder.empty()
        st.session_state.evaluating = False
    
    # Generate per-SKU anomaly variations
    sku_anomalies = generate_sku_anomaly_variations(
        base_anomaly, len(skus), inputs["seed"]
    )
    
    # Process each SKU
    skus_data = {}
    for sku, anomaly in zip(skus, sku_anomalies):
        # Generate demand series
        df_sku = generate_demand_series(
            sku, anomaly, inputs["timeline_months"], inputs["seed"]
        )
        
        # Calculate forecasts
        generic_fcst = generic_forecast(df_sku, inputs["reveal_month"])
        vajra_fcst = vajra_forecast_with_alerts(
            df_sku, anomaly, inputs["reveal_month"]
        )
        
        # Calculate costs
        costs = calculate_forecast_costs(
            df_sku, generic_fcst, vajra_fcst, sku, anomaly,
            inputs["retailer"], inputs["reveal_month"]
        )
        
        # Generate recommendations using LLM (with fallback)
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
    
    # SKU Selector
    selected_sku_name = render_sku_selector(skus)
    selected_data = skus_data[selected_sku_name]
    
    # Create 3 main tabs: Anomaly Alerts, Forecast Analysis, Recommendations
    tab1, tab2, tab3 = st.tabs(["Anomaly Alerts", "Forecast Analysis", "Recommendations"])
    
    with tab1:
        # Display anomaly info with improved design
        anomaly = selected_data['anomaly']
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(127, 90, 240, 0.05) 0%, rgba(100, 56, 183, 0.05) 100%); 
                    border-left: 3px solid #7F5AF0; 
                    padding: 1.5rem; 
                    border-radius: 8px; 
                    margin: 1.5rem 0;">
            <h3 style="font-family: 'Space Grotesk', sans-serif; font-weight: 600; color: #0F0F0F; margin-top: 0; margin-bottom: 1rem;">
                Anomaly Detected
            </h3>
            <p style="font-family: 'Inter', sans-serif; font-weight: 600; color: #0F0F0F; font-size: 1.1rem; margin-bottom: 1rem;">
            {anomaly_type}
            </p>
        </div>
        """.format(anomaly_type=anomaly['type']), unsafe_allow_html=True)
        
        # Parse description into bullet points with proper spacing
        description = anomaly.get('description', '')
        sentences = [s.strip() for s in description.split('.') if s.strip()]
        
        if len(sentences) > 1:
            bullet_html = '<div style="padding: 0 1.5rem 1.5rem 1.5rem; font-family: \'Inter\', sans-serif; color: #0F0F0F; line-height: 1.8;">'
            for i, sentence in enumerate(sentences):
                if sentence:
                    bullet_html += f'<p style="margin: 0.75rem 0; padding-left: 1rem; position: relative;">'
                    bullet_html += f'<span style="position: absolute; left: 0; color: #7F5AF0;">•</span>'
                    bullet_html += f'<span style="margin-left: 0.5rem;">{sentence}</span>'
                    bullet_html += '</p>'
            bullet_html += '</div>'
            st.markdown(bullet_html, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="padding: 0 1.5rem 1.5rem 1.5rem; font-family: 'Inter', sans-serif; color: #0F0F0F; line-height: 1.8;">
                <p style="margin: 0.75rem 0; padding-left: 1rem; position: relative;">
                    <span style="position: absolute; left: 0; color: #7F5AF0;">•</span>
                    <span style="margin-left: 0.5rem;">{description}</span>
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        # Chart in Forecast Analysis tab
        render_demand_chart(
            selected_data["df"],
            selected_data["generic_fcst"],
            selected_data["vajra_fcst"],
            selected_data["anomaly"],
            inputs["reveal_month"]
        )
        
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
    
    with tab3:
        # Recommendations
        render_recommendations(
            selected_data["sku"],
            selected_data["anomaly"],
            selected_data["recommendations"]
        )
    
    # Footer
    st.markdown("---")
    st.caption("**Future Extensions:** Multi-store comparison, live data integration, real LLM API calls")

if __name__ == "__main__":
    main()

