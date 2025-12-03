# Metrics Tracking System

## Overview

The Vajra Prototype includes a comprehensive metrics tracking system that monitors user interactions and performance metrics. All data is stored locally in a SQLite database (`vajra_metrics.db`).

## Tracked Metrics

### 1. Button Clicks
- **Metric**: Number of times "Evaluate Forecast" button is clicked
- **Stored in**: `button_clicks` table
- **Includes**: Timestamp, session ID, retailer, area

### 2. Page Load Time
- **Metric**: Time for entire page to load when user enters URL
- **Stored in**: `page_loads` table
- **Includes**: Timestamp, session ID, load time (ms)

### 3. Evaluation Performance
- **Metric**: Time taken to generate all data when "Evaluate Forecast" is clicked
- **Stored in**: `evaluation_performance` table
- **Breakdown**:
  - **Total Time**: Complete evaluation time
  - **API Calls Time**: Time for all LLM API calls (SKU generation + anomaly generation)
  - **Chart Data Time**: Time to generate all chart data (demand series, forecasts, costs)
  - **Chart Render Time**: Time for chart to render on frontend
  - **SKU Generation Time**: Time for LLM to generate SKUs
  - **Anomaly Generation Time**: Time for LLM to generate anomaly
  - **Recommendations Time**: Time for LLM to generate recommendations
  - **Forecast Calculation Time**: Time to calculate generic and Vajra forecasts
  - **Cost Calculation Time**: Time to calculate costs and savings

## Accessing Admin Panel

### Method 1: Sidebar Button
1. Click the **"🔒 Admin Panel"** button in the sidebar
2. View all metrics and analytics

### Method 2: URL Parameter (if supported)
Add `?admin=true` to the URL

## Admin Panel Features

### Key Metrics Dashboard
- **Total Evaluations**: Count of "Evaluate Forecast" button clicks
- **Avg Page Load**: Average page load time across all sessions
- **Avg Total Time**: Average total evaluation time
- **Avg API Calls**: Average time for API calls

### Performance Breakdown
- **Table View**: Detailed breakdown of all timing metrics
- **Chart View**: Visual representation of API Calls, Chart Data, and Chart Render times

### Recent Evaluations
- Table showing last 20 evaluations with:
  - Timestamp
  - Retailer
  - Area
  - Total time
  - API calls time
  - Chart data time
  - Chart render time

## Database Schema

### button_clicks
```sql
CREATE TABLE button_clicks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    button_name TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    session_id TEXT,
    retailer TEXT,
    area TEXT
)
```

### page_loads
```sql
CREATE TABLE page_loads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    session_id TEXT,
    load_time_ms REAL,
    user_agent TEXT
)
```

### evaluation_performance
```sql
CREATE TABLE evaluation_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    session_id TEXT,
    retailer TEXT,
    area TEXT,
    total_time_ms REAL,
    api_calls_time_ms REAL,
    chart_data_time_ms REAL,
    chart_render_time_ms REAL,
    sku_generation_time_ms REAL,
    anomaly_generation_time_ms REAL,
    recommendations_time_ms REAL,
    forecast_calculation_time_ms REAL,
    cost_calculation_time_ms REAL
)
```

## Technical Implementation

### Metrics Module (`metrics.py`)
- SQLite database for local storage
- Session ID tracking
- Context managers for timing operations
- Helper functions for retrieving statistics

### Instrumentation Points
1. **Page Load**: Tracked at start of `main()` function
2. **Button Click**: Tracked when "Evaluate Forecast" is clicked
3. **API Calls**: Wrapped with `track_time()` context manager
4. **Chart Data**: Tracked around data generation loop
5. **Chart Render**: Tracked around `render_demand_chart()` call

## Data Privacy

- All metrics stored locally in SQLite database
- No external services required
- Database file (`vajra_metrics.db`) is excluded from git
- Session IDs are randomly generated

## Future Enhancements

Potential additions:
- Export metrics to CSV/JSON
- Time-series charts for performance trends
- Filtering by date range
- Comparison across different retailers/areas
- Integration with PostHog (optional external analytics)

