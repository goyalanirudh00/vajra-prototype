"""
Metrics Tracking Module for Vajra Prototype
Tracks user interactions, performance metrics, and system analytics
"""

import sqlite3
import json
import time
from datetime import datetime
from typing import Dict, Optional, List
import os
from contextlib import contextmanager

# Database path
DB_PATH = "vajra_metrics.db"

def init_database():
    """Initialize the metrics database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Button clicks table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS button_clicks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            button_name TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            session_id TEXT,
            retailer TEXT,
            area TEXT
        )
    """)
    
    # Page load times table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS page_loads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            session_id TEXT,
            load_time_ms REAL,
            user_agent TEXT
        )
    """)
    
    # Evaluation performance table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS evaluation_performance (
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
    """)
    
    conn.commit()
    conn.close()

def get_session_id() -> str:
    """Get or create a session ID"""
    try:
        import streamlit as st
        if "session_id" not in st.session_state:
            st.session_state.session_id = f"{int(time.time() * 1000)}_{os.urandom(4).hex()}"
        return st.session_state.session_id
    except:
        # Fallback if streamlit not available
        return f"{int(time.time() * 1000)}_{os.urandom(4).hex()}"

def track_button_click(button_name: str, retailer: str = "", area: str = ""):
    """Track a button click"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        session_id = get_session_id()
        
        cursor.execute("""
            INSERT INTO button_clicks (button_name, session_id, retailer, area)
            VALUES (?, ?, ?, ?)
        """, (button_name, session_id, retailer, area))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error tracking button click: {e}")

def track_page_load(load_time_ms: float):
    """Track page load time"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        session_id = get_session_id()
        
        cursor.execute("""
            INSERT INTO page_loads (session_id, load_time_ms)
            VALUES (?, ?)
        """, (session_id, load_time_ms))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error tracking page load: {e}")

def track_evaluation_performance(metrics: Dict):
    """Track evaluation performance metrics"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        session_id = get_session_id()
        
        cursor.execute("""
            INSERT INTO evaluation_performance (
                session_id, retailer, area, total_time_ms,
                api_calls_time_ms, chart_data_time_ms, chart_render_time_ms,
                sku_generation_time_ms, anomaly_generation_time_ms,
                recommendations_time_ms, forecast_calculation_time_ms,
                cost_calculation_time_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            metrics.get("retailer", ""),
            metrics.get("area", ""),
            metrics.get("total_time_ms", 0),
            metrics.get("api_calls_time_ms", 0),
            metrics.get("chart_data_time_ms", 0),
            metrics.get("chart_render_time_ms", 0),
            metrics.get("sku_generation_time_ms", 0),
            metrics.get("anomaly_generation_time_ms", 0),
            metrics.get("recommendations_time_ms", 0),
            metrics.get("forecast_calculation_time_ms", 0),
            metrics.get("cost_calculation_time_ms", 0)
        ))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error tracking evaluation performance: {e}")

@contextmanager
def track_time(metric_name: str, metrics_dict: Dict):
    """Context manager to track time for a specific operation"""
    start_time = time.time()
    try:
        yield
    finally:
        elapsed_ms = (time.time() - start_time) * 1000
        metrics_dict[metric_name] = elapsed_ms

def get_button_click_count(button_name: str = "Evaluate Forecast") -> int:
    """Get total count of button clicks"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM button_clicks WHERE button_name = ?
        """, (button_name,))
        count = cursor.fetchone()[0]
        conn.close()
        return count
    except Exception as e:
        print(f"Error getting button click count: {e}")
        return 0

def get_average_page_load_time() -> float:
    """Get average page load time in milliseconds"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT AVG(load_time_ms) FROM page_loads
        """)
        result = cursor.fetchone()[0]
        conn.close()
        return result if result else 0.0
    except Exception as e:
        print(f"Error getting average page load time: {e}")
        return 0.0

def get_evaluation_stats() -> Dict:
    """Get statistics about evaluation performance"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get average times
        cursor.execute("""
            SELECT 
                AVG(total_time_ms) as avg_total,
                AVG(api_calls_time_ms) as avg_api,
                AVG(chart_data_time_ms) as avg_chart_data,
                AVG(chart_render_time_ms) as avg_chart_render,
                AVG(sku_generation_time_ms) as avg_sku,
                AVG(anomaly_generation_time_ms) as avg_anomaly,
                AVG(recommendations_time_ms) as avg_recommendations,
                AVG(forecast_calculation_time_ms) as avg_forecast,
                AVG(cost_calculation_time_ms) as avg_cost,
                COUNT(*) as total_evaluations
            FROM evaluation_performance
        """)
        
        row = cursor.fetchone()
        conn.close()
        
        if row and row[9] > 0:  # total_evaluations > 0
            return {
                "total_evaluations": int(row[9]),
                "avg_total_time_ms": row[0] or 0.0,
                "avg_api_calls_time_ms": row[1] or 0.0,
                "avg_chart_data_time_ms": row[2] or 0.0,
                "avg_chart_render_time_ms": row[3] or 0.0,
                "avg_sku_generation_time_ms": row[4] or 0.0,
                "avg_anomaly_generation_time_ms": row[5] or 0.0,
                "avg_recommendations_time_ms": row[6] or 0.0,
                "avg_forecast_calculation_time_ms": row[7] or 0.0,
                "avg_cost_calculation_time_ms": row[8] or 0.0
            }
        else:
            return {
                "total_evaluations": 0,
                "avg_total_time_ms": 0.0,
                "avg_api_calls_time_ms": 0.0,
                "avg_chart_data_time_ms": 0.0,
                "avg_chart_render_time_ms": 0.0,
                "avg_sku_generation_time_ms": 0.0,
                "avg_anomaly_generation_time_ms": 0.0,
                "avg_recommendations_time_ms": 0.0,
                "avg_forecast_calculation_time_ms": 0.0,
                "avg_cost_calculation_time_ms": 0.0
            }
    except Exception as e:
        print(f"Error getting evaluation stats: {e}")
        return {
            "total_evaluations": 0,
            "avg_total_time_ms": 0.0,
            "avg_api_calls_time_ms": 0.0,
            "avg_chart_data_time_ms": 0.0,
            "avg_chart_render_time_ms": 0.0,
            "avg_sku_generation_time_ms": 0.0,
            "avg_anomaly_generation_time_ms": 0.0,
            "avg_recommendations_time_ms": 0.0,
            "avg_forecast_calculation_time_ms": 0.0,
            "avg_cost_calculation_time_ms": 0.0
        }

def get_recent_evaluations(limit: int = 10) -> List[Dict]:
    """Get recent evaluation records"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                timestamp, retailer, area, total_time_ms,
                api_calls_time_ms, chart_data_time_ms, chart_render_time_ms
            FROM evaluation_performance
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                "timestamp": row[0],
                "retailer": row[1],
                "area": row[2],
                "total_time_ms": row[3],
                "api_calls_time_ms": row[4],
                "chart_data_time_ms": row[5],
                "chart_render_time_ms": row[6]
            }
            for row in rows
        ]
    except Exception as e:
        print(f"Error getting recent evaluations: {e}")
        return []

# Initialize database on import (will be called from app.py)
# init_database() is called explicitly in main() to ensure streamlit is available

