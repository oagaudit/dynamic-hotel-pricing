# Revised Streamlit app with Business Blue-White Theme and Free HTML Icons
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date, datetime, timedelta
import sys
import os
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pricing_pipeline import get_pipeline

# Page config
st.set_page_config(
    page_title="Dynamic Pricing Model for Hotel",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Business Theme CSS (Blue-White) with Font Awesome Icons
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 1.5rem;
    }

    .main-header {
        font-size: 2.2rem;
        font-weight: 600;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1.2rem;
    }

    h1, h2, h3, h4 {
        color: #1f77b4 !important;
    }

    .metric-card, .price-card {
        background: #ffffff;
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid #c9d9e9;
        color: #003366;
        text-align: center;
        margin-bottom: 0.8rem;
    }

    .stAlert {
        background-color: #e9f1fa !important;
        color: #003366 !important;
        border-left: 4px solid #1f77b4 !important;
    }

    [data-testid="stSidebar"] {
        background-color: #f7faff;
    }

    .stButton>button {
        background-color: #1f77b4 !important;
        color: white !important;
        border-radius: 6px !important;
        border: none !important;
        padding: 0.5rem 1rem !important;
    }

    .stButton>button:hover {
        background-color: #155a8a !important;
    }
    
    .stMetric label, .stMetric div {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
    }
    
    .scenario-card {
        background: #f8fbff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    
    .historical-chart {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e1e8f0;
    }
    
    .icon-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 1rem;
    }
    
    .icon-header i {
        font-size: 1.5em;
        color: #1f77b4;
    }
    
    .booking-summary {
        background-color: #e9f1fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    
    .booking-summary i {
        color: #1f77b4;
        margin-right: 8px;
        width: 16px;
    }
    
    .info-box {
        background-color: #e9f1fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    
    .info-box i {
        color: #1f77b4;
        margin-right: 8px;
    }
    
     
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for historical data and scenarios
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

if 'scenarios' not in st.session_state:
    st.session_state.scenarios = []

# 🔧 ADD: ฟังก์ชัน get_season ที่หายไป
def get_season(month):
    """Get season from month"""
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Fall"

def main():
    st.markdown('<div class="main-header"><i class="fas fa-chart-line"></i>Dynamic Pricing Model for Hotel</div>', unsafe_allow_html=True)
    st.markdown("### Get AI-powered price recommendations with market insights")

    # Sidebar
    with st.sidebar:
        st.markdown('<div class="icon-header"><i class="fas fa-sliders-h"></i><h3>Booking Parameters</h3></div>', unsafe_allow_html=True)
        
        # Real-time preview section
        st.markdown('<div class="icon-header"><i class="fas fa-search"></i><h4>Quick Preview</h4></div>', unsafe_allow_html=True)
        
        room_type = st.selectbox(
            "Select Room Type",
            ["Standard", "Deluxe", "Suite", "Executive"],
            index=1
        )

        check_in_date = st.date_input(
            "Check-in Date",
            value=date.today() + timedelta(days=30),
            min_value=date.today(),
            max_value=date.today() + timedelta(days=365),
        )

        length_of_stay = st.slider(
            "Number of Nights", 1, 14, 3
        )

        guest_count = st.number_input(
            "Number of Guests", min_value=1, max_value=6, value=2
        )

        check_out_date = check_in_date + timedelta(days=length_of_stay)
        
        # Quick Estimate Button (แทน cache)
        st.markdown("---")
        st.markdown('<div class="icon-header"><i class="fas fa-bolt"></i><h4>Quick Estimate</h4></div>', unsafe_allow_html=True)
        
        quick_estimate_clicked = st.button(
            "🚀 Get Quick Price Estimate",
            type="secondary",
            use_container_width=True,
            key="quick_estimate"
        )
        
        # Calculate and display quick estimate when button is clicked
        quick_price = None
        if quick_estimate_clicked:
            with st.spinner("Calculating quick estimate..."):
                try:
                    pipeline = get_pipeline()
                    quick_input = {
                        'room_type': room_type,
                        'check_in_date': check_in_date.isoformat(),
                        'length_of_stay': length_of_stay,
                        'guest_count': guest_count
                    }
                    quick_price = pipeline.get_quick_estimate(quick_input)
                    # บันทึกผลลัพธ์ใน session state เพื่อแสดงผล
                    st.session_state.last_quick_price = quick_price
                    st.session_state.last_quick_params = quick_input
                except Exception as e:
                    st.error(f"Quick estimate failed: {e}")
        
        # แสดงผล quick estimate ถ้ามี
        if hasattr(st.session_state, 'last_quick_price'):
            quick_price = st.session_state.last_quick_price
            price_display = f"${quick_price:.2f}"
            price_html = f"""
            <div class="quick-estimate-result">
                <i class="fas fa-dollar-sign"></i> Quick Estimate: {price_display}
            </div>
            """
            st.markdown(price_html, unsafe_allow_html=True)
        else:
            placeholder_html = """
            <div class="quick-estimate-placeholder">
                Click "Get Quick Price Estimate" to see price
            </div>
            """
            st.markdown(placeholder_html, unsafe_allow_html=True)

        # Booking Summary (ไม่มี quick estimate แล้ว)
        booking_summary_html = f"""
        <div class="booking-summary">
            <p><strong><i class="fas fa-calendar-alt"></i> Booking Summary:</strong></p>
            <p><i class="fas fa-sign-in-alt"></i> Check-in: {check_in_date.strftime('%Y-%m-%d')}</p>
            <p><i class="fas fa-sign-out-alt"></i> Check-out: {check_out_date.strftime('%Y-%m-%d')}</p>
            <p><i class="fas fa-moon"></i> Total nights: {length_of_stay}</p>
            <p><i class="fas fa-users"></i> Guests: {guest_count}</p>
        </div>
        """
        st.markdown(booking_summary_html, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="icon-header"><i class="fas fa-bullseye"></i><h3>Analysis Actions</h3></div>', unsafe_allow_html=True)
        
        analyze_clicked = st.button(
            "🤖 Get Full AI Price Recommendation",
            type="primary",
            use_container_width=True,
            key="analyze_main"
        )
        
        # Scenario management
        st.markdown('<div class="icon-header"><i class="fas fa-clipboard-list"></i><h4>Scenario Management</h4></div>', unsafe_allow_html=True)
        scenario_name = st.text_input("Scenario Name", placeholder="e.g., Weekend Special")
        
        if st.button("💾 Save Current as Scenario", use_container_width=True, key="save_scenario"):
            if scenario_name:
                save_scenario(room_type, check_in_date, length_of_stay, guest_count, scenario_name)
                st.success(f"Scenario '{scenario_name}' saved!")
            else:
                st.warning("Please enter a scenario name")

    # Main content - Tabs for different features
    tab1, tab2, tab3, tab4 = st.tabs([
        "Dashboard", 
        "Market Analysis", 
        "Scenarios", 
        "Historical Data"
    ])

    with tab1:
        display_dashboard(room_type, length_of_stay, guest_count, check_in_date, analyze_clicked)

    with tab2:
        display_market_analysis()

    with tab3:
        display_scenarios_comparison()

    with tab4:
        display_historical_data()

def display_dashboard(room_type, length_of_stay, guest_count, check_in_date, analyze_clicked):
    """Main dashboard tab"""
    col1, col2 = st.columns([2, 1])

    with col1:
        # Real-time metrics
        st.markdown('<div class="icon-header"><i class="fas fa-info-circle"></i><h4>Current Selection</h4></div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Room Type", room_type)
        c2.metric("Nights", length_of_stay)
        c3.metric("Guests", guest_count)
        c4.metric("Season", get_season(check_in_date.month))  # ✅ ใช้ฟังก์ชันที่เพิ่มมาแล้ว
        
        if hasattr(st.session_state, 'last_quick_price'):
            quick_price = st.session_state.last_quick_price
            price_display = f"${quick_price:.2f}"
            quick_estimate_html = f"""
            <div style="background-color: #f0f8ff; padding: 1rem; border-radius: 8px; border-left: 4px solid #1f77b4; margin-top: 1rem;">
                <p><strong><i class="fas fa-dollar-sign"></i> Quick Estimate: {price_display}</strong></p>
            </div>
            """
            st.markdown(quick_estimate_html, unsafe_allow_html=True)
         
    with col2:
        st.markdown('<div class="icon-header"><i class="fas fa-info-circle"></i><h3>How It Works</h3></div>', unsafe_allow_html=True)
        
        how_it_works_html = """
        <div class="info-box">
            <p><strong>Quick Estimate:</strong></p>
            <p><i class="fas fa-bolt"></i> Instant price calculation using ML model</p>
            <p><i class="fas fa-rocket"></i> Fast response without full analysis</p>
            <p><i class="fas fa-chart-line"></i> Perfect for quick comparisons</p>
            <br>
            <p><strong>Full Analysis:</strong></p>
            <p><i class="fas fa-robot"></i> Complete AI-powered analysis</p>
            <p><i class="fas fa-balance-scale"></i> Competitor intelligence</p>
            <p><i class="fas fa-chart-bar"></i> Market insights & strategy</p>
        </div>
        """
        st.markdown(how_it_works_html, unsafe_allow_html=True)

    if analyze_clicked:
        with st.spinner("🤖 AI is analyzing market conditions, competitors, and pricing strategies..."):
            try:
                user_input = {
                    'room_type': room_type,
                    'check_in_date': check_in_date.isoformat(),
                    'length_of_stay': length_of_stay,
                    'guest_count': guest_count
                }

                pipeline = get_pipeline()
                result = pipeline.get_price_recommendation(user_input)

                if result['success']:
                    # Save to history
                    save_to_history(result, user_input)
                    display_complete_analysis(result)
                else:
                    st.error(f"Analysis failed: {result.get('error', 'Unknown error')}")
            except Exception as e:
                st.error(f"System error: {str(e)}")


def display_market_analysis():
    """Market analysis tab with historical trends"""
    st.markdown('<div class="icon-header"><i class="fas fa-chart-bar"></i><h3>Market Analysis</h3></div>', unsafe_allow_html=True)
    
    try:
        # ✅ โหลดข้อมูลจาก CSV
        sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
        from config.paths import CLEANED_HOTEL_FILE
        
        df = pd.read_csv(CLEANED_HOTEL_FILE)
        
        # ✅ สร้างคอลัมน์วันที่จากข้อมูลที่มี
        df['date'] = pd.to_datetime({
            'year': df['arrival_date_year'],
            'month': df['arrival_date_month'].map({
                'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
            }),
            'day': df['arrival_date_day_of_month']
        })
        
        # ✅ เปลี่ยนเป็นเลือกช่วงเวลาประวัติศาสตร์
        col1, col2 = st.columns([3, 1])
        with col2:
            time_range = st.selectbox(
                "Time Range",
                ["Last 30 Days", "Last 90 Days", "Last 1 Year", "All Data"],
                index=3  # ✅ เลือก All Data เป็น default
            )
        
        # ✅ กรองข้อมูลตามช่วงเวลาประวัติศาสตร์
        if time_range == "Last 30 Days":
            end_date = df['date'].max()
            start_date = end_date - timedelta(days=30)
            filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        elif time_range == "Last 90 Days":
            end_date = df['date'].max()
            start_date = end_date - timedelta(days=90)
            filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        elif time_range == "Last 1 Year":
            end_date = df['date'].max()
            start_date = end_date - timedelta(days=365)
            filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        else:  # All Data
            filtered_df = df
        
        if filtered_df.empty:
            st.error("❌ No data found for the selected time range.")
            st.info(f"Data available from {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
            return
        
        # ✅ คำนวณข้อมูลรายวัน
        daily_data = filtered_df.groupby('date').agg({
            'adr': 'mean',  # ✅ ราคาเรา
            'total_of_special_requests': 'mean',  # ✅ ใช้เป็น demand indicator
            'is_canceled': 'mean'  # ✅ cancellation rate
        }).reset_index()
        
        # ✅ คำนวณ competitor price (สมมติ +10% จากราคาเรา)
        daily_data['competitor_avg_price'] = daily_data['adr'] * 1.1
        
        # ✅ คำนวณ demand index จากข้อมูลที่มี
        daily_data['demand_index'] = (daily_data['total_of_special_requests'] - daily_data['total_of_special_requests'].min()) / (daily_data['total_of_special_requests'].max() - daily_data['total_of_special_requests'].min())
        
        # ✅ กรองข้อมูลที่มีค่าไม่ null
        daily_data = daily_data.dropna()
        
        if daily_data.empty:
            st.error("❌ No valid data available after filtering.")
            return

    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")
        return

    # ✅ Price trend chart จากข้อมูลจริง
    st.markdown(f'<div class="icon-header"><i class="fas fa-chart-line"></i><h4>Historical Price Trends ({time_range})</h4></div>', unsafe_allow_html=True)
    
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=daily_data['date'], 
        y=daily_data['adr'], 
        name='Our Price (ADR)', 
        line=dict(color='#1f77b4'),
        hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
    ))
    fig_trend.add_trace(go.Scatter(
        x=daily_data['date'], 
        y=daily_data['competitor_avg_price'], 
        name='Competitor Avg Price (Estimated)', 
        line=dict(color='#ff7f0e'),
        hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
    ))
    
    fig_trend.update_layout(
        title=f"Historical Price Trends - {time_range}",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # ✅ Market metrics summary
    st.markdown("#### 📊 Historical Market Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_our_price = daily_data['adr'].mean()
        st.metric("Avg Our Price", f"${avg_our_price:.2f}")
    
    with col2:
        avg_competitor_price = daily_data['competitor_avg_price'].mean()
        st.metric("Avg Competitor Price", f"${avg_competitor_price:.2f}")
    
    with col3:
        price_diff = avg_our_price - avg_competitor_price
        diff_percent = (price_diff / avg_competitor_price) * 100
        st.metric("Price Difference", f"{diff_percent:+.1f}%", delta=f"{price_diff:+.2f}")
    
    with col4:
        cancellation_rate = filtered_df['is_canceled'].mean() * 100
        st.metric("Cancellation Rate", f"{cancellation_rate:.1f}%")
    
    # ✅ Demand index chart
    st.markdown(f'<div class="icon-header"><i class="fas fa-chart-area"></i><h4>Market Demand Trends ({time_range})</h4></div>', unsafe_allow_html=True)
    
    fig_demand = go.Figure()
    fig_demand.add_trace(go.Scatter(
        x=daily_data['date'], 
        y=daily_data['demand_index'], 
        name='Demand Index', 
        line=dict(color='#2ca02c'), 
        fill='tozeroy'
    ))
    
    fig_demand.update_layout(
        title=f"Market Demand Trends - {time_range}",
        xaxis_title="Date",
        yaxis_title="Demand Index",
        height=300,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_demand, use_container_width=True)
    
    # ✅ Additional insights
    st.markdown("#### 💡 Historical Insights")
    
    if len(daily_data) > 1:
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            # Price volatility
            price_volatility = daily_data['adr'].std()
            st.metric("Price Volatility", f"${price_volatility:.2f}")
            
            # Trend analysis
            price_trend = (daily_data['adr'].iloc[-1] - daily_data['adr'].iloc[0]) / daily_data['adr'].iloc[0] * 100
            st.metric("Price Trend", f"{price_trend:+.1f}%")
        
        with insights_col2:
            # Booking quality
            avg_special_requests = daily_data['total_of_special_requests'].mean()
            st.metric("Avg Special Requests", f"{avg_special_requests:.1f}")
            
            # Data period
            date_range = f"{daily_data['date'].min().strftime('%Y-%m')} to {daily_data['date'].max().strftime('%Y-%m')}"
            st.metric("Data Period", date_range)



def display_scenarios_comparison():
    """Scenario comparison tab"""
    st.markdown('<div class="icon-header"><i class="fas fa-sync-alt"></i><h3>Scenario Comparison</h3></div>', unsafe_allow_html=True)
    
    if not st.session_state.scenarios:
        st.info("No scenarios saved yet. Save some scenarios from the main dashboard to compare them here.")
        return
    
    # Display saved scenarios
    for i, scenario in enumerate(st.session_state.scenarios):
        with st.container():
            st.markdown(f"""
            <div class="scenario-card">
                <h4><i class="fas fa-clipboard-list"></i> {scenario['name']}</h4>
                <p><i class="fas fa-cog"></i> <strong>Parameters:</strong> {scenario['room_type']}, {scenario['nights']} nights, {scenario['guests']} guests</p>
                <p><i class="fas fa-calendar"></i> <strong>Date:</strong> {scenario['check_in']}</p>
                <p><i class="fas fa-dollar-sign"></i> <strong>Quick Estimate:</strong> ${scenario.get('quick_estimate', 0):.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button(f"Analyze", key=f"analyze_{i}"):
                    st.session_state.auto_analyze = scenario
                    st.rerun()
            with col2:
                if st.button(f"Delete", key=f"delete_{i}"):
                    st.session_state.scenarios.pop(i)
                    st.rerun()
    
    # Comparison chart if we have multiple scenarios
    if len(st.session_state.scenarios) > 1:
        st.markdown('<div class="icon-header"><i class="fas fa-chart-bar"></i><h4>Scenario Comparison</h4></div>', unsafe_allow_html=True)
        
        scenario_names = [s['name'] for s in st.session_state.scenarios]
        scenario_prices = [s.get('quick_estimate', 0) for s in st.session_state.scenarios]
        
        fig = go.Figure(go.Bar(
            x=scenario_names,
            y=scenario_prices,
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(scenario_names)]
        ))
        
        fig.update_layout(
            title="Quick Price Estimates by Scenario",
            xaxis_title="Scenarios",
            yaxis_title="Price ($)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

def display_historical_data():
    """Historical analysis data tab"""
    st.markdown('<div class="icon-header"><i class="fas fa-history"></i><h3>Historical Analysis Data</h3></div>', unsafe_allow_html=True)
    
    if not st.session_state.analysis_history:
        st.info("No historical analysis data yet. Perform some analyses to see historical trends here.")
        return
    
    # Display recent analyses
    st.markdown('<div class="icon-header"><i class="fas fa-clock"></i><h4>Recent Analyses</h4></div>', unsafe_allow_html=True)
    for i, analysis in enumerate(st.session_state.analysis_history[-5:]):
        with st.expander(f"Analysis {i+1} - {analysis['timestamp']}"):
            col1, col2, col3 = st.columns(3)
            col1.metric("Base Price", f"${analysis.get('base_price', 0):.2f}")
            col2.metric("Optimal Price", f"${analysis.get('optimal_price', 0):.2f}")
            col3.metric("Strategy", analysis.get('strategy', 'N/A'))
    
    # Historical trends chart
    st.markdown('<div class="icon-header"><i class="fas fa-chart-line"></i><h4>Historical Price Trends</h4></div>', unsafe_allow_html=True)
    
    if len(st.session_state.analysis_history) > 1:
        history_df = pd.DataFrame(st.session_state.analysis_history)
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=history_df['timestamp'], y=history_df['base_price'], 
                                name='Base Price', line=dict(color='#1f77b4')))
        fig.add_trace(go.Scatter(x=history_df['timestamp'], y=history_df['optimal_price'], 
                                name='Optimal Price', line=dict(color='#2ca02c')))
        
        fig.update_layout(
            title="Historical Price Recommendations",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

def display_complete_analysis(result: dict):
    """Display the complete analysis results"""
    st.success("AI Analysis Complete!")
    
    # Export functionality
    col1, col2, col3 = st.columns([2, 1, 1])
    with col2:
        if st.button("Export PDF Report", use_container_width=True, key="export_pdf"):
            export_pdf_report(result)
    with col3:
        if st.button("Export Data", use_container_width=True, key="export_data"):
            export_analysis_data(result)

    st.markdown("---")
    st.markdown('<div class="icon-header"><i class="fas fa-bullseye"></i><h3>AI Price Recommendation</h3></div>', unsafe_allow_html=True)

    # Pricing information
    base_price = result.get('base_price', 0)
    optimal_price = result.get('optimal_price', 0)
    price_difference = result.get('price_difference', 0)
    price_difference_percent = result.get('price_difference_percent', 0)
    
    market_analysis = result.get('market_analysis', {})
    competitor_price = market_analysis.get('competitor_avg_price', 0)

    # Display pricing metrics
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("ML Base Price", f"${base_price:.2f}")
    p2.metric("Optimal Price", f"${optimal_price:.2f}", 
              delta=f"{price_difference:+.2f}",
              delta_color="inverse" if price_difference < 0 else "normal")
    
    strategy = result.get('strategy', 'revenue_maximization')
    strategy_display = strategy.replace('_', ' ').title()
    p3.metric("Pricing Strategy", strategy_display)
    
    p4.metric("Difference", f"{price_difference_percent:+.1f}%")

    # Explanation
    difference_explanation = result.get('difference_explanation', '')
    if difference_explanation:
        st.info(difference_explanation)

    # Market Intelligence
    st.markdown("---")
    st.markdown('<div class="icon-header"><i class="fas fa-chart-line"></i><h3>Market Intelligence</h3></div>', unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Market Position", market_analysis.get('price_position', 'N/A'))
    c2.metric("Competitiveness", market_analysis.get('competitiveness', 'N/A'))
    c3.metric("Your Rank", f"{market_analysis.get('our_price_rank', 0)}/{market_analysis.get('total_competitors', 0)}")
    
    price_gap_percent = market_analysis.get('price_gap_percent', 0)
    price_gap_label = f"{price_gap_percent:+.1f}%"
    c4.metric("Price Gap", price_gap_label)

    # Price Comparison
    st.markdown('<div class="icon-header"><i class="fas fa-balance-scale"></i><h4>Price Comparison</h4></div>', unsafe_allow_html=True)
    comp_col1, comp_col2, comp_col3 = st.columns(3)
    
    with comp_col1:
        st.metric("Our Optimal Price", f"${optimal_price:.2f}")
    
    with comp_col2:
        competitor_display = f"${competitor_price:.2f}" if competitor_price > 0 else "No data"
        st.metric("Competitor Average", competitor_display)
    
    with comp_col3:
        if competitor_price > 0:
            price_diff_vs_competitor = optimal_price - competitor_price
            diff_percent_vs_competitor = (price_diff_vs_competitor / competitor_price * 100)
            
            delta_color = "inverse" if price_diff_vs_competitor < 0 else "normal"
            
            st.metric("Vs Competitors", f"{diff_percent_vs_competitor:+.1f}%", 
                     delta=f"{price_diff_vs_competitor:+.2f}",
                     delta_color=delta_color)
        else:
            st.metric("Vs Competitors", "N/A", delta="No competitor data")

    # Price Drivers Analysis
    st.markdown("---")
    st.markdown('<div class="icon-header"><i class="fas fa-search"></i><h3>Price Drivers Analysis</h3></div>', unsafe_allow_html=True)

    if 'feature_impacts' in result and result['feature_impacts']:
        st.markdown('<div class="icon-header"><i class="fas fa-list-ol"></i><h4>Top Influencing Factors</h4></div>', unsafe_allow_html=True)
        for i, feature in enumerate(result['feature_impacts'][:5], 1):
            shap_value = feature['shap_value']
            impact_icon = "📈" if shap_value > 0 else "📉"
            st.write(f"{i}. {impact_icon} {feature['feature']}: {shap_value:+.3f}")

        # SHAP Visualization
        if len(result['feature_impacts']) > 0:
            fig = create_shap_plot(result['feature_impacts'])
            st.plotly_chart(fig, use_container_width=True)

    # AI Business Insights
    st.markdown("---")
    st.markdown('<div class="icon-header"><i class="fas fa-robot"></i><h3>AI Business Insights</h3></div>', unsafe_allow_html=True)

    if 'ai_insights' in result and result['ai_insights']['success']:
        insights = result['ai_insights']['recommendations']
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "Summary", 
            "Price Analysis", 
            "Strategy", 
            "Risks"
        ])
        
        with tab1:
            st.markdown('<div class="icon-header"><i class="fas fa-file-alt"></i><h4>Executive Summary</h4></div>', unsafe_allow_html=True)
            summary_text = insights.get('summary', 'No summary available')
            st.write(summary_text)
            
        with tab2:
            st.markdown('<div class="icon-header"><i class="fas fa-dollar-sign"></i><h4>Price Analysis</h4></div>', unsafe_allow_html=True)
            price_analysis = insights.get('price_analysis', 'No price analysis available')
            st.write(price_analysis)
            
        with tab3:
            st.markdown('<div class="icon-header"><i class="fas fa-chess-knight"></i><h4>Strategic Recommendations</h4></div>', unsafe_allow_html=True)
            strategy_text = insights.get('strategy', 'No strategy recommendations available')
            st.write(strategy_text)
            
        with tab4:
            st.markdown('<div class="icon-header"><i class="fas fa-exclamation-triangle"></i><h4>Risk Considerations</h4></div>', unsafe_allow_html=True)
            risks = insights.get('risk_considerations', 'No risk analysis available')
            st.write(risks)

def create_shap_plot(feature_impacts):
    """Create SHAP values visualization"""
    features = [f['feature'] for f in feature_impacts[:10]]
    values = [f['shap_value'] for f in feature_impacts[:10]]
    colors = ['#ff4b4b' if x < 0 else '#0068c9' for x in values]
    
    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation='h',
        marker_color=colors
    ))
    
    fig.update_layout(
        title="Top Feature Impacts (SHAP Values)",
        xaxis_title="Impact on Price",
        yaxis_title="Features",
        showlegend=False,
        height=400
    )
    
    return fig

def save_to_history(result, user_input):
    """Save analysis to history"""
    history_entry = {
        'timestamp': datetime.now().isoformat(),
        'base_price': result.get('base_price', 0),
        'optimal_price': result.get('optimal_price', 0),
        'strategy': result.get('strategy', ''),
        'room_type': user_input.get('room_type', ''),
        'length_of_stay': user_input.get('length_of_stay', 0),
        'user_input': user_input
    }
    st.session_state.analysis_history.append(history_entry)

def save_scenario(room_type, check_in_date, length_of_stay, guest_count, scenario_name):
    """Save current parameters as a scenario"""
    scenario = {
        'name': scenario_name,
        'room_type': room_type,
        'check_in': check_in_date.isoformat(),
        'nights': length_of_stay,
        'guests': guest_count,
        'timestamp': datetime.now().isoformat()
    }
    
    # Get quick estimate for this scenario
    try:
        pipeline = get_pipeline()
        quick_input = {
            'room_type': room_type,
            'check_in_date': check_in_date.isoformat(),
            'length_of_stay': length_of_stay,
            'guest_count': guest_count
        }
        quick_price = pipeline.get_quick_estimate(quick_input)
        scenario['quick_estimate'] = quick_price
    except:
        scenario['quick_estimate'] = 0
    
    st.session_state.scenarios.append(scenario)

def export_pdf_report(result):
    """Export analysis as PDF report"""
    st.info("PDF export functionality would be implemented here with a proper PDF generation library")

def export_analysis_data(result):
    """Export analysis data as JSON/CSV"""
    # Export as JSON
    json_data = json.dumps(result, indent=2, default=str)
    st.download_button(
        label="Download JSON",
        data=json_data,
        file_name=f"pricing_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

if __name__ == "__main__":
    main()