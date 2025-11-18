# run_pricing_demo.py
from src.dynamic_pricing_engine import DynamicPricingEngine
from datetime import datetime, timedelta
import json

def run_demo():
    print("HOTEL DYNAMIC PRICING DEMO")
    print("=" * 50)
    
    # Initialize engine (ทั้งแบบมีและไม่มี ML model)
    try:
        # พยายามโหลด ML model ถ้ามี
        engine = DynamicPricingEngine(
            model_path="models/xgboost_baseline_model_business.pkl",
            features_path="data/processed/selected_features_list_business.csv"
        )
        print("Loaded with ML model")
    except:
        # Fallback ใช้ heuristic pricing
        engine = DynamicPricingEngine()
        print("Loaded with heuristic pricing (fallback)")
    
    # Test scenarios ต่างๆ
    scenarios = [
        {
            'name': 'WEEKEND_PEAK',
            'stay_date': datetime(2025, 12, 20),  # December weekend
            'room_type': 'Deluxe',
            'length_of_stay': 2,
            'guest_count': 2,
            'current_occupancy': 0.85
        },
        {
            'name': 'WEEKDAY_OFF_PEAK', 
            'stay_date': datetime(2025, 2, 5),   # February weekday
            'room_type': 'Standard',
            'length_of_stay': 5,
            'guest_count': 1,
            'current_occupancy': 0.45
        },
        {
            'name': 'BUSINESS_TRAVEL',
            'stay_date': datetime(2025, 9, 15),  # Shoulder season
            'room_type': 'Executive', 
            'length_of_stay': 3,
            'guest_count': 1,
            'current_occupancy': 0.70
        }
    ]
    
    for scenario in scenarios:
        print(f"\n SCENARIO: {scenario['name']}")
        print("-" * 30)
        
        result = engine.predict_optimal_price(scenario)
        
        # Display results
        pricing = result['pricing']
        market = result['market_analysis']
        insights = result['business_insights']
        
        print(f" Stay: {scenario['stay_date'].strftime('%Y-%m-%d')}")
        print(f" Room: {scenario['room_type']} ({scenario['length_of_stay']} nights)")
        print(f" Pricing: ${pricing['base_price']} → ${pricing['optimal_price']} ({pricing['price_change_percent']:+}%)")
        print(f" Strategy: {pricing['strategy'].replace('_', ' ').title()}")
        print(f" Market: {market['price_position']} (Rank {market['our_price_rank']}/{market['total_competitors']})")
        print(f" Forecast: {result['demand_forecast']:.0%} demand, {pricing['expected_occupancy']:.0%} expected occupancy")
        print(f" Top Insight: {insights['recommendations'][0] if insights['recommendations'] else 'No specific recommendation'}")
        
        print(f" Revenue Impact: {pricing['revenue_impact']}")

if __name__ == "__main__":
    run_demo()