# src/feature_mapper.py
import math
from datetime import datetime, date
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class FeatureMapper:
    def __init__(self):
        self.room_type_mapping = {
            'Standard': {'reserved_room_type_A': 0, 'hotel_type_encoded': 0},
            'Deluxe': {'reserved_room_type_A': 1, 'hotel_type_encoded': 1},
            'Suite': {'reserved_room_type_A': 0, 'hotel_type_encoded': 2},
            'Executive': {'reserved_room_type_A': 0, 'hotel_type_encoded': 1}
        }
    
    def map_user_input_to_features(self, user_input: Dict) -> Dict[str, float]:
        """
        Convert user-friendly input to model features
        """
        features = {}
        
        # 1. Room type mapping
        room_type = user_input.get('room_type', 'Standard')
        if room_type in self.room_type_mapping:
            features.update(self.room_type_mapping[room_type])
        
        # 2. Date processing
        check_in_date = user_input.get('check_in_date')
        if check_in_date:
            if isinstance(check_in_date, str):
                check_in_date = datetime.strptime(check_in_date, '%Y-%m-%d').date()
            
            # Seasonal encoding (arrival_month_sin)
            month = check_in_date.month
            features['arrival_month_sin'] = math.sin(2 * math.pi * (month - 1) / 12)
            
            # Advance booking (days until check-in)
            today = date.today()
            advance_days = (check_in_date - today).days
            features['advance_booking'] = max(0, advance_days)
        
        # 3. Stay duration
        length_of_stay = user_input.get('length_of_stay', 1)
        features['stays_in_week_nights'] = length_of_stay
        features['room_nights'] = length_of_stay
        
        # 4. Lead time calculation
        features['lead_time_x_total_nights'] = features.get('advance_booking', 7) * length_of_stay
        
        # 5. Set default values for other features
        # NOTE: market_demand_index, competitor_avg_price calculate by DynamicPricingEngine
        default_features = {
            'is_canceled': 0,
            'market_demand_index': 0.7,  # Temporary - replace by engine
            'competitor_avg_price': 120.0,  # Temporary - replace by engine
            'adr_lag_7d': 110.0,
            'adr_rolling_mean_7d': 115.0,
            'hotel_premium_index': 1.0,
            'price_percentile': 0.5,
            'country_CZE': 0, 'country_DEU': 1, 'country_IRQ': 0, 'country_DMA': 0,
            'market_segment_Offline TA/TO': 1, 'market_segment_Groups': 0
        }
        
        features.update(default_features)
        
        logger.info(f"Mapped user input to {len(features)} model features")
        return features

# Test the feature mapper
if __name__ == "__main__":
    mapper = FeatureMapper()
    
    test_input = {
        'room_type': 'Deluxe',
        'check_in_date': '2024-12-25',  # Christmas
        'length_of_stay': 3,
        'guest_count': 2
    }
    
    features = mapper.map_user_input_to_features(test_input)
    print("Mapped features:")
    for key, value in list(features.items())[:10]:  # Show first 10
        print(f"  {key}: {value}")