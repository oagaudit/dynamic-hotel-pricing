import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import Dict, List, Any

class CompetitorIntelligence:
    def __init__(self):
        self.competitors = {
            'Luxury_A': {
                'strategy': 'premium',
                'base_multiplier': 1.3,
                'rating': 4.5,
                'capacity': 80
            },
            'Business_B': {
                'strategy': 'competitive', 
                'base_multiplier': 0.9,
                'rating': 4.2,
                'capacity': 120
            },
            'Boutique_C': {
                'strategy': 'value',
                'base_multiplier': 1.1,
                'rating': 4.4,
                'capacity': 60
            },
            'Resort_D': {
                'strategy': 'experience',
                'base_multiplier': 1.25,
                'rating': 4.3,
                'capacity': 100
            }
        }
    
    def generate_competitor_prices(self, base_price: float, stay_date: datetime, 
                                 room_type: str, length_of_stay: int) -> List[Dict]:
        """Generate realistic competitor pricing"""
        
        competitor_prices = []
        
        for comp_name, comp_info in self.competitors.items():
            # Start with base price adjustment
            comp_base = base_price * comp_info['base_multiplier']
            
            # Seasonal adjustment
            seasonal_factor = self._get_seasonal_factor(stay_date)
            
            # Weekend premium
            is_weekend = stay_date.weekday() >= 5
            weekend_factor = 1.15 if is_weekend else 1.0
            
            # Length of stay discount
            stay_discount = max(0.9, 1.0 - (length_of_stay * 0.02))
            
            # Random variation (±5%)
            random_variation = random.uniform(0.95, 1.05)
            
            # Final price calculation
            final_price = comp_base * seasonal_factor * weekend_factor * stay_discount * random_variation
            
            competitor_prices.append({
                'competitor_name': comp_name,
                'price': round(final_price, 2),
                'strategy': comp_info['strategy'],
                'rating': comp_info['rating'],
                'distance_km': round(random.uniform(0.5, 3.0), 1),
                'occupancy_rate': random.uniform(0.6, 0.95)
            })
        
        return competitor_prices
    
    def analyze_market_position(self, our_price: float, competitor_prices: List[Dict]) -> Dict:
        """Analyze our position in the market"""
        prices = [comp['price'] for comp in competitor_prices]
        
        market_avg = np.mean(prices)
        market_min = min(prices)
        market_max = max(prices)
        
        # Calculate position
        if our_price < market_avg * 0.9:
            position = "Budget Leader"
            competitiveness = "High"
        elif our_price < market_avg * 0.95:
            position = "Value Play"
            competitiveness = "Good"
        elif our_price < market_avg * 1.05:
            position = "Market Average" 
            competitiveness = "Neutral"
        elif our_price < market_avg * 1.1:
            position = "Premium Option"
            competitiveness = "Challenging"
        else:
            position = "Luxury Position"
            competitiveness = "Difficult"
        
        return {
            'market_avg_price': round(market_avg, 2),
            'market_min_price': round(market_min, 2),
            'market_max_price': round(market_max, 2),
            'our_price_rank': sum(1 for p in prices if p < our_price) + 1,
            'total_competitors': len(prices),
            'price_position': position,
            'competitiveness': competitiveness,
            'price_gap_percent': round(((our_price - market_avg) / market_avg) * 100, 1)
        }
    
    def _get_seasonal_factor(self, date: datetime) -> float:
        """Get seasonal pricing multiplier"""
        month = date.month
        
        # High season (summer, holidays)
        if month in [6, 7, 8, 12]:
            return random.uniform(1.2, 1.4)
        # Shoulder season  
        elif month in [4, 5, 9, 10, 11]:
            return random.uniform(1.0, 1.1)
        # Low season
        else:
            return random.uniform(0.8, 0.95)
    
    def get_demand_forecast(self, stay_date: datetime, length_of_stay: int) -> float:
        """Simple demand forecasting"""
        days_until_stay = (stay_date - datetime.now()).days
        
        # Base demand based on season
        seasonal_demand = self._get_seasonal_demand(stay_date.month)
        
        # Lead time effect
        if days_until_stay <= 3:
            lead_time_effect = 0.7  # Last-minute bookings
        elif days_until_stay <= 7:
            lead_time_effect = 0.8
        elif days_until_stay <= 14:
            lead_time_effect = 0.9  
        else:
            lead_time_effect = 1.0
        
        # Weekend effect
        is_weekend = stay_date.weekday() >= 5
        weekend_effect = 1.2 if is_weekend else 1.0
        
        final_demand = seasonal_demand * lead_time_effect * weekend_effect
        
        return min(1.0, max(0.3, final_demand))
    
    def _get_seasonal_demand(self, month: int) -> float:
        """Get seasonal demand factor"""
        if month in [6, 7, 8, 12]:  # High season
            return random.uniform(0.8, 1.0)
        elif month in [4, 5, 9, 10, 11]:  # Shoulder season
            return random.uniform(0.6, 0.8)
        else:  # Low season
            return random.uniform(0.3, 0.6)