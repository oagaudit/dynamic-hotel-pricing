import numpy as np
from typing import Dict, List, Any
from enum import Enum

class PricingStrategy(Enum):
    MARKET_PENETRATION = "market_penetration"      # Price lower than market to gain market share
    REVENUE_MAXIMIZATION = "revenue_maximization"  # Maximize revenue as much as possible
    COMPETITIVE_MATCH = "competitive_match"        # Follow main competitors
    PREMIUM_POSITIONING = "premium_positioning"    # Price higher than market
    DEMAND_BASED = "demand_based"                  # Based on demand

class PriceOptimizer:
    def __init__(self):
        self.strategies = {
            PricingStrategy.MARKET_PENETRATION: {
                'target_position': 'budget_leader',
                'max_discount': 0.15,
                'min_occupancy': 0.6
            },
            PricingStrategy.REVENUE_MAXIMIZATION: {
                'target_position': 'value_play', 
                'price_range': (0.95, 1.1),
                'demand_sensitivity': 0.2
            },
            PricingStrategy.COMPETITIVE_MATCH: {
                'target_position': 'market_average',
                'competitor_weight': 0.8,
                'max_deviation': 0.05
            },
            PricingStrategy.PREMIUM_POSITIONING: {
                'target_position': 'premium',
                'min_premium': 0.1,
                'max_premium': 0.3
            },
            PricingStrategy.DEMAND_BASED: {
                'demand_weight': 0.7,
                'competitor_weight': 0.3
            }
        }
    
    def optimize_price(self, base_price: float, market_analysis: Dict, 
                      demand_forecast: float, current_occupancy: float = 0.7,
                      strategy: PricingStrategy = PricingStrategy.REVENUE_MAXIMIZATION) -> Dict:
        """Calculate optimal price using selected strategy"""
        
        strategy_config = self.strategies[strategy]
        market_avg = market_analysis['market_avg_price']
        
        if strategy == PricingStrategy.MARKET_PENETRATION:
            optimal_price = self._market_penetration_strategy(base_price, market_avg, current_occupancy)
            
        elif strategy == PricingStrategy.REVENUE_MAXIMIZATION:
            optimal_price = self._revenue_maximization_strategy(base_price, market_avg, demand_forecast)
            
        elif strategy == PricingStrategy.COMPETITIVE_MATCH:
            optimal_price = self._competitive_match_strategy(base_price, market_avg, market_analysis)
            
        elif strategy == PricingStrategy.PREMIUM_POSITIONING:
            optimal_price = self._premium_positioning_strategy(base_price, market_avg, demand_forecast)
            
        elif strategy == PricingStrategy.DEMAND_BASED:
            optimal_price = self._demand_based_strategy(base_price, market_avg, demand_forecast)
        
        else:
            optimal_price = base_price
        
        # Apply final constraints
        optimal_price = self._apply_constraints(optimal_price, base_price, market_analysis)
        
        price_comparison = self._validate_price_comparison(optimal_price, market_avg)
        
        return {
            'optimal_price': round(optimal_price, 2),
            'base_price': round(base_price, 2),
            'strategy': strategy.value,
            'price_change_percent': round(((optimal_price - base_price) / base_price) * 100, 1),
            'market_position': self._calculate_market_position(optimal_price, market_avg),
            'expected_occupancy': self._estimate_occupancy(optimal_price, market_avg, demand_forecast),
            'revenue_impact': self._estimate_revenue_impact(base_price, optimal_price, demand_forecast),
            'price_comparison': {
                'optimal_vs_market': price_comparison,
                'optimal_price': optimal_price,
                'market_avg_price': market_avg,
                'is_competitive': optimal_price <= market_avg
            }
        }
    
    def _validate_price_comparison(self, optimal_price: float, market_avg: float) -> str:
        """ FIXED: Correct indentation"""
        if optimal_price < market_avg:
            return f"lower than market average (${optimal_price:.2f} vs ${market_avg:.2f})"
        elif optimal_price > market_avg:
            return f"higher than market average (${optimal_price:.2f} vs ${market_avg:.2f})"
        else:
            return f"equal to market average (${optimal_price:.2f})"
    
    def _market_penetration_strategy(self, base_price: float, market_avg: float, occupancy: float) -> float:
        """Aggressive pricing to gain market share"""
        if occupancy < 0.7:
            # More aggressive discount if occupancy is low
            target_price = market_avg * 0.85
        else:
            target_price = market_avg * 0.92
            
        return min(base_price, target_price)
    
    def _revenue_maximization_strategy(self, base_price: float, market_avg: float, demand: float) -> float:
        """Balance price and demand for maximum revenue"""
        # Price elasticity model (simplified)
        elasticity = -1.5  # 1% price increase → 1.5% demand decrease
        
        # Test different price points
        price_points = np.linspace(base_price * 0.8, base_price * 1.2, 50)
        expected_revenues = []
        
        for price in price_points:
            price_ratio = price / base_price
            demand_change = elasticity * (price_ratio - 1)
            adjusted_demand = max(0.1, demand * (1 + demand_change))
            expected_revenue = price * adjusted_demand
            expected_revenues.append(expected_revenue)
        
        optimal_idx = np.argmax(expected_revenues)
        return price_points[optimal_idx]
    
    def _competitive_match_strategy(self, base_price: float, market_avg: float, market_analysis: Dict) -> float:
        """Price close to market average"""
        our_rank = market_analysis['our_price_rank']
        total_comps = market_analysis['total_competitors']
        
        # Aim for top 30% position
        target_rank = max(1, total_comps * 0.3)
        
        if our_rank > target_rank:
            # We're too expensive, reduce price
            adjustment = -0.08
        elif our_rank < target_rank * 0.7:
            # We're too cheap, increase price
            adjustment = 0.05
        else:
            adjustment = 0.02  # Small premium for good position
            
        return market_avg * (1 + adjustment)
    
    def _premium_positioning_strategy(self, base_price: float, market_avg: float, demand: float) -> float:
        """Maintain premium pricing position"""
        min_premium = 0.1  # 10% minimum premium
        max_premium = 0.3  # 30% maximum premium
        
        # Adjust premium based on demand
        demand_bonus = (demand - 0.5) * 0.2  # ±10% based on demand
        
        target_premium = min_premium + demand_bonus
        target_premium = min(max_premium, max(min_premium, target_premium))
        
        return market_avg * (1 + target_premium)
    
    def _demand_based_strategy(self, base_price: float, market_avg: float, demand: float) -> float:
        """Price primarily based on demand"""
        # Strong demand → higher prices, weak demand → lower prices
        demand_factor = 0.5 + demand  # 0.8-1.5 range
        
        # Still consider market position
        market_weight = 0.3
        demand_weight = 0.7
        
        return (market_avg * market_weight + base_price * demand_factor * demand_weight) / (market_weight + demand_weight)
    
    def _apply_constraints(self, price: float, base_price: float, market_analysis: Dict) -> float:
        """Apply business constraints to price"""
        min_price = base_price * 0.7   # No more than 30% discount
        max_price = base_price * 1.5   # No more than 50% premium
        
        # Don't go below cheapest competitor by more than 10%
        min_market = market_analysis['market_min_price'] * 0.9
        
        constrained_price = max(min_price, min(max_price, price))
        constrained_price = max(min_market, constrained_price)
        
        return constrained_price
    
    def _calculate_market_position(self, price: float, market_avg: float) -> str:
        """Calculate market position based on price"""
        ratio = price / market_avg
        
        if ratio < 0.9:
            return "Budget Leader"
        elif ratio < 0.95:
            return "Value Play"
        elif ratio < 1.05:
            return "Market Average"
        elif ratio < 1.1:
            return "Premium Option"
        else:
            return "Luxury Position"
    
    def _estimate_occupancy(self, price: float, market_avg: float, base_demand: float) -> float:
        """Estimate occupancy based on price position"""
        price_ratio = price / market_avg
        
        if price_ratio < 0.9:
            demand_boost = 0.2
        elif price_ratio < 0.95:
            demand_boost = 0.1
        elif price_ratio < 1.05:
            demand_boost = 0.0
        elif price_ratio < 1.1:
            demand_boost = -0.1
        else:
            demand_boost = -0.2
            
        return min(0.95, max(0.1, base_demand + demand_boost))
    
    def _estimate_revenue_impact(self, old_price: float, new_price: float, demand: float) -> str:
        """Estimate revenue impact of price change"""
        old_revenue = old_price * demand
        new_demand = self._estimate_occupancy(new_price, old_price, demand)  # Simplified
        new_revenue = new_price * new_demand
        
        change_percent = ((new_revenue - old_revenue) / old_revenue) * 100
        
        if change_percent > 5:
            return f"+{change_percent:.1f}% (Significant Increase)"
        elif change_percent > 2:
            return f"+{change_percent:.1f}% (Moderate Increase)"
        elif change_percent > -2:
            return f"{change_percent:.1f}% (Neutral)"
        elif change_percent > -5:
            return f"{change_percent:.1f}% (Moderate Decrease)"
        else:
            return f"{change_percent:.1f}% (Significant Decrease)"
    
    def recommend_strategy(self, market_analysis: Dict, current_occupancy: float, 
                         demand_forecast: float) -> PricingStrategy:
        """Recommend best pricing strategy based on conditions"""
        
        market_position = market_analysis['price_position']
        competitiveness = market_analysis['competitiveness']
        
        if current_occupancy < 0.6:
            return PricingStrategy.MARKET_PENETRATION
            
        elif competitiveness == "Difficult" and current_occupancy < 0.8:
            return PricingStrategy.COMPETITIVE_MATCH
            
        elif demand_forecast > 0.8 and current_occupancy > 0.85:
            return PricingStrategy.PREMIUM_POSITIONING
            
        elif demand_forecast < 0.4:
            return PricingStrategy.DEMAND_BASED
            
        else:
            return PricingStrategy.REVENUE_MAXIMIZATION
