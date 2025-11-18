import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from typing import Dict, Any, List
from src.competitor_intelligence import CompetitorIntelligence
from src.price_optimizer import PriceOptimizer, PricingStrategy

class DynamicPricingEngine:
    def __init__(self, model_path: str = None, features_path: str = None):
        self.base_model = None
        self.feature_names = None
        self.competitor_intel = CompetitorIntelligence()
        self.price_optimizer = PriceOptimizer()
        
        if model_path and features_path:
            self.load_base_model(model_path, features_path)
    
    def load_base_model(self, model_path: str, features_path: str):
        """Load the trained XGBoost model and feature list"""
        try:
            self.base_model = joblib.load(model_path)
            self.feature_names = pd.read_csv(features_path)['feature'].tolist()
            print(f"Loaded model with {len(self.feature_names)} features")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.base_model = None
    
    def predict_optimal_price(self, input_data: Dict) -> Dict[str, Any]:
        """Main method to predict optimal price"""
        
        # 1. Get base price from ML model
        base_price = self._get_base_price(input_data)
        
        # 2. Get competitor intelligence
        competitor_data = self.competitor_intel.generate_competitor_prices(
            base_price=base_price,
            stay_date=input_data['stay_date'],
            room_type=input_data['room_type'],
            length_of_stay=input_data['length_of_stay']
        )
        
        # 3. Analyze market position with BASE price (for initial analysis only)
        initial_market_analysis = self.competitor_intel.analyze_market_position(base_price, competitor_data)
        
        # 4. Get demand forecast
        demand_forecast = self.competitor_intel.get_demand_forecast(
            input_data['stay_date'], 
            input_data['length_of_stay']
        )
        
        # 5. Recommend and apply pricing strategy
        recommended_strategy = self.price_optimizer.recommend_strategy(
            initial_market_analysis, 
            input_data.get('current_occupancy', 0.7),
            demand_forecast
        )
        
        # 6. Calculate optimal price
        optimization_result = self.price_optimizer.optimize_price(
            base_price=base_price,
            market_analysis=initial_market_analysis,
            demand_forecast=demand_forecast,
            current_occupancy=input_data.get('current_occupancy', 0.7),
            strategy=recommended_strategy
        )
        
        # Use OPTIMAL price to analyze market 
        optimal_price = optimization_result['optimal_price']
        final_market_analysis = self.competitor_intel.analyze_market_position(optimal_price, competitor_data)
        
        # optimization_result with new market analysis
        optimization_result['market_analysis'] = final_market_analysis
        
        # 7. Generate business insights 
        business_insights = self._generate_business_insights(
            optimization_result, final_market_analysis, competitor_data, input_data, base_price
        )
        
        return {
            'pricing': optimization_result,
            'market_analysis': final_market_analysis,   
            'competitor_analysis': competitor_data,
            'demand_forecast': demand_forecast,
            'business_insights': business_insights,
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_base_price(self, input_data: Dict) -> float:
        """Get base price from ML model or fallback"""
        if self.base_model is not None:
            try:
                # Create feature vector and predict
                feature_vector = self._create_feature_vector(input_data)
                base_price = self.base_model.predict([feature_vector])[0]
                return float(max(0, base_price))  
            except Exception as e:
                print(f"ML model prediction failed: {e}")
        
        # Fallback: heuristic base price
        return self._calculate_heuristic_price(input_data)
    
    def _create_feature_vector(self, input_data: Dict) -> List[float]:
        """Create feature vector for ML model prediction"""
        feature_vector = []
        
        # Map input data to features (simplified)
        for feature in self.feature_names:
            if feature in input_data:
                feature_vector.append(input_data[feature])
            else:
                feature_vector.append(0.0)  # Default value
        
        return feature_vector
    
    def _calculate_heuristic_price(self, input_data: Dict) -> float:
        """Calculate fallback price using business rules"""
        base_rates = {
            'Standard': 100,
            'Deluxe': 150, 
            'Suite': 250,
            'Executive': 200
        }
        
        base_price = base_rates.get(input_data['room_type'], 120)
        
        # Adjustments
        if input_data['length_of_stay'] > 7:
            base_price *= 0.9  # Long stay discount
        
        if input_data['stay_date'].weekday() >= 5:  # Weekend
            base_price *= 1.15
        
        # Seasonal adjustment (simplified)
        month = input_data['stay_date'].month
        if month in [6, 7, 8, 12]:  # High season
            base_price *= 1.2
        
        return base_price
    
    def _generate_business_insights(self, pricing_result: Dict, market_analysis: Dict,
                                  competitor_data: List[Dict], input_data: Dict, base_price: float) -> Dict:
        """Generate actionable business insights with CORRECT price comparisons"""
        
        insights = {
            'key_drivers': [],
            'opportunities': [],
            'risks': [],
            'recommendations': [],
            'price_analysis': {}
        }
        
        optimal_price = float(pricing_result['optimal_price'])
        base_price = float(base_price)
        competitor_avg_price = float(market_analysis.get('market_avg_price', 0))   
        

        price_comparison = self._validate_price_comparison(optimal_price, competitor_avg_price)
        base_vs_optimal_comparison = self._validate_price_comparison(optimal_price, base_price)
        
        is_competitive = optimal_price <= competitor_avg_price
        
        insights['price_analysis'] = {
            'optimal_price': optimal_price,
            'base_price': base_price,
            'competitor_avg_price': competitor_avg_price,
            'price_change_percent': float(pricing_result['price_change_percent']),
            'price_gap_percent': float(market_analysis.get('price_gap_percent', 0)),
            

            'competitor_comparison': {
                'text': f"Our optimal price ${optimal_price:.2f} is {price_comparison} market average ${competitor_avg_price:.2f}",
                'our_price': optimal_price,
                'competitor_price': competitor_avg_price,
                'difference': optimal_price - competitor_avg_price,
                'is_competitive': is_competitive  
            },
            
            'base_price_comparison': {
                'text': f"Optimal price ${optimal_price:.2f} is {base_vs_optimal_comparison} base price ${base_price:.2f}",
                'difference': optimal_price - base_price,
                'difference_percent': float(pricing_result['price_change_percent'])
            }
        }
        
        # Key drivers
        if pricing_result['price_change_percent'] > 5:
            insights['key_drivers'].append("High demand period detected")
        
        competitiveness = market_analysis.get('competitiveness', 'Medium')
        if competitiveness == "High":
            insights['key_drivers'].append("Strong competitive position")
        
        if input_data['length_of_stay'] > 5:
            insights['key_drivers'].append("Long stay booking")
        
        # Opportunities
        if market_analysis.get('our_price_rank', 99) <= 2:
            insights['opportunities'].append("Consider premium pricing - top market position")
        
        expected_occupancy = pricing_result.get('expected_occupancy', 0)
        if expected_occupancy > 0.9:
            insights['opportunities'].append("High occupancy expected - optimize for revenue")
        
        # Risks
        competitor_comparison = insights['price_analysis']['competitor_comparison']
        if not competitor_comparison['is_competitive'] and competitor_comparison['difference'] > 10:
            insights['risks'].append(f"Price significantly above competitors - {competitor_comparison['text']}")
        
        if expected_occupancy < 0.5:
            insights['risks'].append("Low occupancy expected - consider promotional pricing")
        
        # Recommendations
        strategy = pricing_result['strategy']
        if strategy == "market_penetration":
            insights['recommendations'].append("Focus on increasing market share with competitive pricing")
        elif strategy == "revenue_maximization":
            insights['recommendations'].append("Optimize for maximum revenue with balanced pricing")
        elif strategy == "premium_positioning":
            insights['recommendations'].append("Leverage strong market position for premium pricing")
        
        if competitor_comparison['is_competitive']:
            insights['recommendations'].append("Maintain competitive pricing to attract price-sensitive customers")
        else:
            insights['recommendations'].append("Consider adjusting price to improve competitiveness")
        
        return insights
    
    def _validate_price_comparison(self, our_price: float, comparison_price: float) -> str:
        """Validate and generate correct price comparison text"""
        if our_price < comparison_price:
            return "lower than"
        elif our_price > comparison_price:
            return "higher than"
        else:
            return "equal to"
    
    def batch_predict(self, input_list: List[Dict]) -> List[Dict]:
        """Predict optimal prices for multiple inputs"""
        results = []
        for input_data in input_list:
            try:
                result = self.predict_optimal_price(input_data)
                results.append(result)
            except Exception as e:
                print(f"Error processing {input_data}: {e}")
                results.append({'error': str(e)})
        
        return results

# Example usage
if __name__ == "__main__":
    # Initialize the pricing engine
    engine = DynamicPricingEngine(
        model_path="models/xgboost_baseline_model_business.pkl",
        features_path="data/processed/selected_features_list_business.csv"
    )
    
    # Test prediction
    test_input = {
        'stay_date': datetime(2025, 10, 20),
        'room_type': 'Deluxe',
        'length_of_stay': 3,
        'guest_count': 2,
        'current_occupancy': 0.75
    }
    
    result = engine.predict_optimal_price(test_input)
    print("Pricing Result:", result['pricing'])
    print("Market Analysis:", result['market_analysis'])
    print("Business Insights:", result['business_insights'])
    print("\n Price Analysis:", result['business_insights']['price_analysis']['competitor_comparison']['text'])