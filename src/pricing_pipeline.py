# src/pricing_pipeline.py
import sys
import os
import logging
from typing import Dict, Any
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_mapper import FeatureMapper
from src.shap_explainer import SHAPExplainer
from src.llm_recommender import LLMRecommender
from src.dynamic_pricing_engine import DynamicPricingEngine

logger = logging.getLogger(__name__)

class HotelPricingPipeline:
    """
    Main pipeline that integrates ALL components for Streamlit usage
    FIXED: Now SHAP and DynamicPricing use the same features
    """
    
    def __init__(self):
        logger.info("Initializing Hotel Pricing Pipeline...")
        
        try:
            # Initialize all components
            self.feature_mapper = FeatureMapper()
            
            # Dynamic Pricing Engine (includes competitor intelligence + price optimization)
            self.dynamic_pricing_engine = DynamicPricingEngine(
                model_path="models/xgboost_baseline_model_final.pkl",
                features_path="data/processed/selected_features_list_final.csv"
            )
            
            # SHAP for explanations
            self.shap_explainer = SHAPExplainer(
                model_path="models/xgboost_baseline_model_final.pkl", 
                features_path="data/processed/selected_features_list_final.csv"
            )
            
            # LLM for business insights
            self.llm_recommender = LLMRecommender()
            
            logger.info("Hotel Pricing Pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise
    
    def get_price_recommendation(self, user_input: Dict) -> Dict[str, Any]:
        """
        Main method: Transform user input -> complete pricing analysis
        FIXED: SHAP and DynamicPricing now use consistent features
        """
        try:
            logger.info("Processing pricing recommendation...")
            
            # 1. Map user input to model features (ใช้ร่วมกันทั้งระบบ)
            features = self.feature_mapper.map_user_input_to_features(user_input)
            logger.info(f"Mapped to {len(features)} features")
            
            # 2. Get BASE PRICE from ML model (ใช้ features เดียวกับ SHAP)
            shap_result = self.shap_explainer.analyze_prediction(features)
            base_price = shap_result['prediction']
            logger.info(f"ML Model Base Price: ${base_price:.2f}")
            
            # 3. Get OPTIMAL PRICE from Dynamic Pricing Engine 
            # ใช้ features เดียวกัน + business logic
            pricing_result = self.dynamic_pricing_engine.predict_optimal_price({
                'stay_date': datetime.strptime(user_input['check_in_date'], '%Y-%m-%d') if isinstance(user_input['check_in_date'], str) else user_input['check_in_date'],
                'room_type': user_input['room_type'],
                'length_of_stay': user_input['length_of_stay'],
                'guest_count': user_input.get('guest_count', 2),
                'current_occupancy': 0.7
            })
            
            optimal_price = pricing_result['pricing']['optimal_price']
            logger.info(f"Dynamic Pricing Optimal Price: ${optimal_price:.2f}")
            
            # 4. Calculate price difference and reason
            price_difference = optimal_price - base_price
            price_difference_percent = (price_difference / base_price) * 100
            
            difference_explanation = self._explain_price_difference(
                price_difference, 
                pricing_result['pricing']['strategy'],
                pricing_result['market_analysis']
            )
            
            # 5. Get AI business insights
            pricing_context = {
                'room_type': user_input['room_type'],
                'length_of_stay': user_input['length_of_stay'],
                'guest_count': user_input.get('guest_count', 2),
                'competitor_price': pricing_result['market_analysis']['market_avg_price'],
                'base_price': base_price,
                'optimal_price': optimal_price,
                'price_difference': price_difference
            }
            
            llm_result = self.llm_recommender.generate_recommendations(shap_result, pricing_context)
            
            # ✅ วางก่อน return - จะได้ทำงานจริง
            if llm_result['success']:
                logger.info("🤖 LLM Recommendations Details:")
                recs = llm_result['recommendations']
                logger.info(f"Summary: {recs.get('summary', 'N/A')}")
                logger.info(f"Price Analysis: {recs.get('price_analysis', 'N/A')}")
                logger.info(f"Strategy: {recs.get('strategy', 'N/A')}")
                logger.info(f"Risks: {recs.get('risk_considerations', 'N/A')}")
            else:
                logger.warning(f"LLM failed: {llm_result.get('error')}")
            
            # 6. Create interactive visualization
            interactive_plot = self.shap_explainer.create_simple_plot(shap_result)
            
            # 7. Prepare Streamlit-ready response
            return {
                'success': True,
                
                # Pricing information - show both price
                'base_price': base_price,                    # price from ML model
                'optimal_price': optimal_price,              # price from  Dynamic Pricing
                'price_difference': price_difference,        # differ result
                'price_difference_percent': price_difference_percent,
                'difference_explanation': difference_explanation,
                
                # Strategy and context
                'strategy': pricing_result['pricing']['strategy'],
                'price_change_percent': pricing_result['pricing']['price_change_percent'],
                
                # Market intelligence
                'market_analysis': {
                'price_position': pricing_result['pricing']['market_position'],
                'competitiveness': pricing_result['market_analysis'].get('competitiveness', 'Medium'),
                'our_price_rank': pricing_result['market_analysis'].get('our_price_rank', 1),
                'total_competitors': pricing_result['market_analysis'].get('total_competitors', 4),
                'price_gap_percent': pricing_result['market_analysis'].get('price_gap_percent', 0),
                # ✅ **เพิ่มฟิลด์นี้ - ใช้ market_avg_price จาก dynamic_pricing**
                'competitor_avg_price': pricing_result['market_analysis'].get('market_avg_price', 0)
            },
            
            'competitor_analysis': pricing_result['competitor_analysis'],
            'demand_forecast': pricing_result['demand_forecast'],
                
                # AI Explanations
                'shap_analysis': shap_result,
                'feature_impacts': shap_result.get('top_features', []),
                'interactive_plot': interactive_plot,
                
                # Business Insights
                'ai_insights': llm_result,
                'business_insights': pricing_result['business_insights'],
                
                # Metadata
                'timestamp': datetime.now().isoformat(),
                'input_features': features
            }
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return {
                'success': False,
                'error': str(e),
                'base_price': 0.0,
                'optimal_price': 0.0,
                'timestamp': datetime.now().isoformat()
            } 
    
    def _explain_price_difference(self, difference: float, strategy: str, market_analysis: Dict) -> str:
        """
        Explain why optimal price differs from base price
        """
        abs_difference = abs(difference)
        
        if abs_difference < 5:
            return "Minimal adjustment for market conditions"
        
        elif difference > 0:
            # Optimal price is HIGHER than base
            if strategy == 'premium_positioning':
                return f"Premium positioning strategy adds ${abs_difference:.2f} for better market position"
            elif strategy == 'revenue_maximization':
                return f"Revenue optimization increases price by ${abs_difference:.2f} based on demand"
            else:
                return f"Market conditions support ${abs_difference:.2f} price increase"
        
        else:
            # Optimal price is LOWER than base  
            if strategy == 'market_penetration':
                return f"Market penetration strategy reduces price by ${abs_difference:.2f} to gain share"
            elif market_analysis.get('competitiveness') == 'Difficult':
                return f"Competitive market requires ${abs_difference:.2f} price reduction"
            else:
                return f"Strategic pricing adjustment: ${abs_difference:.2f} reduction for better positioning"
    
    def get_quick_estimate(self, user_input: Dict) -> float:
        """
        Fast price prediction using ML model only
        For real-time price preview in Streamlit
        """
        try:
            features = self.feature_mapper.map_user_input_to_features(user_input)
            shap_result = self.shap_explainer.analyze_prediction(features)
            return shap_result.get('prediction', 0.0)
        except Exception as e:
            logger.error(f"Quick estimate error: {e}")
            return 0.0

# Global instance for Streamlit (cached)
_pipeline_instance = None

def get_pipeline():
    """
    Get or create pipeline instance
    For Streamlit caching to avoid re-initialization
    """
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = HotelPricingPipeline()
    return _pipeline_instance

# Test the fixed pipeline
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    try:
        print("Testing FIXED Pricing Pipeline...")
        
        pipeline = HotelPricingPipeline()
        
        test_input = {
            'room_type': 'Deluxe',
            'check_in_date': '2024-12-25',
            'length_of_stay': 3,
            'guest_count': 2
        }
        
        print("User Input:", test_input)
        
        result = pipeline.get_price_recommendation(test_input)
        
        if result['success']:
            print("Fixed Pipeline Test Successful!")
            print(f"ML Base Price: ${result['base_price']:.2f}")
            print(f"Optimal Price: ${result['optimal_price']:.2f}")
            print(f"Price Difference: ${result['price_difference']:+.2f} ({result['price_difference_percent']:+.1f}%)")
            print(f"Explanation: {result['difference_explanation']}")
            print(f"Strategy: {result['strategy']}")
            print(f"Market Position: {result['market_analysis']['price_position']}")
        else:
            print(f"Pipeline failed: {result.get('error')}")
            
    except Exception as e:
        print(f"Test failed: {e}")