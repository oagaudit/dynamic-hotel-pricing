# src/shap_explainer.py
import shap
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import joblib
from typing import Dict, List, Any, Union
import os
import sys
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SHAPExplainer:
    def __init__(self, model_path: str, features_path: str):
        """
        Simple SHAP Explainer for LLM integration and Streamlit dashboard
        """
        try:
            logger.info(f"Loading model from: {model_path}")
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded successfully. Model type: {type(self.model)}")
            
            # Get feature names from model (source of truth)
            self.feature_names = self._get_model_feature_names()
            logger.info(f"Model expects {len(self.feature_names)} features")
            
            logger.info("Initializing SHAP Explainer...")
            self.explainer = self._initialize_shap_explainer()
            logger.info("SHAP Explainer initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing SHAPExplainer: {e}")
            raise
    
    def _get_model_feature_names(self) -> List[str]:
        """Extract feature names from the trained model"""
        try:
            if hasattr(self.model, 'get_booster'):
                booster = self.model.get_booster()
                if hasattr(booster, 'feature_names') and booster.feature_names:
                    return booster.feature_names
            elif hasattr(self.model, 'feature_names_in_'):
                return self.model.feature_names_in_.tolist()
        except Exception as e:
            logger.warning(f"Could not extract feature names from model: {e}")
        
        # Fallback - read from CSV
        try:
            feature_data = pd.read_csv("data/processed/selected_features_list_final.csv")
            if 'feature' in feature_data.columns:
                return feature_data['feature'].tolist()
        except Exception as e:
            logger.error(f"Could not read features from CSV: {e}")
        
        raise ValueError("Could not determine feature names")
    
    def _initialize_shap_explainer(self):
        """Initialize SHAP explainer with simple approach"""
        def model_predict(X):
            return self.model.predict(X)
        
        # Generate minimal sample data
        sample_data = self._get_sample_data()
        return shap.Explainer(model_predict, sample_data)
    
    def _get_sample_data(self) -> pd.DataFrame:
        """Generate minimal sample data for SHAP"""
        sample_data = {}
        for feature in self.feature_names:
            # Simple default values based on feature type
            if any(keyword in feature.lower() for keyword in ['price', 'adr']):
                sample_data[feature] = np.random.uniform(50, 300, 50)
            elif any(keyword in feature.lower() for keyword in ['country', 'segment', 'type']):
                sample_data[feature] = np.random.randint(0, 2, 50)
            else:
                sample_data[feature] = np.random.uniform(0, 1, 50)
        
        return pd.DataFrame(sample_data)
    
    def _convert_value(self, value: Union[str, float, int]) -> float:
        """Convert input values to float, handling string formats"""
        if isinstance(value, (int, float)):
            return float(value)
        
        if isinstance(value, str):
            # Handle "[9.8783585E1]" format
            cleaned = value.strip().strip('[]').strip()
            try:
                return float(cleaned)
            except ValueError:
                logger.warning(f"Could not convert to float: {value}")
        
        return 0.0
    
    def prepare_features(self, input_data: Dict) -> np.ndarray:
        """Prepare feature vector with correct order"""
        feature_vector = []
        for feature in self.feature_names:
            value = input_data.get(feature, 0.0)
            feature_vector.append(self._convert_value(value))
        
        return np.array(feature_vector).reshape(1, -1)
    
    def analyze_prediction(self, input_data: Dict) -> Dict[str, Any]:
        """
        Perform SHAP analysis and return clean results for LLM and Streamlit
        
        Returns:
            Dict with:
            - prediction: float
            - shap_values: list of SHAP values
            - feature_impacts: sorted list of feature impacts
            - top_features: top 5 influential features
        """
        logger.info("Performing SHAP analysis for prediction")
        
        try:
            # Prepare features
            feature_df = pd.DataFrame([input_data], columns=self.feature_names)
            for feature in self.feature_names:
                if feature in input_data:
                    feature_df[feature] = self._convert_value(input_data[feature])
                else:
                    feature_df[feature] = 0.0
            
            # Calculate SHAP values
            shap_values = self.explainer(feature_df)
            prediction = self.model.predict(feature_df)[0]
            
            # Create feature impacts
            feature_impacts = []
            for i, feature_name in enumerate(self.feature_names):
                feature_impacts.append({
                    'feature': feature_name,
                    'shap_value': float(shap_values.values[0, i]),
                    'feature_value': float(feature_df.iloc[0, i]),
                    'impact_rank': 0
                })
            
            # Sort by absolute impact
            feature_impacts.sort(key=lambda x: abs(x['shap_value']), reverse=True)
            for i, impact in enumerate(feature_impacts):
                impact['impact_rank'] = i + 1
            
            # Get top features for LLM
            top_features = feature_impacts[:5]
            
            logger.info(f"Prediction: {prediction:.2f}")
            logger.info(f"Top features: {[f['feature'] for f in top_features]}")
            
            return {
                'prediction': float(prediction),
                'base_value': float(shap_values.base_values[0]),
                'shap_values': shap_values.values[0].tolist(),
                'feature_impacts': feature_impacts,
                'top_features': top_features,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error in SHAP analysis: {e}")
            return {
                'success': False,
                'error': str(e),
                'prediction': self.model.predict(self.prepare_features(input_data))[0] if 'input_data' in locals() else 0.0
            }
    
    def create_simple_plot(self, shap_analysis: Dict) -> go.Figure:
        """
        Create simple interactive plot for Streamlit dashboard
        """
        if not shap_analysis['success']:
            # Return empty plot if analysis failed
            fig = go.Figure()
            fig.update_layout(title='SHAP Analysis Failed', 
                            annotations=[dict(text="Analysis not available", x=0.5, y=0.5, showarrow=False)])
            return fig
        
        top_features = shap_analysis['top_features']
        features = [f['feature'] for f in top_features]
        shap_vals = [f['shap_value'] for f in top_features]
        
        fig = go.Figure(data=[
            go.Bar(
                x=shap_vals,
                y=features,
                orientation='h',
                marker_color=['green' if val > 0 else 'red' for val in shap_vals],
                text=[f'{val:.3f}' for val in shap_vals],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title='Top 5 Feature Impacts (SHAP Values)',
            xaxis_title='Impact on Price',
            yaxis_title='Features',
            showlegend=False,
            height=400
        )
        
        return fig

# Example usage
if __name__ == "__main__":
    try:
        logger.info("=== Simple SHAP Explainer Test ===")
        
        explainer = SHAPExplainer(
            model_path="models/xgboost_baseline_model_final.pkl",
            features_path="data/processed/selected_features_list_final.csv"
        )
        
        # Test data
        sample_input = {
            'is_canceled': 0,
            'stays_in_week_nights': 3,
            'arrival_month_sin': 0.5,
            'advance_booking': 14,
            'room_nights': 3,
            'market_demand_index': 0.8,
            'hotel_type_encoded': 1,
            'hotel_premium_index': 1.2,
            'competitor_avg_price': 150.0,
            'price_percentile': 0.6,
            'lead_time_x_total_nights': 42,
            'adr_lag_7d': 145.0,
            'adr_rolling_mean_7d': 140.0,
            'country_CZE': 0, 'country_DEU': 1, 'country_DMA': 0, 'country_IRQ': 0,
            'market_segment_Groups': 0, 'market_segment_Offline TA/TO': 1,
            'reserved_room_type_A': 1
        }
        
        # Get SHAP analysis
        result = explainer.analyze_prediction(sample_input)
        
        if result['success']:
            logger.info(f"Prediction: ${result['prediction']:.2f}")
            logger.info("Top features:")
            for feature in result['top_features']:
                logger.info(f"   {feature['feature']}: {feature['shap_value']:.4f}")
            
            # Create plot
            fig = explainer.create_simple_plot(result)
            logger.info("SHAP analysis completed successfully!")
        else:
            logger.error(f"SHAP analysis failed: {result.get('error')}")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")