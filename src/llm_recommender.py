# src/llm_recommender.py
import requests
import json
import re
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class LLMRecommender:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2:1b"):
        """
        LLM Recommender for generating business insights
        
        Args:
            base_url: Ollama server URL
            model: Ollama model name
        """
        self.base_url = base_url
        self.model = model
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to Ollama server"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = [m['name'] for m in response.json().get('models', [])]
                logger.info(f"Connected to Ollama. Available models: {models}")
            else:
                logger.warning("Could not connect to Ollama")
        except Exception as e:
            logger.warning(f"Ollama connection failed: {e}")
    
    def generate_recommendations(self, shap_data: Dict, pricing_context: Dict = None) -> Dict[str, Any]:
        """
        Generate business recommendations using LLM
        """
        if pricing_context is None:
            pricing_context = {}
        
        try:
            #  Get correct price data from pricing_context
            optimal_price = pricing_context.get('optimal_price', 0)
            competitor_price = pricing_context.get('competitor_price', 0)
            base_price = pricing_context.get('base_price', 0)
            
            # Prepare prompt for LLM
            prompt = self._create_business_prompt(shap_data, pricing_context, optimal_price, competitor_price, base_price)
            
            # Generate response from LLM
            llm_response = self._call_ollama(prompt)
            
            # Parse and structure the response
            recommendations = self._parse_llm_response(llm_response)
            
            logger.info("LLM recommendations generated successfully")
            return {
                'success': True,
                'recommendations': recommendations,
                'raw_response': llm_response
            }
            
        except Exception as e:
            logger.error(f"LLM recommendation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'fallback_recommendations': self._generate_fallback_recommendations(shap_data, pricing_context)
            }
    
    def _create_business_prompt(self, shap_data: Dict, pricing_context: Dict, 
                              optimal_price: float, competitor_price: float, base_price: float) -> str:
        """
        Create a detailed business analysis prompt for LLM with EXPLICIT STRUCTURE
        """
        prediction = optimal_price
        top_features = shap_data.get('top_features', [])
        
        # Use correct price data
        prompt = f"""
You are a hotel pricing strategy expert. Analyze this price prediction data and provide business recommendations.

**PRICE ANALYSIS DATA:**
- ML Base Price: ${base_price:.2f}
- AI Optimal Price: ${optimal_price:.2f}
- Price Change: {((optimal_price - base_price) / base_price * 100):+.1f}%
- Competitor Average Price: ${competitor_price:.2f}
- Our Position: {'COMPETITIVE' if optimal_price <= competitor_price else 'PREMIUM'}

**TOP FACTORS INFLUENCING PRICE (by importance):"""
        
        for i, feature in enumerate(top_features, 1):
            feature_name = feature['feature']
            shap_value = feature['shap_value']
            impact_type = "INCREASES price" if shap_value > 0 else "DECREASES price"
            
            prompt += f"\n{i}. {feature_name}: {shap_value:+.3f} ({impact_type})"
        
        prompt += f"""

**ADDITIONAL CONTEXT:**
- Room Type: {pricing_context.get('room_type', 'Standard')}
- Length of Stay: {pricing_context.get('length_of_stay', 1)} nights
- Guest Count: {pricing_context.get('guest_count', 2)}
- Pricing Strategy: {pricing_context.get('strategy', 'Revenue Maximization')}

**IMPORTANT: Base your analysis on the AI Optimal Price (${optimal_price:.2f}), NOT the ML Base Price.**

**CRITICAL: You MUST structure your response EXACTLY as follows:**

**SUMMARY:**
[Provide a concise 2-3 sentence executive summary of the key findings and recommendations]

**PRICE ANALYSIS:**
[Analyze whether the optimal price ${optimal_price:.2f} is appropriate compared to competitor ${competitor_price:.2f}. Consider market position, value proposition, and competitive landscape.]

**STRATEGY:**
[Provide specific, actionable strategic recommendations. Focus on how to maximize revenue while maintaining competitiveness. Include pricing adjustments, target segments, and implementation steps.]

**RISKS:**
[Identify potential risks and considerations. Include market risks, competitive responses, operational impacts, and mitigation strategies.]

**FORMAT REQUIREMENTS:**
- Use the EXACT section headers: **SUMMARY:**, **PRICE ANALYSIS:**, **STRATEGY:**, **RISKS:**
- Use bullet points (*) for lists
- Keep each section focused and concise
- Provide actionable insights for hotel managers
"""
        return prompt
    
    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API to generate response"""
        # Try different API endpoints
        endpoints = [
            f"{self.base_url}/api/generate",
            f"{self.base_url}/api/chat",
            f"{self.base_url}/v1/chat/completions"
        ]
        
        for endpoint in endpoints:
            try:
                logger.info(f"Trying Ollama endpoint: {endpoint}")
                
                if "chat" in endpoint or "completions" in endpoint:
                    # Use chat format
                    payload = {
                        "model": self.model,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "stream": False
                    }
                else:
                    # Use generate format
                    payload = {
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False
                    }
                
                response = requests.post(endpoint, json=payload, timeout=120)
                response.raise_for_status()
                
                result = response.json()
                
                # Extract response based on endpoint type
                if "chat" in endpoint:
                    llm_response = result.get('message', {}).get('content', '')
                    if not llm_response:
                        llm_response = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                else:
                    llm_response = result.get('response', '')
                
                llm_response = llm_response.strip()
                
                if llm_response:
                    logger.info(f"Success with endpoint: {endpoint}")
                    logger.info("Received response from LLM")
                    return llm_response
                    
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    logger.warning(f"Endpoint not found: {endpoint}")
                    continue
                else:
                    logger.warning(f"HTTP error with {endpoint}: {e}")
                    continue
            except Exception as e:
                logger.warning(f"Error with {endpoint}: {e}")
                continue
        
        # If all endpoints fail
        raise Exception("All Ollama API endpoints failed. Please check if Ollama is running and the model is available.")
    
    def _parse_llm_response(self, llm_response: str) -> Dict[str, Any]:
        """
        Parse LLM response into structured format - FIXED FALLBACK
        """
        try:
            recommendations = {
                'summary': '',
                'price_analysis': '',
                'strategy': '',
                'risk_considerations': ''
            }
            
            # IMPROVED: Flexible section header detection
            sections = {
                'summary': ['**SUMMARY:**', '**Summary:**'],
                'price_analysis': ['**PRICE ANALYSIS:**', '**Price Analysis:**'], 
                'strategy': ['**STRATEGY:**', '**Strategy:**'],
                'risk_considerations': ['**RISKS:**', '**Risks:**', '**RISK CONSIDERATIONS:**']
            }
            
            current_section = None
            lines = llm_response.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check for section headers (case insensitive)
                line_lower = line.lower()
                for section, headers in sections.items():
                    for header in headers:
                        if line_lower.startswith(header.lower()):
                            current_section = section
                            # Skip the header line itself
                            break
                    if current_section:
                        break
                else:
                    # If no section header found, add to current section
                    if current_section and line:
                        # Skip markdown formatting and other metadata
                        if not any(keyword in line_lower for keyword in 
                                  ['**price analysis data', '**top factors', '**additional context', 
                                   '**important:', '**request:', '**critical:', '**format requirements']):
                            
                            # Clean the line
                            clean_line = line.replace('* ', '').replace('+ ', '').replace('- ', '').strip()
                            if clean_line and not clean_line.startswith('**'):
                                recommendations[current_section] += clean_line + ' '
            
            # FINAL CLEANUP - IMPROVED
            for section in recommendations:
                content = recommendations[section].strip()
                if not content:
                    # FIX: Actually call the fallback method
                    content = self._extract_section_fallback(llm_response, section)
                    recommendations[section] = content
                else:
                    # Clean up existing content
                    recommendations[section] = ' '.join(content.split())
            
            # FIX: Validate and ensure minimum content
            self._validate_recommendations(recommendations, llm_response)
            
            logger.info(f" Final parsing - Summary: {len(recommendations['summary'])} chars, "
                       f"Price Analysis: {len(recommendations['price_analysis'])} chars, "
                       f"Strategy: {len(recommendations['strategy'])} chars, "
                       f"Risks: {len(recommendations['risk_considerations'])} chars")
            
            return recommendations
            
        except Exception as e:
            logger.warning(f"Could not parse LLM response structure: {e}")
            return self._fallback_parse(llm_response)

    def _validate_recommendations(self, recommendations: Dict, llm_response: str):
        """Ensure all sections have meaningful content"""
        for section, content in recommendations.items():
            if not content.strip() or "not properly extracted" in content:
                # Try one more extraction attempt
                fallback_content = self._extract_section_fallback(llm_response, section)
                if fallback_content and "not properly extracted" not in fallback_content:
                    recommendations[section] = fallback_content
                elif section == 'summary':
                    # Use first paragraph as summary
                    paragraphs = [p.strip() for p in llm_response.split('\n\n') if p.strip()]
                    if paragraphs:
                        recommendations[section] = paragraphs[0]
    
    def _extract_section_fallback(self, llm_response: str, section: str) -> str:
        """Improved fallback section extraction"""
        try:
            section_patterns = {
                'summary': [r'\*\*Summary:\*\*(.*?)(?=\*\*|$)', r'executive summary(.*?)(?=\n\n)'],
                'price_analysis': [r'\*\*Price Analysis:\*\*(.*?)(?=\*\*|$)', r'price analysis(.*?)(?=\n\n)'],
                'strategy': [r'\*\*Strategy:\*\*(.*?)(?=\*\*|$)', r'strategic recommendations(.*?)(?=\n\n)'],
                'risk_considerations': [r'\*\*Risks:\*\*(.*?)(?=\*\*|$)', r'risk considerations(.*?)(?=\n\n)']
            }
            
            for pattern in section_patterns.get(section, []):
                matches = re.findall(pattern, llm_response, re.IGNORECASE | re.DOTALL)
                if matches:
                    content = matches[0].strip()
                    # Clean up markdown and bullet points
                    content = re.sub(r'[\*\-]\s*\*?', '', content)  # Remove bullets
                    content = re.sub(r'\*\*', '', content)  # Remove bold
                    content = ' '.join(content.split())  # Normalize spaces
                    if content and len(content) > 10:  # Meaningful content
                        return content
            
            # If regex fails, try simple keyword-based extraction
            return self._simple_keyword_extraction(llm_response, section)
            
        except Exception as e:
            logger.warning(f"Fallback extraction failed for {section}: {e}")
            return f"Content for {section} not properly extracted."
    
    def _simple_keyword_extraction(self, llm_response: str, section: str) -> str:
        """Simple keyword-based content extraction"""
        keywords = {
            'summary': ['summary', 'executive', 'overview', 'based on'],
            'price_analysis': ['price', 'pricing', 'appropriate', 'competitor', 'market'],
            'strategy': ['strategy', 'strategic', 'recommend', 'suggest', 'propose'],
            'risk_considerations': ['risk', 'potential', 'consideration', 'challenge', 'mitigation']
        }
        
        lines = llm_response.split('\n')
        relevant_lines = []
        
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in keywords[section]):
                clean_line = line.strip()
                if (clean_line and 
                    not clean_line.startswith('**') and 
                    len(clean_line) > 20):  # Meaningful line
                    relevant_lines.append(clean_line)
        
        if relevant_lines:
            return ' '.join(relevant_lines[:3])  # Return first 3 relevant lines
        
        return f"Content for {section} not properly extracted."

    def _fallback_parse(self, llm_response: str) -> Dict[str, Any]:
        """Ultimate fallback parsing"""
        # Split by double newlines and assign to sections
        paragraphs = [p.strip() for p in llm_response.split('\n\n') if p.strip()]
        
        result = {
            'summary': paragraphs[0] if paragraphs else "No summary available",
            'price_analysis': "Price analysis not available", 
            'strategy': "Strategy recommendations not available",
            'risk_considerations': "Risk analysis not available"
        }
        
        # Try to find relevant content for each section
        llm_lower = llm_response.lower()
        
        if 'price' in llm_lower and 'analysis' in llm_lower:
            for p in paragraphs:
                if 'price' in p.lower() and 'analysis' in p.lower():
                    result['price_analysis'] = p
                    break
                    
        if 'strategy' in llm_lower or 'recommend' in llm_lower:
            for p in paragraphs:
                if 'strategy' in p.lower() or 'recommend' in p.lower():
                    result['strategy'] = p
                    break
                    
        if 'risk' in llm_lower:
            for p in paragraphs:
                if 'risk' in p.lower():
                    result['risk_considerations'] = p
                    break
        
        return result
    
    def _generate_fallback_recommendations(self, shap_data: Dict, pricing_context: Dict) -> Dict[str, str]:
        """
        Generate fallback recommendations when LLM fails - WITH CONSISTENT KEYS
        """
        optimal_price = pricing_context.get('optimal_price', 0)
        competitor_price = pricing_context.get('competitor_price', 0)
        base_price = pricing_context.get('base_price', 0)
        top_features = shap_data.get('top_features', [])
        strategy = pricing_context.get('strategy', 'revenue_maximization')
        
        # Calculate metrics
        price_change_percent = ((optimal_price - base_price) / base_price * 100) if base_price > 0 else 0
        is_competitive = optimal_price <= competitor_price
        
        # FIX: Use consistent 'strategy' key
        return {
            'summary': f"AI recommends ${optimal_price:.2f} ({price_change_percent:+.1f}% from base ${base_price:.2f}). "
                      f"This is {'COMPETITIVE' if is_competitive else 'PREMIUM'} vs competitor ${competitor_price:.2f}.",
            
            'price_analysis': f"Optimal price ${optimal_price:.2f} is recommended based on {strategy.replace('_', ' ')} strategy.",
            
            'strategy': f"Focus on {strategy.replace('_', ' ')} to maximize revenue. "
                       f"Monitor booking patterns and adjust pricing dynamically.",
            
            'risk_considerations': f"Standard market risks apply. {'Competitive position looks strong.' if is_competitive else 'Monitor booking velocity at premium price.'}"
        }


# Example usage and testing
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        logger.info("=== LLM Recommender Test ===")
        
        # Initialize LLM Recommender
        recommender = LLMRecommender()
        
        # Test data (from SHAP analysis)
        test_shap_data = {
            'prediction': 126.31,
            'top_features': [
                {'feature': 'price_percentile', 'shap_value': -82.36},
                {'feature': 'stays_in_week_nights', 'shap_value': -0.44},
                {'feature': 'room_nights', 'shap_value': 0.21},
                {'feature': 'advance_booking', 'shap_value': 0.12},
                {'feature': 'market_segment_Groups', 'shap_value': -0.11}
            ]
        }
        
        # Use correct price data
        test_pricing_context = {
            'room_type': 'Deluxe',
            'length_of_stay': 3,
            'guest_count': 2,
            'base_price': 123.76,
            'optimal_price': 87.02,  # Actual calculated price
            'competitor_price': 131.17,  # Competitor price
            'strategy': 'Revenue Maximization'
        }
        
        # Generate recommendations
        result = recommender.generate_recommendations(test_shap_data, test_pricing_context)
        
        if result['success']:
            logger.info("LLM Recommendations:")
            recs = result['recommendations']
            logger.info(f"Summary: {recs.get('summary', 'N/A')}")
            logger.info(f"Price Analysis: {recs.get('price_analysis', 'N/A')[:100]}...")
            logger.info(f"Strategy: {recs.get('strategy', 'N/A')[:100]}...")
            logger.info(f"Risks: {recs.get('risk_considerations', 'N/A')[:100]}...")
        else:
            logger.error(f"LLM failed: {result.get('error')}")
            logger.info(f"Fallback: {result.get('fallback_recommendations', {})}")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
