import os
from typing import List, Dict, Any
import json
import google.generativeai as genai
from dotenv import load_dotenv

class GeminiProvider:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found. Add it to .env file")
        genai.configure(api_key=api_key)
        #self.model = genai.GenerativeModel('gemini-pro')
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
        
    def generate_response(self, schema_context: str, user_prompt: str) -> str:
        """Generate a response using Gemini with enhanced graph awareness."""
        try:
            # Enhanced prompt template
            prompt = f'''100% valid Arangodb AQL. No exceptions. Using the following graph database schema and relationships:
{schema_context}

Generate an AQL query for this request: {user_prompt}

Key Points:
1. Use proper graph traversal when joining collections through relationships
2. Always place LIMIT before RETURN
3. Include field names exactly as shown in schema
4. For complex joins, use edge collections defined in relationships

Return only the AQL query without any explanation or markdown.'''

            print(prompt)
            response = self.model.generate_content(prompt)
            text1=self._clean_aql_response(response.text) 
            print(text1)
            return text1
        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")

    def _clean_aql_response(self, response: str) -> str:
        """Clean and validate the AQL response."""
        # Remove any markdown formatting
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("aql"):
                response = response[3:]
        response = response.strip("` \n")
        
        # Basic AQL validation
        if not response.upper().startswith("FOR"):
            raise ValueError("Invalid AQL: Query must start with FOR")
            
        if "RETURN" not in response.upper():
            raise ValueError("Invalid AQL: Query must include RETURN statement")
            
        return response

    def explain_results(self, results: List[Dict[str, Any]], query: str) -> str:
        """Generate explanation for query results."""
        try:
            prompt = f"""Analyze these supply chain query results in the context of:
- Supplier relationships
- Risk factors
- Parts and dependencies
- Inventory levels

Query: {query}
Results: {json.dumps(results, indent=2)}

Provide insights focusing on:
1. Key findings (2-3 points)
2. Notable patterns
3. Business implications
"""
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error explaining results: {str(e)}"
                
    def suggest_visualization(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Suggest visualization configuration."""
        try:
            prompt = """Analyze this supply chain data and suggest a visualization.
Choose from:
- Bar chart for comparisons
- Line chart for trends
- Scatter plot for relationships
- Pie chart for proportions

Return exactly this JSON format:
{
    "chart_type": "bar|line|pie|scatter",
    "config": {
        "data": {
            "x": ["x_values"],
            "y": ["y_values"]
        },
        "layout": {
            "title": "chart_title"
        }
    }
}"""
            
            response = self.model.generate_content(f"{prompt}\nData: {json.dumps(results, indent=2)}")
            return self._parse_visualization_response(response.text)
        except Exception as e:
            return self._generate_default_visualization()


    def _parse_visualization_response(self, response: str) -> Dict[str, Any]:
        """Parse and validate visualization response."""
        try:
            text = response.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            config = json.loads(text.strip())
            
            # Validate required fields
            if not all(key in config for key in ["chart_type", "config"]):
                raise ValueError("Missing required fields in visualization config")
            if not all(key in config["config"] for key in ["data", "layout"]):
                raise ValueError("Missing required fields in config")
                
            return config
        except Exception:
            return self._generate_default_visualization()

    def _generate_default_visualization(self) -> Dict[str, Any]:
        """Generate a safe default visualization configuration."""
        return {
            "chart_type": "bar",
            "config": {
                "data": {
                    "x": ["No Data"],
                    "y": [0]
                },
                "layout": {
                    "title": "Data Visualization"
                }
            }
        }
    
def get_llm_provider() -> GeminiProvider:
    return GeminiProvider()