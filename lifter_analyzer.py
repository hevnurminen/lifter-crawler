import pandas as pd
import requests
import json
import logging
from datetime import datetime
import time
from typing import Dict, List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def query_ollama(prompt: str, model: str = "deepseek-r1:7b") -> Optional[str]:
    """
    Send a query to the locally running Ollama instance
    Returns the complete response as a string
    """
    url = "http://localhost:11434/api/generate"
    
    data = {
        "model": model,
        "prompt": prompt,
        "stream": True
    }
    
    try:
        response = requests.post(url, json=data, stream=True)
        response.raise_for_status()
        
        full_response = ""
        for line in response.iter_lines():
            if line:
                json_response = json.loads(line)
                if "response" in json_response:
                    full_response += json_response["response"]
        
        return full_response.strip()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error occurred: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing response: {e}")
        return None

def analyze_lifter_progress(df: pd.DataFrame) -> str:
    """
    Create a prompt for analyzing lifter progress
    """
    # Sort by date to ensure chronological order
    df = df.sort_values('date')
    
    # Create a detailed prompt for the analysis
    prompt = f"""Analyze this weightlifter's competition history and provide a detailed analysis of their progress:

Competition History:
{df.to_string()}

Please provide a detailed analysis including:
1. Overall progress in their lifts (snatch, clean & jerk, total)
2. Notable achievements or records
3. Weight class changes if any
4. Competition frequency
5. Areas of improvement or decline
6. Best performances

Format the analysis in a clear, structured way."""

    return prompt

def process_lifter_data(csv_file: str, num_lifters: int = 10) -> List[Dict]:
    """
    Process the lifter data from the CSV file and analyze their progress
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Extract lifter ID from filename
        lifter_id = csv_file.split('_')[1].split('.')[0]
        
        logger.info(f"Processing lifter ID: {lifter_id}")
        
        # Create analysis prompt
        prompt = analyze_lifter_progress(df)
        
        # Query Ollama for analysis
        analysis = query_ollama(prompt)
        
        if analysis:
            results = [{
                'lifter_id': lifter_id,
                'lifter_name': df['lifter'].iloc[0],
                'analysis': analysis,
                'num_competitions': len(df),
                'best_total': df['total'].max(),
                'best_snatch': df['snatch'].max(),
                'best_clean_jerk': df['clean_jerk'].max()
            }]
            logger.info(f"Successfully analyzed lifter {lifter_id}")
            return results
        else:
            logger.error(f"Failed to analyze lifter {lifter_id}")
            return []
    
    except Exception as e:
        logger.error(f"Error processing lifter data: {str(e)}")
        return []

def save_analysis(results: List[Dict]) -> None:
    """
    Save the analysis results to a CSV file
    """
    if not results:
        logger.warning("No results to save")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"lifter_analysis_{timestamp}.csv"
    
    # Save to CSV
    df.to_csv(filename, index=False, encoding='utf-8')
    logger.info(f"Analysis saved to {filename}")
    
    # Log summary
    logger.info("\n=== Analysis Summary ===")
    logger.info(f"Total lifters analyzed: {len(results)}")
    logger.info(f"Output file: {filename}")

def main():
    # Use specific CSV file
    csv_file = r"C:\Users\Heku\code\ollama\lifter_90_20250315_131655.csv"
    
    if not os.path.exists(csv_file):
        logger.error(f"CSV file not found: {csv_file}")
        return
    
    logger.info(f"Using CSV file: {csv_file}")
    
    # Process lifters and get analysis
    results = process_lifter_data(csv_file)
    
    # Save results
    save_analysis(results)

if __name__ == "__main__":
    import os
    main() 