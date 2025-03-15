from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import time
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create a thread pool for Selenium operations
thread_pool = ThreadPoolExecutor(max_workers=2)

def setup_driver():
    """
    Set up and return a configured Chrome driver
    """
    logger.info("Setting up Chrome options...")
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    logger.info("Initializing Chrome driver...")
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=chrome_options
    )
    return driver

async def get_lifter_data_async(lifter_id: int, max_retries: int = 3) -> Optional[List[Dict]]:
    """
    Asynchronously fetch and parse lifter data from SPNL website using Selenium
    """
    url = f"https://tilasto.painonnosto.fi/lifter.php?lifter_id={lifter_id}"
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries} for lifter {lifter_id}")
            logger.info(f"Fetching URL: {url}")
            
            # Run Selenium operations in thread pool
            loop = asyncio.get_event_loop()
            driver = await loop.run_in_executor(thread_pool, setup_driver)
            
            try:
                # Navigate to URL
                await loop.run_in_executor(thread_pool, lambda: driver.get(url))
                
                # Wait for table to be present
                logger.info("Waiting for table to load...")
                wait = WebDriverWait(driver, 10)
                table = await loop.run_in_executor(
                    thread_pool,
                    lambda: wait.until(EC.presence_of_element_located((By.TAG_NAME, "table")))
                )
                
                # Additional wait to ensure JavaScript content is loaded
                await asyncio.sleep(3)
                
                # Get the page source after JavaScript has loaded
                page_source = await loop.run_in_executor(thread_pool, lambda: driver.page_source)
                soup = BeautifulSoup(page_source, 'html.parser')
                
                # Find the table
                table = soup.find('table')
                if not table:
                    logger.warning("No table found on the page")
                    logger.debug(f"Page content preview: {soup.prettify()[:1000]}")
                    if attempt < max_retries - 1:
                        logger.info("Retrying...")
                        continue
                    return None
                
                # Extract data
                data = []
                rows = table.find_all('tr')[1:]  # Skip header row
                
                if not rows:
                    logger.warning("No data rows found in table")
                    logger.debug(f"Table HTML: {table.prettify()}")
                    if attempt < max_retries - 1:
                        logger.info("Retrying...")
                        continue
                    return None
                
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 12:  # Ensure we have all columns
                        # Check if we're getting template variables
                        if any('{{' in col.text.strip() for col in cols):
                            logger.warning("Found template variables instead of actual values")
                            logger.info("This suggests the data is loaded via JavaScript")
                            if attempt < max_retries - 1:
                                logger.info("Retrying...")
                                continue
                            return None
                        
                        row_data = {
                            'date': cols[0].text.strip(),
                            'lifter': cols[1].text.strip(),
                            'birthday': cols[2].text.strip(),
                            'club': cols[3].text.strip(),
                            'location': cols[4].text.strip(),
                            'sex': cols[5].text.strip(),
                            'class': cols[6].text.strip(),
                            'bodyweight': cols[7].text.strip(),
                            'snatch': cols[8].text.strip(),
                            'clean_jerk': cols[9].text.strip(),
                            'total': cols[10].text.strip(),
                            'sinclair': cols[11].text.strip()
                        }
                        data.append(row_data)
                
                if data:
                    logger.info(f"Successfully found {len(data)} records for lifter {lifter_id}")
                    return data
                else:
                    logger.warning("No valid data found in table")
                    if attempt < max_retries - 1:
                        logger.info("Retrying...")
                        continue
                    return None
                
            finally:
                # Ensure driver is properly closed
                await loop.run_in_executor(thread_pool, lambda: driver.quit())
            
        except Exception as e:
            logger.error(f"Error during data extraction for lifter {lifter_id}: {str(e)}")
            if attempt < max_retries - 1:
                logger.info("Retrying...")
                continue
            return None
    
    return None

async def save_to_csv_async(data: List[Dict], lifter_id: int) -> bool:
    """
    Asynchronously save the data to a CSV file
    Returns True if data was saved, False if data was empty
    """
    if not data:
        return False
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"lifter_{lifter_id}_{timestamp}.csv"
    
    # Save to CSV
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(thread_pool, lambda: df.to_csv(filename, index=False, encoding='utf-8'))
    logger.info(f"Data saved to {filename}")
    
    # Log first few rows for verification
    logger.info("\nFirst few rows of saved data:")
    logger.info(df.head())
    
    return True

async def process_lifter(lifter_id: int) -> bool:
    """
    Process a single lifter's data
    Returns True if data was found and saved, False if no data was found
    """
    logger.info(f"Processing lifter ID: {lifter_id}")
    
    data = await get_lifter_data_async(lifter_id)
    
    if data:
        logger.info(f"Found {len(data)} lifting records for lifter {lifter_id}")
        return await save_to_csv_async(data, lifter_id)
    else:
        logger.warning(f"No data found for lifter {lifter_id}")
        return False

async def merge_csv_files() -> None:
    """
    Merge all CSV files in the current directory into a single file
    """
    try:
        # Get all CSV files that match our pattern
        csv_files = [f for f in os.listdir('.') if f.startswith('lifter_') and f.endswith('.csv')]
        
        if not csv_files:
            logger.warning("No CSV files found to merge")
            return
        
        # Read and concatenate all CSV files
        dfs = []
        for file in csv_files:
            df = pd.read_csv(file)
            dfs.append(df)
        
        # Concatenate all dataframes
        merged_df = pd.concat(dfs, ignore_index=True)
        
        # Save merged data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        merged_filename = f"merged_lifters_{timestamp}.csv"
        merged_df.to_csv(merged_filename, index=False, encoding='utf-8')
        
        logger.info(f"Successfully merged {len(csv_files)} files into {merged_filename}")
        logger.info(f"Total records in merged file: {len(merged_df)}")
        
        # Clean up individual CSV files
        for file in csv_files:
            os.remove(file)
            logger.info(f"Removed individual file: {file}")
            
    except Exception as e:
        logger.error(f"Error merging CSV files: {str(e)}")

async def main():
    # Track successful and failed lifter IDs
    successful_ids = []
    failed_ids = []
    last_successful_id = None
    consecutive_empty_count = 0
    batch_size = 10  # Process 10 lifters at a time
    current_id = 1
    max_consecutive_empty = 10  # Stop after 10 consecutive empty responses
    
    while True:
        # Create tasks for current batch
        tasks = [process_lifter(lifter_id) for lifter_id in range(current_id, current_id + batch_size)]
        
        # Run batch concurrently and collect results
        results = await asyncio.gather(*tasks)
        
        # Process results
        for lifter_id, success in enumerate(results, start=current_id):
            if success:
                successful_ids.append(lifter_id)
                last_successful_id = lifter_id
                consecutive_empty_count = 0  # Reset counter on success
            else:
                failed_ids.append(lifter_id)
                consecutive_empty_count += 1
        
        # Log batch summary
        logger.info(f"\n=== Batch Summary (IDs {current_id}-{current_id + batch_size - 1}) ===")
        logger.info(f"Successfully processed lifters: {[id for id in range(current_id, current_id + batch_size) if id in successful_ids]}")
        logger.info(f"Skipped lifters (no data): {[id for id in range(current_id, current_id + batch_size) if id in failed_ids]}")
        logger.info(f"Consecutive empty responses: {consecutive_empty_count}")
        
        # Check if we should stop
        if consecutive_empty_count >= max_consecutive_empty:
            logger.info(f"\nStopping after {max_consecutive_empty} consecutive empty responses")
            break
        
        # Move to next batch
        current_id += batch_size
        
        # Add a small delay between batches to be polite to the server
        await asyncio.sleep(2)
    
    # Log final summary
    logger.info("\n=== Final Processing Summary ===")
    logger.info(f"Last successful lifter ID: {last_successful_id}")
    logger.info(f"Total successful lifters: {len(successful_ids)}")
    logger.info(f"Total skipped lifters: {len(failed_ids)}")
    logger.info(f"Successfully processed lifters: {sorted(successful_ids)}")
    logger.info(f"Skipped lifters: {sorted(failed_ids)}")
    
    # Merge all CSV files
    logger.info("\nMerging CSV files...")
    await merge_csv_files()
    
    # Clean up thread pool
    thread_pool.shutdown(wait=True)

if __name__ == "__main__":
    asyncio.run(main()) 