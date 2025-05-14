"""
Biotech Options Data Collector

This Cloud Function fetches and processes options contract data from the Polygon API for biotech catalyst events.
It retrieves both call and put options for each event in a biotech catalysts dataset stored in Google Cloud Storage,
along with corresponding stock price data, and saves the processed data for further analysis.
"""

import pandas as pd
import tempfile
import math
import time
import gc
import os
import logging
from polygon import RESTClient
from typing import Literal, List, Dict, Any, Tuple
from datetime import timedelta, datetime, time
from zoneinfo import ZoneInfo
from google.cloud import storage
from google.cloud import secretmanager
from option import Option

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hardcoded configuration values
BUCKET_NAME = 'biotech-options-data'
PROJECT_ID = 'biotech-option-signals'

def setup_gcs_client():
    """Set up Google Cloud Storage client."""
    return storage.Client()

def get_polygon_api_key():
    """Get Polygon API key from Secret Manager."""
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{PROJECT_ID}/secrets/polygon_api/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode('UTF-8')

def list_blobs_in_gcs(prefix=None):
    """List all blobs in the Cloud Storage bucket with optional prefix."""
    storage_client = setup_gcs_client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blobs = bucket.list_blobs(prefix=prefix)
    return [blob.name for blob in blobs]

def blob_exists_in_gcs(blob_name):
    """Check if a blob exists in the Cloud Storage bucket."""
    storage_client = setup_gcs_client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)
    return blob.exists()

def upload_to_gcs(source_file_name, destination_blob_name):
    """Upload a file to Cloud Storage bucket."""
    storage_client = setup_gcs_client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    logger.info(f"File uploaded to gs://{BUCKET_NAME}/{destination_blob_name}.")

def download_from_gcs(source_blob_name, destination_file_name):
    """Download a file from Cloud Storage bucket."""
    storage_client = setup_gcs_client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    logger.info(f"Downloaded gs://{BUCKET_NAME}/{source_blob_name} to {destination_file_name}.")

def fetch_contracts_at_date(client, ticker, option_type, as_of_datetime, announcement_datetime, max_retries=3):
    """
    Fetch options contracts for a specific ticker as of a particular date.
    Includes retry logic for API failures.
    """
    # Determine if announcement is after market close (4:00 PM ET)
    market_close_time = time(16, 0, 0, tzinfo=ZoneInfo('America/New_York'))
    if announcement_datetime.time() > market_close_time:
        # For after-hours announcements, use next day as effective date for options
        announcement_date = (announcement_datetime + timedelta(days=1)).date()
    else:
        announcement_date = announcement_datetime.date()
    
    # Convert as_of to date only
    as_of_date = as_of_datetime.date()
    
    retry_count = 0
    while retry_count < max_retries:
        try:
            # Fetch the contracts from Polygon API
            contracts = []
            for c in client.list_options_contracts(
                underlying_ticker=ticker,
                contract_type=option_type,
                expiration_date_gte=announcement_date,
                as_of=as_of_date,
                strike_price_gte=0,
                order='asc',
                limit=1000,
                sort='expiration_date'
                ):
                contracts.append(c)
            
            return contracts
        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                logger.error(f"Failed to fetch contracts for {ticker} after {max_retries} attempts: {str(e)}")
                raise
            wait_time = 2 ** retry_count  # Exponential backoff
            logger.warning(f"Attempt {retry_count} failed for {ticker}, retrying in {wait_time}s: {str(e)}")
            time.sleep(wait_time)

def fetch_unique_contracts_over_time(client, ticker, option_type, start_date, announcement_datetime, max_retries=3):
    """
    Fetch unique options contracts for a ticker over a period of time.
    Includes error handling and retry logic.
    """
    increment = timedelta(1)  # Daily increment
    all_contracts_dict = {}   # Dictionary to store unique contracts by ticker
    current_start_date = start_date
    
    retries_left = max_retries
    
    # Iterate through each day from start_date to announcement_date
    while current_start_date < announcement_datetime and current_start_date < datetime.now(ZoneInfo('UTC')):
        try:
            contracts = fetch_contracts_at_date(client, ticker, option_type, current_start_date, announcement_datetime)
            
            # Add contracts to dictionary, using contract ticker as key to ensure uniqueness
            for contract in contracts:
                all_contracts_dict[contract.ticker] = contract
            
            current_start_date += increment
            retries_left = max_retries  # Reset retries on success
            
        except Exception as e:
            retries_left -= 1
            if retries_left <= 0:
                logger.error(f"Failed to fetch contracts for {ticker} on {current_start_date}: {str(e)}")
                break
            
            wait_time = 2 ** (max_retries - retries_left)  # Exponential backoff
            logger.warning(f"Error fetching contracts, retrying in {wait_time}s: {str(e)}")
            time.sleep(wait_time)
    
    # Return list of unique contracts
    return list(all_contracts_dict.values())

def fetch_contract_stock_data(client, contract, start_date, end_date, increment, max_retries=3):
    """
    Fetch historical price data for a specific options contract.
    Includes error handling and retry logic.
    """
    # Adjust end date for API query (exclusive)
    query_end_date = end_date - timedelta(days=1)
    contract_agg_list = []
    
    retry_count = 0
    while retry_count < max_retries:
        try:
            # Fetch aggregate data from Polygon API
            for a in client.list_aggs(
                ticker=contract.ticker,
                multiplier=increment,
                timespan='day',
                from_=start_date,
                to=query_end_date,
                adjusted=True,
                sort='asc',
                limit=50,
            ):
                contract_agg_list.append(a)
            
            break  # Success, exit the retry loop
        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                logger.error(f"Failed to fetch data for {contract.ticker} after {max_retries} attempts: {str(e)}")
                return pd.DataFrame()  # Return empty DataFrame on failure
            wait_time = 2 ** retry_count  # Exponential backoff
            logger.warning(f"Attempt {retry_count} failed for {contract.ticker}, retrying in {wait_time}s: {str(e)}")
            time.sleep(wait_time)
    
    # Return empty DataFrame if no data found
    if not contract_agg_list:
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(contract_agg_list)
    
    # Process timestamps and set as index
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.set_index('timestamp')
    
    # Remove unnecessary columns
    if 'otc' in df.columns:
        df = df.drop(columns=['otc'])
    
    # Convert Polygon contract to our Option class
    option = Option.from_polygon_contract(contract)
    
    # Create MultiIndex columns with Option object as first level
    original_columns = df.columns.tolist()
    multi_columns = [(option, col) for col in original_columns]
    df.columns = pd.MultiIndex.from_tuples(multi_columns)
    
    return df

def fetch_stock_vwap(client, ticker, start_date, end_date, increment, max_retries=3):
    """
    Fetch volume-weighted average price (VWAP) data for an underlying stock.
    Includes error handling and retry logic.
    """
    # Adjust end date for API query and ensure we don't query future dates
    query_end_date = min(end_date - timedelta(days=1), datetime.now(ZoneInfo('UTC')))
    stock_agg_list = []
    
    # Validate date range
    if start_date > query_end_date:
        raise ValueError('The event date is too far in future, increase lookback')
    
    retry_count = 0  
    while retry_count < max_retries:
        try:
            # Fetch aggregate data from Polygon API
            for a in client.list_aggs(
                ticker=ticker,
                multiplier=increment,
                timespan='day',
                from_=start_date,
                to=query_end_date,
                adjusted=True,
                sort='asc',
                limit=50,
            ):
                stock_agg_list.append(a)
            
            break  # Success, exit the retry loop
        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                logger.error(f"Failed to fetch VWAP for {ticker} after {max_retries} attempts: {str(e)}")
                raise
            wait_time = 2 ** retry_count  # Exponential backoff
            logger.warning(f"Attempt {retry_count} failed for {ticker} VWAP, retrying in {wait_time}s: {str(e)}")
            time.sleep(wait_time)

    # Convert to DataFrame
    df = pd.DataFrame(stock_agg_list)

    # Process timestamps and set as index
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.set_index('timestamp')

    # Return the VWAP series and the index
    return df['vwap'], df.index

def clean_final_df(df, event_date, option_type):
    """
    Clean and format the final options data DataFrame.
    """
    df = df.copy()
    df = df.sort_index()

    # Fill NA values for volume and transaction columns with zeros
    volume_cols = [col for col in df.columns if col[1] in ['volume', 'transactions', 'stock_vwap']]
    for col in volume_cols:
        df[col] = df[col].fillna(0)
    
    # Forward and backward fill price columns to handle missing values
    price_cols = [col for col in df.columns if col[1] in ['open', 'high', 'low', 'close', 'vwap']]
    for col in price_cols:
        df[col] = df[col].ffill().bfill()

    # Sort columns by strike price (ascending for calls, descending for puts)
    sorted_cols = sorted(df.columns, key=lambda col: col[0].strike_price, reverse=option_type == 'put')

    df = df[sorted_cols]
    
    return df

def process_event(client, event, lookback, increment, option_types, dry_run=False, force=False):
    """
    Process a single biotech catalyst event to fetch and save options data to GCS.
    """
    event_result = {
        "ticker": event.ticker,
        "date": event.Index.strftime('%Y-%m-%d'),
        "success": False,
        "details": {},
        "error": None
    }
    
    try:
        # Fetch stock VWAP data for reference
        logger.info(f"Fetching stock VWAP data for {event.ticker}")
        stock_vwap, standard_index = fetch_stock_vwap(
            client, 
            event.ticker, 
            event.Index - timedelta(lookback), 
            event.Index, 
            increment
        )
        
        # Process each option type
        for option_type in option_types:
            # Check if output file already exists in GCS
            output_path = f'option_data/{event.ticker.lower()}_{event.Index.strftime("%Y-%m-%d %H:%M:%S")}_{option_type}_{int(event.result)}.pkl'
            
            # Check if the file exists in GCS
            if blob_exists_in_gcs(output_path) and not force:
                logger.info(f"Skipping {event.ticker} {option_type} (already exists). Use force=True to re-fetch.")
                event_result["details"][option_type] = "skipped (already exists)"
                continue
                
            logger.info(f"Processing {event.ticker} {option_type} options for event on {event.Index.strftime('%Y-%m-%d')}")
            
            if dry_run:
                logger.info(f"  [DRY RUN] Would process {event.ticker} {option_type} options")
                event_result["details"][option_type] = "dry run - would process"
                continue
            
            # Fetch unique contracts over the lookback period
            contracts = fetch_unique_contracts_over_time(
                client,
                event.ticker, 
                option_type, 
                event.Index - timedelta(lookback), 
                event.Index
            )
            
            if not contracts:
                logger.warning(f"  No contracts found for {event.ticker} {option_type}")
                event_result["details"][option_type] = "no contracts found"
                continue
                
            logger.info(f"  Found {len(contracts)} unique contracts")
            
            agg_dfs = []
            
            # Process each contract
            for contract in contracts:
                # Fetch historical data for this contract
                agg_df = fetch_contract_stock_data(
                    client,
                    contract, 
                    event.Index - timedelta(lookback), 
                    event.Index, 
                    increment
                )
                
                # If data exists, process it
                if not agg_df.empty:
                    # Reindex to match standard index
                    agg_df = agg_df.reindex(standard_index)
                    
                    # Add stock VWAP for reference
                    stock_vwap_reindexed = stock_vwap.reindex(agg_df.index)
                    agg_df[(Option.from_polygon_contract(contract), 'stock_vwap')] = stock_vwap_reindexed
                    
                    agg_dfs.append(agg_df)
                
                # Clear some memory
                del agg_df
                gc.collect()
            
            # Combine all contract data into a single DataFrame
            if agg_dfs:
                option_contracts_df = pd.concat(agg_dfs, axis=1)
                
                # Clean and format the final DataFrame
                option_contracts_df = clean_final_df(option_contracts_df, event.Index, option_type)
                
                # Save to GCS using a temporary file
                with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_file:
                    option_contracts_df.to_pickle(temp_file.name)
                    temp_file.flush()
                    temp_file_path = temp_file.name
                
                try:
                    upload_to_gcs(temp_file_path, output_path)
                    logger.info(f"  Saved data to gs://{BUCKET_NAME}/{output_path}")
                    event_result["details"][option_type] = f"saved {len(agg_dfs)} contracts"
                finally:
                    # Clean up the temporary file
                    os.unlink(temp_file_path)
                
                del option_contracts_df
                gc.collect()
            else:
                logger.warning(f"  No data found for {event.ticker} {option_type} options")
                event_result["details"][option_type] = "no data found"
        
        event_result["success"] = True
        return event_result
        
    except Exception as e:
        logger.error(f"Error processing {event.ticker}: {str(e)}")
        event_result["error"] = str(e)
        return event_result

def collect_options_data(request):
    """
    Cloud Function entry point for collecting options data.
    """
    # Start timing the function execution
    start_time = time.time()
    max_runtime_seconds = 540  # 9-minute max for HTTP functions (with buffer)
    
    # Parse request parameters
    request_json = request.get_json(silent=True)
    request_args = request.args
    
    # Get parameters with fallbacks
    if request_json:
        catalyst = request_json.get('catalyst')
        option_type = request_json.get('option_type', 'both')
        lookback = int(request_json.get('lookback', 98))
        increment = int(request_json.get('increment', 14))
        dry_run = request_json.get('dry_run', False)
        force = request_json.get('force', False)
    elif request_args:
        catalyst = request_args.get('catalyst')
        option_type = request_args.get('option_type', 'both')
        lookback = int(request_args.get('lookback', 98)) if request_args.get('lookback') else 98
        increment = int(request_args.get('increment', 14)) if request_args.get('increment') else 14
        dry_run = request_args.get('dry_run', '').lower() == 'true'
        force = request_args.get('force', '').lower() == 'true'
    else:
        # Default values if no parameters provided
        catalyst = None
        option_type = 'both'
        lookback = 98
        increment = 14
        dry_run = False
        force = False
    
    logger.info(f"Starting options data collection with parameters: catalyst={catalyst}, option_type={option_type}, lookback={lookback}, dry_run={dry_run}, force={force}")
    
    # Get API key from Secret Manager
    try:
        api_key = get_polygon_api_key()
        logger.info("Successfully retrieved Polygon API key")
    except Exception as e:
        error_msg = f"Failed to retrieve API key: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}, 500
    
    # Initialize Polygon API client
    client = RESTClient(api_key)
    
    # Ensure lookback is a multiple of increment
    lookback = math.ceil(lookback/increment) * increment
    
    # Determine which option types to process
    option_types = []
    if option_type in ['call', 'both']:
        option_types.append('call')
    if option_type in ['put', 'both']:
        option_types.append('put')
    
    # Load biotech catalyst events data from GCS
    try:
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_file:
            download_from_gcs(
                source_blob_name='biotech_catalysts/biotech_catalysts_data.pkl',
                destination_file_name=temp_file.name
            )
            temp_file_path = temp_file.name
        
        biotech_catalysts_df = pd.read_pickle(temp_file_path)
        os.unlink(temp_file_path)
    except Exception as e:
        error_msg = f"Failed to load biotech catalysts data: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}, 500
    
    # Filter by catalyst ticker if specified
    if catalyst:
        biotech_catalysts_df = biotech_catalysts_df[biotech_catalysts_df.ticker == catalyst.upper()]
        if len(biotech_catalysts_df) == 0:
            return {"error": f"No events found for ticker {catalyst.upper()}"}, 404
    
    logger.info(f"Processing {len(biotech_catalysts_df)} events with {len(option_types)} option types")
    
    results = []
    completed_tickers = []
    
    # Process each biotech catalyst event
    for i, event in enumerate(biotech_catalysts_df.itertuples()):
        # Check if we're approaching the time limit
        if time.time() - start_time > max_runtime_seconds - 30:  # 30-second buffer
            remaining_tickers = [e.ticker for e in biotech_catalysts_df.iloc[i:].itertuples()]
            logger.warning(f"Time limit approaching, stopping after {i} of {len(biotech_catalysts_df)} events")
            return {
                "status": "partial_completion",
                "message": "Time limit approaching, stopping execution",
                "completed_events": i,
                "total_events": len(biotech_catalysts_df),
                "completed_tickers": completed_tickers,
                "remaining_tickers": remaining_tickers,
                "results": results
            }
        
        logger.info(f"Processing event {i+1}/{len(biotech_catalysts_df)}: {event.ticker} on {event.Index.strftime('%Y-%m-%d')}")
        
        result = process_event(
            client=client,
            event=event,
            lookback=lookback,
            increment=increment,
            option_types=option_types,
            dry_run=dry_run,
            force=force
        )
        
        results.append(result)
        if result["success"]:
            completed_tickers.append(event.ticker)
        
        # Wait between processing different events to avoid API rate limits
        if i < len(biotech_catalysts_df) - 1 and not dry_run:
            wait_time = 5  # Shorter wait time for cloud environment
            logger.info(f"Waiting {wait_time} seconds before processing next event...")
            time.sleep(wait_time)
    
    # Calculate execution time
    execution_time = time.time() - start_time
    
    return {
        "status": "complete",
        "message": "Processing complete!",
        "execution_time_seconds": execution_time,
        "processed_events": len(results),
        "successful_events": sum(1 for r in results if r["success"]),
        "results": results
    }