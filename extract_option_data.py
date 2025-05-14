"""
Biotech Options Data Collector

This script fetches and processes options contract data from the Polygon API for biotech catalyst events. 
It retrieves both call and put options for each event in a biotech catalysts dataset, along with 
corresponding stock price data, and saves the processed data as pickle files for further analysis.

The script is designed to analyze options behavior around key biotech events such as FDA decisions and
clinical trial results announcements.

Dependencies:
    - pandas: Data manipulation
    - configparser: Reading API keys from config file
    - polygon: Polygon.io API client for financial data
    - zoneinfo: Timezone handling
    - Option: Custom class for option contract representation
    - argparse: Command-line argument parsing

Usage:
    python collect_options_data.py [OPTIONS]

Options:
    --catalyst TICKER    Process only the specified ticker
    --option-type TYPE   Process only 'call', 'put', or 'both' (default: both)
    --lookback DAYS      Lookback period in days (default: 98)
    --increment DAYS     Aggregation increment in days (default: 14)
    --dry-run            Run without saving data, just print what would be processed
    --force              Re-fetch data even if it already exists

Output:
    Pickle files in 'option_data/' directory with naming pattern:
    {ticker}_{event_datetime}_{option_type}_{result}.pkl
    
    Where:
    - ticker: Company stock symbol
    - event_datetime: Date and time of the biotech catalyst event
    - option_type: Either 'call' or 'put'
    - result: Event outcome (1 for positive, -1 for negative, 0 for pending)
"""

import pandas as pd
import configparser
from polygon import RESTClient
from polygon.rest.models import OptionsContract, Agg
from typing import Literal, List, Dict, Any, Tuple
from datetime import timedelta, datetime, time, date
from zoneinfo import ZoneInfo
from option import Option
import time as t
import math
import argparse
import os

# Load configuration with API key
config = configparser.ConfigParser()
config.read('config/config.ini')
api_key = config['polygon']['api_key']

# Initialize Polygon API client
client = RESTClient(api_key)

def fetch_contracts_at_date(ticker: str, option_type: Literal['call', 'put'], 
                          as_of_datetime: datetime, announcement_datetime: datetime) -> List[OptionsContract]:
    """
    Fetch options contracts for a specific ticker as of a particular date.

    Parameters:
        ticker (str): The underlying stock ticker symbol.
        option_type (Literal['call', 'put']): Type of option contract to fetch.
        as_of_datetime (datetime): Date to fetch contracts as they existed on this date.
        announcement_datetime (datetime): The date of the catalyst event announcement.

    Returns:
        List[OptionsContract]: List of option contracts available as of the specified date.
        
    Notes:
        - For after-market announcements, uses the next trading day for options expiration filtering.
        - Only returns contracts that expire on or after the announcement date.
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
    
    # Fetch the contracts from Polygon API
    contracts = []
    for c in client.list_options_contracts(
        underlying_ticker=ticker,
        contract_type=option_type,
        expiration_date_gte=announcement_date,  # Only get options expiring on or after the announcement
        as_of=as_of_date,
        strike_price_gte=0,
        order='asc',
        limit=1000,
        sort='expiration_date'
        ):
        contracts.append(c)
    
    return contracts

def fetch_unique_contracts_over_time(ticker: str, option_type: Literal['call', 'put'], 
                                  start_date: datetime, announcement_datetime: datetime) -> List[OptionsContract]:
    """
    Fetch unique options contracts for a ticker over a period of time.
    
    This function collects contracts day by day from start_date to announcement_date,
    ensuring we capture all unique contracts that existed during that period.

    Parameters:
        ticker (str): The underlying stock ticker symbol.
        option_type (Literal['call', 'put']): Type of option contract to fetch.
        start_date (datetime): Initial date to start fetching contracts.
        announcement_datetime (datetime): The date of the catalyst event announcement.

    Returns:
        List[OptionsContract]: List of unique option contracts available over the time period.
    """
    increment = timedelta(1)  # Daily increment
    all_contracts_dict = {}   # Dictionary to store unique contracts by ticker
    current_start_date = start_date
    
    # Iterate through each day from start_date to announcement_date
    while current_start_date < announcement_datetime and current_start_date < datetime.now(ZoneInfo('UTC')):
        contracts = fetch_contracts_at_date(ticker, option_type, current_start_date, announcement_datetime)
        
        # Add contracts to dictionary, using contract ticker as key to ensure uniqueness
        for contract in contracts:
            all_contracts_dict[contract.ticker] = contract
        
        current_start_date += increment
    
    # Return list of unique contracts
    return list(all_contracts_dict.values())

def fetch_contract_stock_data(contract: OptionsContract, start_date: datetime, 
                           end_date: datetime, increment: int) -> pd.DataFrame:
    """
    Fetch historical price data for a specific options contract.

    Parameters:
        contract (OptionsContract): The options contract to fetch data for.
        start_date (datetime): Start date for the historical data.
        end_date (datetime): End date for the historical data.
        increment (int): Number of days to aggregate in each data point.

    Returns:
        pd.DataFrame: DataFrame containing the historical price data with MultiIndex columns.
                    First level is Option object, second level is the data type (open, high, etc.)
                    Returns empty DataFrame if no data is available.
    """
    # Adjust end date for API query (exclusive)
    query_end_date = end_date - timedelta(days=1)
    query_end_date = min(end_date - timedelta(days=1), datetime.now(ZoneInfo('UTC')))
    contract_agg_list = []

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

    # Return empty DataFrame if no data found
    if not contract_agg_list:
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(contract_agg_list)
    
    # Process timestamps and set as index
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.set_index('timestamp')
    
    # Remove unnecessary columns
    df = df.drop(columns=['otc'])
    
    # Convert Polygon contract to our Option class
    option = Option.from_polygon_contract(contract)
    
    # Create MultiIndex columns with Option object as first level
    original_columns = df.columns.tolist()
    multi_columns = [(option, col) for col in original_columns]
    df.columns = pd.MultiIndex.from_tuples(multi_columns)
    
    return df

def fetch_stock_vwap(ticker: str, start_date: datetime, end_date: datetime, increment: int) -> Tuple[pd.Series, pd.Index]:
    """
    Fetch volume-weighted average price (VWAP) data for an underlying stock.

    Parameters:
        ticker (str): The stock ticker symbol.
        start_date (datetime): Start date for the historical data.
        end_date (datetime): End date for the historical data.
        increment (int): Number of days to aggregate in each data point.

    Returns:
        Tuple[pd.Series, pd.Index]: 
            - pd.Series: Series containing VWAP values
            - pd.Index: DateTimeIndex of the data points

    Raises:
        ValueError: If the event date is too far in the future relative to lookback period.
    """
    # Adjust end date for API query and ensure we don't query future dates
    query_end_date = min(end_date - timedelta(days=1), datetime.now(ZoneInfo('UTC')))
    stock_agg_list = []
    
    # Validate date range
    if start_date > query_end_date:
        raise ValueError('The event date is too far in future, increase lookback')
    
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

    # Convert to DataFrame
    df = pd.DataFrame(stock_agg_list)

    # Process timestamps and set as index
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.set_index('timestamp')

    # Return the VWAP series and the index
    return df['vwap'], df.index

def clean_final_df(df: pd.DataFrame, event_date: datetime, option_type: str) -> pd.DataFrame:
    """
    Clean and format the final options data DataFrame.

    Parameters:
        df (pd.DataFrame): The raw DataFrame containing options data.
        event_date (datetime): The date of the catalyst event.
        option_type (str): Either 'call' or 'put'.

    Returns:
        pd.DataFrame: Cleaned and formatted DataFrame.
        
    Notes:
        - Fills NA values for volume and transaction columns with zeros
        - Forward and backward fills price data to handle missing values
        - Sorts columns by strike price (ascending for calls, descending for puts)
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

    # Uncomment to convert index to days before event
    # df.index = [(event_date - idx).days for idx in df.index]
    
    return df

def process_event(event, lookback, increment, option_types, dry_run=False, force=False):
    """
    Process a single biotech catalyst event to fetch and save options data.
    
    Parameters:
        event: A row from the biotech catalysts DataFrame (as namedtuple)
        lookback: Number of days to look back before the event
        increment: Aggregation increment in days
        option_types: List of option types to process ('call', 'put', or both)
        dry_run: If True, don't save data, just print info
        force: If True, re-fetch data even if it exists
    
    Returns:
        bool: True if successful, False if error
    """
    try:
        # Fetch stock VWAP data for reference
        stock_vwap, standard_index = fetch_stock_vwap(
            event.ticker, 
            event.Index - timedelta(lookback), 
            event.Index, 
            increment
        )
        
        # Process each option type
        for option_type in option_types:
            # Check if output file already exists
            output_path = f'option_data/{event.ticker.lower()}_{event.Index.strftime("%Y-%m-%d %H:%M:%S")}_{option_type}_{int(event.result)}.pkl'
            
            if os.path.exists(output_path) and not force:
                print(f"Skipping {event.ticker} {option_type} (already exists). Use --force to re-fetch.")
                continue
                
            print(f"Processing {event.ticker} {option_type} options for event on {event.Index.strftime('%Y-%m-%d')}")
            
            if dry_run:
                print(f"  [DRY RUN] Would process {event.ticker} {option_type} options")
                continue
            
            # Fetch unique contracts over the lookback period
            contracts = fetch_unique_contracts_over_time(
                event.ticker, 
                option_type, 
                event.Index - timedelta(lookback), 
                event.Index
            )
            
            if not contracts:
                print(f"  No contracts found for {event.ticker} {option_type}")
                continue
                
            print(f"  Found {len(contracts)} unique contracts")
            
            agg_dfs = []
            
            # Process each contract
            for contract in contracts:
                # Fetch historical data for this contract
                agg_df = fetch_contract_stock_data(
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
            
            # Combine all contract data into a single DataFrame
            if agg_dfs:
                option_contracts_df = pd.concat(agg_dfs, axis=1)
                
                # Clean and format the final DataFrame
                option_contracts_df = clean_final_df(option_contracts_df, event.Index, option_type)
                
                # Save to pickle file
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                option_contracts_df.to_pickle(output_path)
                print(f"  Saved data to {output_path}")
            else:
                print(f"  No data found for {event.ticker} {option_type} options")
        
        return True
        
    except Exception as e:
        print(f"Error processing {event.ticker}: {str(e)}")
        return False

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Fetch options data for biotech catalyst events')
    parser.add_argument('--catalyst', type=str, help='Process only the specified ticker')
    parser.add_argument('--option-type', type=str, choices=['call', 'put', 'both'], default='both',
                        help='Process only calls, only puts, or both (default: both)')
    parser.add_argument('--lookback', type=int, default=98, 
                        help='Lookback period in days (default: 98)')
    parser.add_argument('--increment', type=int, default=14,
                        help='Aggregation increment in days (default: 14)')
    parser.add_argument('--dry-run', action='store_true', 
                        help='Run without saving data, just print what would be processed')
    parser.add_argument('--force', action='store_true',
                        help='Re-fetch data even if it already exists')
    parser.add_argument('--function', type=str, choices=['fetch', 'analyze_iv', 'analyze_skew', 'cp_ratio', 'implied_move'],
                        default='fetch', help='Function to run (default: fetch)')
    
    args = parser.parse_args()
    
    # If the function is something other than fetch, import analysis_option_data and run the appropriate function
    if args.function != 'fetch':
        try:
            import analysis_option_data
            
            print(f"Running {args.function}...")
            
            if args.function == 'analyze_iv':
                analysis_option_data.analyze_and_save_all_implied_volatility()
            elif args.function == 'analyze_skew':
                analysis_option_data.analyze_and_save_iv_skew()
            elif args.function == 'cp_ratio':
                analysis_option_data.calculate_and_save_call_put_ratio()
            elif args.function == 'implied_move':
                analysis_option_data.save_implied_move()
                
            print(f"Completed {args.function}")
            return
            
        except ImportError:
            print("analysis_option_data module not found. Make sure it's in the same directory.")
            return
        except Exception as e:
            print(f"Error running {args.function}: {str(e)}")
            return
    
    # Otherwise, proceed with the fetch function
    
    # Ensure lookback is a multiple of increment
    lookback = math.ceil(args.lookback/args.increment) * args.increment
    
    # Determine which option types to process
    option_types = []
    if args.option_type in ['call', 'both']:
        option_types.append('call')
    if args.option_type in ['put', 'both']:
        option_types.append('put')
    
    # Load biotech catalyst events data
    biotech_catalysts_df = pd.read_pickle('biotech_catalysts/biotech_catalysts_data.pkl')
    
    # Filter by catalyst ticker if specified
    if args.catalyst:
        biotech_catalysts_df = biotech_catalysts_df[biotech_catalysts_df.ticker == args.catalyst.upper()]
        if len(biotech_catalysts_df) == 0:
            print(f"No events found for ticker {args.catalyst.upper()}")
            return
    
    print(f"Processing {len(biotech_catalysts_df)} events with {len(option_types)} option types")
    print(f"Lookback: {lookback} days, Increment: {args.increment} days")
    if args.dry_run:
        print("DRY RUN MODE: No data will be saved")
    
    # Process each biotech catalyst event
    for i, event in enumerate(biotech_catalysts_df.itertuples()):
        print(f"Event {i+1}/{len(biotech_catalysts_df)}: {event.ticker} on {event.Index.strftime('%Y-%m-%d')}")
        
        success = process_event(
            event=event,
            lookback=lookback,
            increment=args.increment,
            option_types=option_types,
            dry_run=args.dry_run,
            force=args.force
        )
        
        # Wait between processing different events to avoid API rate limits
        if i < len(biotech_catalysts_df) - 1 and success and not args.dry_run:
            print(f"Waiting 30 seconds before processing next event...")
            t.sleep(30)
    
    print("Processing complete!")
    
if __name__ == '__main__':
    main()