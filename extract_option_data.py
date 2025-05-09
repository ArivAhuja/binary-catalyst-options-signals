import numpy as np
import pandas as pd
import configparser
from polygon import RESTClient
from polygon.rest.models import OptionsContract, Agg
from typing import Literal, List, Dict, Any
from datetime import timedelta, datetime, time
from zoneinfo import ZoneInfo
from loguru import logger

config = configparser.ConfigParser()
config.read('config/config.ini')
api_key = config['polygon']['api_key']

# Initialize client
client = RESTClient(api_key)

def fetch_contracts_at_date(ticker: str, option_type: Literal['call', 'put'], as_of_datetime: datetime, announcement_datetime: datetime) -> List[OptionsContract]:
    market_close_time = time(16, 0, 0, tzinfo=ZoneInfo('America/New_York'))
    if announcement_datetime.time() > market_close_time:
        announcement_date = (announcement_datetime + timedelta(days=1)).date()
    else:
        announcement_date = announcement_datetime.date()
    as_of_date = as_of_datetime.date()
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

def fetch_unique_contracts_over_time(ticker: str, option_type: Literal['call', 'put'], start_date: datetime, announcement_datetime: datetime, increment: int) -> List[OptionsContract]:
    increment = timedelta(increment)
    all_contracts_dict = {}
    current_start_date = start_date
    while current_start_date < announcement_datetime:
        contracts = fetch_contracts_at_date(ticker, option_type, announcement_datetime, current_start_date)
        for contract in contracts:
            all_contracts_dict[contract.ticker] = contract
        current_start_date += increment
    
    return list(all_contracts_dict.values())


def fetch_contract_data(contract: OptionsContract, start_date: datetime, end_date: datetime, increment: int) -> pd.DataFrame:
    agg_list = []
    query_end_date = end_date - timedelta(days=1)
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
        agg_list.append(a)
    
    if not agg_list:
        return pd.DataFrame()
    
    df = pd.DataFrame(agg_list)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')
    days_before_event = [(ts.date() - end_date.date()).days for ts in df.index]
    df.index = days_before_event
    df = df.sort_index()
    
    df = df.drop(columns=['otc'])
    
    # Create contract key tuple
    contract_key = (
        contract.ticker,
        contract.strike_price,
        contract.expiration_date,
        contract.contract_type,
        contract.underlying_ticker
    )
    
    # Create MultiIndex columns
    original_columns = df.columns.tolist()
    multi_columns = [(contract_key, col) for col in original_columns]
    df.columns = pd.MultiIndex.from_tuples(multi_columns)
    
    return df


def clean_final_df(df: pd.DataFrame, event_date: datetime) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_index()


    # First, clean volume and transactions columns
    volume_cols = [col for col in df.columns if col[1] in ['volume', 'transactions']]
    for col in volume_cols:
        df[col] = df[col].fillna(0)
    
    # Price columns forward and backward fill
    price_cols = [col for col in df.columns if col[1] in ['open', 'high', 'low', 'close', 'vwap']]
    for col in price_cols:
        df[col] = df[col].ffill().bfill()
    
    return df
    
if __name__ == '__main__':
    lookback, increment = 98, 14
    biotech_catalysts_df = pd.read_pickle('biotech_catalysts.pkl')
    option_types = ['call', 'put']
    for option_type in option_types:
        for event in biotech_catalysts_df.itertuples():
            contracts = fetch_unique_contracts_over_time(event.ticker, option_type, event.Index - timedelta(lookback), event.Index, increment)
            standard_index = np.arange(-lookback, 0, increment)
            agg_dfs = []
            for contract in contracts:
                agg_df = fetch_contract_data(contract, event.Index - timedelta(lookback), event.Index, increment)
                if not agg_df.empty:
                    agg_df = agg_df.reindex(standard_index)
                    agg_dfs.append(agg_df)
            option_contracts_df = pd.concat(agg_dfs, axis=1)
            option_contracts_df = clean_final_df(option_contracts_df, event.Index)
            option_contracts_df.to_pickle(f'option_data/{event.ticker.lower()}_{option_type}_option_data.pkl')