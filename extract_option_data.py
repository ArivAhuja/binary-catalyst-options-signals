import numpy as np
import pandas as pd
import configparser
from polygon import RESTClient
from polygon.rest.models import OptionsContract, Agg
from typing import Literal, List, Dict, Any, Tuple
from datetime import timedelta, datetime, time
from zoneinfo import ZoneInfo
from loguru import logger
from option import Option

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


def fetch_contract_stock_data(contract: OptionsContract, start_date: datetime, end_date: datetime, increment: int) -> pd.DataFrame:
    query_end_date = end_date - timedelta(days=1)
    contract_agg_list = []
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
    
    if not contract_agg_list:
        return pd.DataFrame()
    
    df = pd.DataFrame(contract_agg_list)
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.set_index('timestamp')
    
    df = df.drop(columns=['otc'])
    
    option = Option.from_polygon_contract(contract)
    
    # Create MultiIndex columns
    original_columns = df.columns.tolist()
    multi_columns = [(option, col) for col in original_columns]
    df.columns = pd.MultiIndex.from_tuples(multi_columns)
    
    return df

def fetch_stock_vwap(ticker: str, start_date: datetime, end_date: datetime, increment: int) -> Tuple[pd.Series, pd.Index]:
    query_end_date = end_date - timedelta(days=1)
    stock_agg_list = []
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

    df = pd.DataFrame(stock_agg_list)

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.set_index('timestamp')

    return df['vwap'], df.index

def clean_final_df(df: pd.DataFrame, event_date: datetime, option_type: str) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_index()

    # First, clean volume and transactions columns
    volume_cols = [col for col in df.columns if col[1] in ['volume', 'transactions', 'stock_vwap']]
    for col in volume_cols:
        df[col] = df[col].fillna(0)
    
    # Price columns forward and backward fill
    price_cols = [col for col in df.columns if col[1] in ['open', 'high', 'low', 'close', 'vwap']]
    for col in price_cols:
        df[col] = df[col].ffill().bfill()

    sorted_cols = sorted(df.columns, key=lambda col: col[0].strike_price, reverse=option_type == 'put')

    df = df[sorted_cols]
    
    return df
    
if __name__ == '__main__':
    lookback, increment = 98, 14
    biotech_catalysts_df = pd.read_pickle('biotech_catalysts/biotech_catalysts_data.pkl')
    option_types = ['call', 'put']
    for event in biotech_catalysts_df.itertuples():
        stock_vwap, standard_index = fetch_stock_vwap(event.ticker, event.Index - timedelta(lookback), event.Index, increment)
        for option_type in option_types:
            contracts = fetch_unique_contracts_over_time(event.ticker, option_type, event.Index - timedelta(lookback), event.Index, increment)
            agg_dfs = []
            for contract in contracts:
                agg_df = fetch_contract_stock_data(contract, event.Index - timedelta(lookback), event.Index, increment)
                if not agg_df.empty:
                    agg_df = agg_df.reindex(standard_index)
                    stock_vwap = stock_vwap.reindex(agg_df.index)
                    agg_df[(Option.from_polygon_contract(contract), 'stock_vwap')] = stock_vwap
                    agg_dfs.append(agg_df)
            option_contracts_df = pd.concat(agg_dfs, axis=1)
            option_contracts_df = clean_final_df(option_contracts_df, event.Index, option_type)
            option_contracts_df.to_pickle(f'option_data/{event.ticker.lower()}_{option_type}_option_data.pkl')