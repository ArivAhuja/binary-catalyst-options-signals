from dataclasses import dataclass
from datetime import date
from typing import Literal, ClassVar
from polygon.rest.models import OptionsContract
from py_vollib.black_scholes.implied_volatility import implied_volatility
from py_lets_be_rational.exceptions import BelowIntrinsicException
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, time
from zoneinfo import ZoneInfo
import configparser
from polygon import RESTClient

config = configparser.ConfigParser()
config.read('config/config.ini')
api_key = config['polygon']['api_key']

# Initialize client
client = RESTClient(api_key)

@dataclass(frozen=True)  # frozen=True makes it immutable and suitable as a dictionary key
class Option:
    """Class representing an options contract."""
    ticker: str
    strike_price: float
    expiration_date: datetime
    contract_type: Literal['call', 'put']
    underlying_ticker: str

    def __eq__(self, other):
        if not isinstance(other, Option):
            return NotImplemented
        return (self.ticker == other.ticker and 
                self.strike_price == other.strike_price and 
                self.expiration_date == other.expiration_date and
                self.contract_type == other.contract_type and
                self.underlying_ticker == other.underlying_ticker)
    
    def __lt__(self, other):
        if not isinstance(other, Option):
            return NotImplemented
        return self.strike_price < other.strike_price
    
    def __str__(self) -> str:
        return f"{self.underlying_ticker} {self.contract_type.upper()} ${self.strike_price} {self.expiration_date}"
    
    @classmethod
    def from_polygon_contract(cls, contract: OptionsContract) -> 'Option':
        """Create an Option instance from a Polygon OptionsContract."""
        expiry_date = datetime.strptime(contract.expiration_date, "%Y-%m-%d").date()
        expiry_datetime = datetime.combine(
            expiry_date, 
            time(16, 0),
            tzinfo=ZoneInfo("America/New_York")
        )
        return cls(
            ticker=contract.ticker,
            strike_price=contract.strike_price,
            expiration_date=expiry_datetime,
            contract_type=contract.contract_type,
            underlying_ticker=contract.underlying_ticker,
        )
    
    def to_key(self) -> tuple:
        """Get a tuple key representation of this option contract."""
        return (
            self.ticker,
            self.strike_price,
            self.expiration_date,
            self.contract_type,
            self.underlying_ticker
        )
    
    def get_risk_free_rate(self, curr_date: datetime):
        """Get risk-free rate from treasury yields based on option expiry."""
        option_expiry_days = (self.expiration_date - curr_date).days
        
        try:
            # Fetch most recent yield data
            yield_data = next(client.vx.list_treasury_yields(order="desc", sort="date", limit=1), None)
            
            if not yield_data:
                return 0.04  # Default rate if no data
            
            # Define yield attribute mappings by priority for each expiry range
            expiry_to_yields = {
                range(0, 31): ['yield_1_month', 'yield_3_month', 'yield_1_year'],
                range(31, 91): ['yield_3_month', 'yield_1_month', 'yield_1_year'],
                range(91, 366): ['yield_1_year', 'yield_2_year', 'yield_3_month'],
                range(366, 731): ['yield_2_year', 'yield_1_year', 'yield_5_year'],
                range(731, 1826): ['yield_5_year', 'yield_2_year', 'yield_10_year'],
                range(1826, 100000): ['yield_10_year', 'yield_30_year', 'yield_5_year']
            }
            
            # Get the right yield attributes to try based on expiry
            for expiry_range, yield_attrs in expiry_to_yields.items():
                if option_expiry_days in expiry_range:
                    # Try each yield attribute in priority order
                    for attr in yield_attrs:
                        if hasattr(yield_data, attr) and getattr(yield_data, attr) is not None:
                            return getattr(yield_data, attr) / 100.0
            
            # Last resort: use any available yield
            for attr in ['yield_1_month', 'yield_3_month', 'yield_1_year', 
                        'yield_2_year', 'yield_5_year', 'yield_10_year', 'yield_30_year']:
                if hasattr(yield_data, attr) and getattr(yield_data, attr) is not None:
                    return getattr(yield_data, attr) / 100.0
                    
        except Exception as e:
            print(f'Treasury yield error: {e}')
        
        # Ultimate fallback
        return 0.04 
            

    def calculate_implied_volatility(self, option_price: float, stock_price: float, curr_date: datetime):
        risk_free_rate = self.get_risk_free_rate(curr_date)
        time_to_expiry = (self.expiration_date - curr_date).days / 365.0
        flag = 'c' if self.contract_type == 'call' else 'p'
        try:
            iv = implied_volatility(
                price=option_price, 
                S=stock_price, 
                K=self.strike_price, 
                t=time_to_expiry, 
                r=risk_free_rate,
                flag=flag
            )
        except BelowIntrinsicException:
            return None
        return iv
        
        
