"""
Option Contract Module

This module defines the Option class for representing financial options contracts, 
providing functionality for implied volatility calculations and integration with 
Polygon.io's API. It serves as a core component for the biotech catalyst analysis 
system, enabling the analysis of options data around key biotech events.

The Option class handles:
- Representation of call and put options with their key properties
- Conversion from Polygon.io API option contract format
- Dynamic risk-free rate calculation based on treasury yield data
- Black-Scholes implied volatility calculation with proper error handling

Dependencies:
    - dataclasses: For immutable structured data representation
    - polygon: For API connection to fetch financial data
    - py_vollib: For Black-Scholes option pricing calculations
    - py_lets_be_rational: For accurate implied volatility calculations

Usage:
    option = Option.from_polygon_contract(polygon_contract)
    iv = option.calculate_implied_volatility(price, stock_price, date)
"""

from dataclasses import dataclass
from typing import Literal
from polygon.rest.models import OptionsContract
from py_vollib.black_scholes.implied_volatility import implied_volatility
from py_lets_be_rational.exceptions import BelowIntrinsicException
from datetime import datetime, time
from zoneinfo import ZoneInfo
import configparser
from polygon import RESTClient

# Load Polygon API key from configuration file
config = configparser.ConfigParser()
config.read('config/config.ini')
api_key = config['polygon']['api_key']

# Initialize Polygon API client
client = RESTClient(api_key)

@dataclass(frozen=True)  # frozen=True makes it immutable and suitable as a dictionary key
class Option:
    """
    Class representing an options contract with all essential attributes.
    
    The class is immutable (frozen) to allow instances to be used as dictionary keys
    in data structures that track option metrics over time.
    
    Attributes:
        ticker (str): The specific option contract ticker symbol
        strike_price (float): The strike price of the option
        expiration_date (datetime): The expiration date and time (with timezone)
        contract_type (Literal['call', 'put']): Whether the option is a call or put
        underlying_ticker (str): The ticker symbol of the underlying stock
    """
    ticker: str
    strike_price: float
    expiration_date: datetime
    contract_type: Literal['call', 'put']
    underlying_ticker: str

    def __eq__(self, other):
        """
        Determine if two Option objects are equal based on their attributes.
        
        Args:
            other: Another object to compare with this Option
            
        Returns:
            bool: True if the two options have identical attributes, False otherwise
        """
        if not isinstance(other, Option):
            return NotImplemented
        return (self.ticker == other.ticker and 
                self.strike_price == other.strike_price and 
                self.expiration_date == other.expiration_date and
                self.contract_type == other.contract_type and
                self.underlying_ticker == other.underlying_ticker)
    
    def __lt__(self, other):
        """
        Compare two Option objects based on strike price for sorting.
        
        Args:
            other: Another Option object to compare with this one
            
        Returns:
            bool: True if this option's strike price is less than the other's
        """
        if not isinstance(other, Option):
            return NotImplemented
        return self.strike_price < other.strike_price
    
    def __str__(self) -> str:
        """
        Create a human-readable string representation of the option.
        
        Returns:
            str: A formatted string showing the option's key details
                 Example: "AAPL CALL $150.00 2023-06-16 16:00:00-04:00"
        """
        return f"{self.underlying_ticker} {self.contract_type.upper()} ${self.strike_price} {self.expiration_date}"
    
    @classmethod
    def from_polygon_contract(cls, contract: OptionsContract) -> 'Option':
        """
        Create an Option instance from a Polygon OptionsContract object.
        
        This factory method converts the date-only expiration from Polygon's API
        into a full datetime with the standard options market close time (4:00 PM ET).
        
        Args:
            contract (OptionsContract): A contract object from Polygon.io API
            
        Returns:
            Option: A new Option instance with data from the Polygon contract
        """
        expiry_date = datetime.strptime(contract.expiration_date, "%Y-%m-%d").date()
        expiry_datetime = datetime.combine(
            expiry_date, 
            time(16, 0),  # Options expire at market close (4:00 PM)
            tzinfo=ZoneInfo("America/New_York")
        )
        return cls(
            ticker=contract.ticker,
            strike_price=contract.strike_price,
            expiration_date=expiry_datetime,
            contract_type=contract.contract_type,
            underlying_ticker=contract.underlying_ticker,
        )
    
    def get_risk_free_rate(self, curr_date: datetime):
        """
        Get the appropriate risk-free interest rate based on the option's time to expiry.
        
        This method fetches the most recent U.S. Treasury yield data from Polygon.io
        and selects the appropriate yield based on the option's time to expiration.
        
        Args:
            curr_date (datetime): The current date for calculating time to expiry
            
        Returns:
            float: The annual risk-free rate as a decimal (e.g., 0.04 for 4%)
            
        Note:
            - Uses Treasury yields that best match the option's time to expiry
            - Falls back to alternative durations if the preferred one is unavailable
            - Uses 4% as a default rate if no yield data can be retrieved
        """
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
                            return getattr(yield_data, attr) / 100.0  # Convert percentage to decimal
            
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
        """
        Calculate the implied volatility of this option using the Black-Scholes model.
        
        This method uses the py_vollib library to solve for the implied volatility
        that would produce the observed market price, given the other parameters.
        
        Args:
            option_price (float): The market price of the option contract
            stock_price (float): The current price of the underlying stock
            curr_date (datetime): The date for which to calculate IV (for time to expiry)
            
        Returns:
            float or None: The implied volatility as a decimal (e.g., 0.25 for 25%),
                           or None if the calculation fails due to the option price
                           being below intrinsic value or other numerical issues
                           
        Note:
            Implied volatility is a key metric that represents the market's expectation
            of future volatility of the underlying stock, derived from option prices.
        """
        # Get the risk-free rate appropriate for this option's expiry
        risk_free_rate = self.get_risk_free_rate(curr_date)
        
        # Calculate time to expiry in years
        time_to_expiry = (self.expiration_date - curr_date).days / 365.0
        
        # Set flag for py_vollib (c=call, p=put)
        flag = 'c' if self.contract_type == 'call' else 'p'
        
        try:
            # Calculate implied volatility using py_vollib
            iv = implied_volatility(
                price=option_price, 
                S=stock_price, 
                K=self.strike_price, 
                t=time_to_expiry, 
                r=risk_free_rate,
                flag=flag
            )
            return iv
        except BelowIntrinsicException:
            # Option priced below intrinsic value, cannot compute valid IV
            return None