"""
Option Contract Module

This module defines the Option class for representing financial options contracts, 
providing functionality for implied volatility calculations.
"""

from dataclasses import dataclass
from typing import Literal
from polygon.rest.models import OptionsContract
from py_vollib.black_scholes.implied_volatility import implied_volatility
from py_lets_be_rational.exceptions import BelowIntrinsicException
from datetime import datetime, time
from zoneinfo import ZoneInfo
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass(frozen=True)  # frozen=True makes it immutable and suitable as a dictionary key
class Option:
    """
    Class representing an options contract with all essential attributes.
    """
    ticker: str
    strike_price: float
    expiration_date: datetime
    contract_type: Literal['call', 'put']
    underlying_ticker: str

    def __eq__(self, other):
        """Compare two Option objects for equality"""
        if not isinstance(other, Option):
            return NotImplemented
        return (self.ticker == other.ticker and 
                self.strike_price == other.strike_price and 
                self.expiration_date == other.expiration_date and
                self.contract_type == other.contract_type and
                self.underlying_ticker == other.underlying_ticker)
    
    def __lt__(self, other):
        """Compare two Option objects based on strike price"""
        if not isinstance(other, Option):
            return NotImplemented
        return self.strike_price < other.strike_price
    
    def __hash__(self):
        """Generate a hash value for the Option instance"""
        return hash((self.ticker, self.strike_price, self.expiration_date, 
                    self.contract_type, self.underlying_ticker))
    
    def __str__(self) -> str:
        """Human-readable string representation"""
        return f"{self.underlying_ticker} {self.contract_type.upper()} ${self.strike_price} {self.expiration_date}"
    
    @classmethod
    def from_polygon_contract(cls, contract: OptionsContract) -> 'Option':
        """Create an Option instance from a Polygon OptionsContract object"""
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
        """Get a fixed risk-free rate (simplified version)"""
        # Using a fixed 4% rate for simplicity
        return 0.04
            
    def calculate_implied_volatility(self, option_price: float, stock_price: float, curr_date: datetime):
        """Calculate the implied volatility using Black-Scholes model"""
        # Validate inputs
        if option_price <= 0 or stock_price <= 0:
            logger.warning(f"Invalid price inputs: option_price={option_price}, stock_price={stock_price}")
            return None
        
        # Fixed risk-free rate
        risk_free_rate = 0.04
        
        # Calculate time to expiry in years
        time_to_expiry = (self.expiration_date - curr_date).days / 365.0
        
        # Avoid calculation errors with expired options
        if time_to_expiry <= 0.0001:
            logger.warning(f"Option expired or very close to expiry: {time_to_expiry} years")
            return None
        
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
            # Option priced below intrinsic value
            return None
        except Exception as e:
            logger.warning(f"Failed to calculate IV for {self.ticker}: {str(e)}")
            return None