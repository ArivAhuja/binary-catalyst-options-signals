# Binary Catalyst Options Signals

An analytical framework for studying options market behavior around binary biotech catalyst events such as FDA decisions and clinical trial results.

## Overview

This project analyzes options market activity before and after high-impact biotech catalysts to identify patterns in implied volatility, volume, and other metrics that might indicate market expectations or informed trading. It leverages the Polygon.io API to collect options data around specific biotech events and provides tools for various analyses.

## Features

- Collection and storage of biotech catalyst event data (FDA decisions, clinical trial results)
- Fetching historical options data from Polygon.io API
- Calculation of volume-weighted implied volatility
- Volatility skew analysis (downside skew, upside skew, smile curvature)
- Call/put ratio tracking
- Implied move calculations using ATM straddles
- Detection of elevated options trading volume
- Visualization of key metrics over time

## Folder Structure

- `biotech_catalysts/`: Contains structured data about biotech catalyst events
- `config/`: Configuration files, including API keys
- `iv_matrix/`: Implied volatility matrices by strike price and date
- `option_analysis/`: Output of analyses including IV skew, call/put ratio, and implied moves
- `option_data/`: Raw options data fetched from Polygon.io API
- `venv/`: Python virtual environment

## Main Components

### 1. Data Collection

- `extract_biotech_catalyst_data.py`: Creates and maintains the dataset of biotech catalyst events
- `extract_option_data.py`: Fetches options contract data from Polygon.io API for specified tickers and dates

### 2. Data Analysis

- `analysis_option_data.py`: Provides tools for analyzing options data, including:
  - Implied volatility calculation
  - Volatility skew metrics
  - Call/put ratio analysis
  - Implied move calculation
  - Volume spike detection

### 3. Core Classes

- `option.py`: Defines the Option class which represents options contracts and provides methods for implied volatility calculations

## Setup and Installation

1. Clone the repository
   ```
   git clone https://github.com/yourusername/binary-catalyst-options-signals.git
   cd binary-catalyst-options-signals
   ```

2. Create and activate a virtual environment
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```
   pip install -r requirements.txt
   ```

4. Set up your configuration
   - Create `config/config.ini` with your Polygon.io API key:
     ```
     [polygon]
     api_key = YOUR_API_KEY_HERE
     ```

## Command-Line Usage

### 1. Creating or updating the biotech catalysts dataset

```bash
python extract_biotech_catalyst_data.py
```

This will create a pickle file at `biotech_catalysts/biotech_catalysts_data.pkl` containing structured information about biotech catalyst events.

### 2. Fetching options data for these events

```bash
# Fetch data for all catalyst events
python extract_option_data.py

# Fetch data for a specific ticker
python extract_option_data.py --catalyst MRNA

# Fetch only call options with a custom lookback period
python extract_option_data.py --option-type call --lookback 60

# Preview what would be processed without saving data
python extract_option_data.py --dry-run

# Re-fetch data even if it already exists
python extract_option_data.py --force

# Run a specific analysis function directly
python extract_option_data.py --function analyze_iv
```

Available options:
- `--catalyst TICKER`: Process only the specified ticker
- `--option-type TYPE`: Process only 'call', 'put', or 'both' (default: both)
- `--lookback DAYS`: Lookback period in days (default: 98)
- `--increment DAYS`: Aggregation increment in days (default: 14)
- `--dry-run`: Run without saving data, just print what would be processed
- `--force`: Re-fetch data even if it already exists
- `--function FUNCTION`: Run a specific analysis function after data collection

### 3. Running analysis

```bash
# Run all analyses (default if no function specified)
python analysis_option_data.py

# Run specific analyses
python analysis_option_data.py --analyze-iv --cp-ratio

# Process only a specific ticker
python analysis_option_data.py --all --ticker MRNA

# Print detailed progress information
python analysis_option_data.py --analyze-skew --verbose

# Force re-analysis even if output files exist
python analysis_option_data.py --implied-move --force
```

Available analysis functions:
- `--analyze-iv`: Calculate and save implied volatility
- `--analyze-skew`: Calculate IV skew metrics
- `--cp-ratio`: Calculate call/put volume ratio
- `--implied-move`: Calculate expected price moves
- `--iv-matrix`: Calculate IV matrix by strike
- `--plot-smiles`: Plot volatility smile curves
- `--all`: Run all analyses

Additional options:
- `--ticker TICKER`: Process only the specified ticker
- `--verbose`: Print detailed progress information
- `--force`: Re-run analysis even if output files exist

## Example Outputs

The analysis generates various visualizations and data files in the `option_analysis/` directory:

- Implied volatility time series
- Volatility skew metrics
- Call/put ratio charts
- Implied move expectations

## Dependencies

- pandas, numpy: Data manipulation
- matplotlib: Visualization
- polygon-api-client: API access to financial data
- py_vollib: Black-Scholes option pricing calculations
- py_lets_be_rational: Accurate implied volatility calculations
- tqdm: Progress tracking for long-running calculations
- argparse: Command-line argument parsing
