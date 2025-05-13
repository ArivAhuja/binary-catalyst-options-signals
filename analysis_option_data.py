"""
Biotech Options Analysis Toolkit

This module provides comprehensive tools for analyzing options data related to biotech catalyst events.
It focuses on analyzing implied volatility patterns, volume changes, volatility skew, and implied moves
around key biotech catalysts such as FDA decisions and clinical trial results.

The module works with options data collected using the biotech_catalysts and polygon API scripts,
and provides various analytical functions and visualization capabilities:

Key Features:
    - Implied volatility calculation and visualization for call and put options
    - Volume-weighted implied volatility analysis across strike prices
    - IV skew metrics calculation (downside skew, upside skew, smile curvature)
    - Call/put ratio analysis to detect sentiment shifts
    - Implied move computation based on ATM straddle pricing
    - Detection of elevated options trading volume
    - Advanced visualization of all metrics over time

Dependencies:
    - pandas, numpy: Data manipulation
    - matplotlib: Visualization
    - tqdm: Progress tracking for long-running calculations
    - Option: Custom class for option contract representation (must support calculate_implied_volatility method)
    - argparse: Command-line argument parsing

Usage:
    python analysis_option_data.py [FUNCTION] [OPTIONS]
    
    Functions:
        --analyze-iv        Calculate and save implied volatility
        --analyze-skew      Calculate IV skew metrics
        --cp-ratio          Calculate call/put volume ratio
        --implied-move      Calculate expected price moves
        --iv-matrix         Calculate IV matrix by strike
        --plot-smiles       Plot volatility smile curves
        --all               Run all analyses (default if no function specified)

    Options:
        --ticker TICKER     Process only the specified ticker
        --verbose           Print detailed progress information
        --force             Re-run analysis even if output files exist

Output:
    Results are saved to the 'option_analysis' directory with standardized naming conventions:
    {ticker}_{date}_{metric}.{extension}
    
    Where:
    - ticker: Company stock symbol
    - date: Date of the catalyst event
    - metric: Type of analysis (iv, iv_skew, cp_ratio, implied_move)
    - extension: File format (.pkl for data, .png for visualizations, .csv for easy inspection)
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pickle
import argparse
import sys

def create_implied_volatility_vector(df) -> pd.DataFrame:
    """
    Calculate volume-weighted implied volatility for each timestamp.
    
    Parameters:
    - df: DataFrame containing options data
    
    Returns:
    - DataFrame with timestamp index and implied volatility values
    """
    option_objects = df.columns.get_level_values(0).unique()
    iv_v = np.zeros(len(df.index))
    
    for i, (idx, row) in enumerate(tqdm(df.iterrows(), total=len(df))):
        weighted_iv, total_volume = 0, 0
        for option in option_objects:
            volume = df.loc[idx, (option, 'volume')]
            if volume == 0:
                continue
            option_price = df.loc[idx, (option, 'vwap')]
            stock_price = df.loc[idx, (option, 'stock_vwap')]
            iv = option.calculate_implied_volatility(
                option_price=option_price,
                stock_price=stock_price,
                curr_date=idx 
            )
            if not iv:
                continue
            weighted_iv += iv * volume
            total_volume += volume
        iv_v[i] = weighted_iv / total_volume if total_volume > 0 else np.nan
    
    # Create DataFrame with the original index and the calculated IV values
    result_df = pd.DataFrame({
        'implied_volatility': iv_v
    }, index=df.index)
    
    return result_df

def save_implied_volatility(call_df, put_df=None):
    """
    Calculate volume-weighted implied volatility for call and/or put options and save to option_analysis directory.
    
    Parameters:
    - call_df: DataFrame containing call options data
    - put_df: Optional DataFrame containing put options data (default: None)
    
    Returns:
    - Dictionary with calculated DataFrame(s)
    """
    output_dir = "option_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    # Process call data
    call_filename = os.path.basename(call_df._metadata['filepath']) if hasattr(call_df, '_metadata') else "unknown_call"
    call_parts = call_filename.split('_')
    ticker = call_parts[0] if len(call_parts) > 0 else "unknown"
    date_str = '_'.join(call_parts[1:3]) if len(call_parts) > 2 else "unknown_date"
    
    print(f"Calculating volume-weighted implied volatility for {ticker} calls...")
    call_iv_df = create_implied_volatility_vector(call_df)
    results['call'] = call_iv_df
    
    # Save call IV
    output_filename = f"{ticker}_{date_str}_call_iv.pkl"
    output_path = os.path.join(output_dir, output_filename)
    call_iv_df.to_pickle(output_path)
    
    # Also save as CSV for easier inspection
    csv_path = os.path.join(output_dir, f"{ticker}_{date_str}_call_iv.csv")
    call_iv_df.to_csv(csv_path)
    
    print(f"Saved call IV data to {output_path}")
    
    # Plot call IV
    plot_title = f"Volume-Weighted Call Implied Volatility - {ticker} {date_str}"
    plot_path = os.path.join(output_dir, f"{ticker}_{date_str}_call_iv.png")
    plot_implied_volatility(call_iv_df, plot_title, plot_path)
    
    # Process put data if provided
    if put_df is not None:
        put_filename = os.path.basename(put_df._metadata['filepath']) if hasattr(put_df, '_metadata') else "unknown_put"
        put_parts = put_filename.split('_')
        
        print(f"Calculating volume-weighted implied volatility for {ticker} puts...")
        put_iv_df = create_implied_volatility_vector(put_df)
        results['put'] = put_iv_df
        
        # Save put IV
        output_filename = f"{ticker}_{date_str}_put_iv.pkl"
        output_path = os.path.join(output_dir, output_filename)
        put_iv_df.to_pickle(output_path)
        
        # Also save as CSV for easier inspection
        csv_path = os.path.join(output_dir, f"{ticker}_{date_str}_put_iv.csv")
        put_iv_df.to_csv(csv_path)
        
        print(f"Saved put IV data to {output_path}")
        
        # Plot put IV
        plot_title = f"Volume-Weighted Put Implied Volatility - {ticker} {date_str}"
        plot_path = os.path.join(output_dir, f"{ticker}_{date_str}_put_iv.png")
        plot_implied_volatility(put_iv_df, plot_title, plot_path)
        
        # If we have both call and put data, calculate and plot combined/average IV
        combined_iv = pd.DataFrame({
            'call_iv': call_iv_df['implied_volatility'],
            'put_iv': put_iv_df['implied_volatility'],
            'avg_iv': (call_iv_df['implied_volatility'] + put_iv_df['implied_volatility']) / 2
        })
        
        results['combined'] = combined_iv
        
        # Save combined IV
        output_filename = f"{ticker}_{date_str}_combined_iv.pkl"
        output_path = os.path.join(output_dir, output_filename)
        combined_iv.to_pickle(output_path)
        
        # Also save as CSV
        csv_path = os.path.join(output_dir, f"{ticker}_{date_str}_combined_iv.csv")
        combined_iv.to_csv(csv_path)
        
        print(f"Saved combined IV data to {output_path}")
        
        # Plot combined IV
        plot_title = f"Volume-Weighted Implied Volatility Comparison - {ticker} {date_str}"
        plot_path = os.path.join(output_dir, f"{ticker}_{date_str}_combined_iv.png")
        plot_combined_iv(combined_iv, plot_title, plot_path)
    
    return results

def plot_implied_volatility(iv_df, title, output_path):
    """
    Create a plot of volume-weighted implied volatility over time.
    
    Parameters:
    - iv_df: DataFrame with implied volatility data
    - title: Title for the plot
    - output_path: Path to save the plot
    """
    plt.figure(figsize=(14, 8))
    
    # Plot IV
    plt.plot(iv_df.index, iv_df['implied_volatility'], color='blue', linewidth=2)
    
    # Format as percentage
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
    
    # Add labels and title
    plt.title(title, fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Volume-Weighted Implied Volatility', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Format x-axis dates
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Saved plot to {output_path}")

def plot_combined_iv(combined_iv, title, output_path):
    """
    Create a plot comparing call, put, and average volume-weighted implied volatility.
    
    Parameters:
    - combined_iv: DataFrame with call_iv, put_iv, and avg_iv columns
    - title: Title for the plot
    - output_path: Path to save the plot
    """
    plt.figure(figsize=(14, 8))
    
    # Plot different IV series
    plt.plot(combined_iv.index, combined_iv['call_iv'], color='green', linewidth=2, label='Call VWIV')
    plt.plot(combined_iv.index, combined_iv['put_iv'], color='red', linewidth=2, label='Put VWIV')
    plt.plot(combined_iv.index, combined_iv['avg_iv'], color='blue', linewidth=2.5, label='Average VWIV')
    
    # Format as percentage
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
    
    # Add labels and title
    plt.title(title, fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Volume-Weighted Implied Volatility', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Format x-axis dates
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Saved combined IV plot to {output_path}")
    
def analyze_and_save_all_implied_volatility(ticker_filter=None, verbose=False, force=False):
    """
    Process all option data file pairs, calculate implied volatility,
    and save results to the option_analysis folder.
    
    Parameters:
    - ticker_filter: Optional ticker to process only specific files
    - verbose: Print detailed information if True
    - force: Re-run analysis even if output files exist
    """
    output_dir = "option_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all option data files
    all_files = glob.glob("option_data/*.pkl")
    
    # Apply ticker filter if specified
    if ticker_filter:
        ticker_filter = ticker_filter.lower()
        all_files = [f for f in all_files if f.lower().startswith(f"option_data/{ticker_filter}_")]
        
        if not all_files:
            print(f"No files found for ticker: {ticker_filter}")
            return
    
    # Identify call files and find their put counterparts
    call_files = [f for f in all_files if "_call_" in os.path.basename(f)]
    
    print(f"Found {len(call_files)} call files to process")
    
    for call_file in tqdm(call_files, desc="Processing files"):
        call_filename = os.path.basename(call_file)
        
        # Check if output already exists
        parts = call_filename.split('_')
        ticker = parts[0]
        date_str = '_'.join(parts[1:3]) if len(parts) > 2 else "unknown_date"
        output_check = os.path.join(output_dir, f"{ticker}_{date_str}_call_iv.pkl")
        
        if os.path.exists(output_check) and not force:
            if verbose:
                print(f"Skipping {call_filename} - output already exists. Use --force to re-run.")
            continue
        
        # Replace "_call_" with "_put_" to find the corresponding put file
        put_filename = call_filename.replace("_call_", "_put_")
        put_file = os.path.join(os.path.dirname(call_file), put_filename)
        
        if os.path.exists(put_file):
            if verbose:
                print(f"\nProcessing pair: {call_filename} and {put_filename}")
            
            try:
                # Load data
                call_df = pd.read_pickle(call_file)
                put_df = pd.read_pickle(put_file)
                
                # Add filepath to metadata for reference
                call_df._metadata = {'filepath': call_file}
                put_df._metadata = {'filepath': put_file}
                
                # Calculate and save implied volatility
                save_implied_volatility(call_df, put_df)
                
            except Exception as e:
                print(f"Error processing {call_filename} and {put_filename}: {str(e)}")
        else:
            if verbose:
                print(f"\nProcessing only call file: {call_filename}")
            
            try:
                # Load data
                call_df = pd.read_pickle(call_file)
                
                # Add filepath to metadata for reference
                call_df._metadata = {'filepath': call_file}
                
                # Calculate and save implied volatility (only for calls)
                save_implied_volatility(call_df)
                
            except Exception as e:
                print(f"Error processing {call_filename}: {str(e)}")
    
    print(f"Implied volatility analysis complete. Results saved to {output_dir}/")

def calculate_iv_matrix(df):
    """
    Calculate implied volatility for each option at each timestamp to create a matrix.
    
    Parameters:
    - df: DataFrame containing options data with MultiIndex columns
    
    Returns:
    - DataFrame with timestamps as index and options as columns, containing IV values
    """
    option_objects = df.columns.get_level_values(0).unique()
    iv_matrix_df = pd.DataFrame(index=df.index, columns=option_objects)
    for i, (idx, row) in enumerate(tqdm(df.iterrows(), total=len(df))):
        for option in option_objects:
            volume = df.loc[idx, (option, 'volume')]
            if volume == 0:
                continue
            option_price = df.loc[idx, (option, 'vwap')]
            stock_price = df.loc[idx, (option, 'stock_vwap')]
            iv = option.calculate_implied_volatility(
                option_price=option_price,
                stock_price=stock_price,
                curr_date=idx 
            )
            if not iv:
                iv = np.nan
            iv_matrix_df.loc[idx, option] = iv

    return iv_matrix_df


def detect_elevated_volume(df, threshold_multiplier=2.0):
    """
    Detect periods of elevated volume relative to a cumulative average.
    
    Parameters:
    - df: DataFrame containing option data
    - threshold_multiplier: How many times the average volume to consider "elevated" (default: 2.0)
    
    Returns:
    - Series with boolean values indicating elevated volume periods
    """
    # Get option objects to iterate through columns
    option_objects = df.columns.get_level_values(0).unique()
    
    # Calculate total volume for each timestamp
    total_volumes = []
    
    for idx in tqdm(df.index, desc="Calculating volumes"):
        row_volume = 0
        for option in option_objects:
            volume = df.loc[idx, (option, 'volume')]
            if not pd.isna(volume):
                row_volume += volume
        total_volumes.append(row_volume)
    
    # Convert to Series with the same index as the DataFrame
    volume_series = pd.Series(total_volumes, index=df.index)
    
    # Initialize elevated volume detection
    elevated_volume = pd.Series(False, index=df.index)
    
    # Need at least 2 points to establish a baseline
    if len(volume_series) < 2:
        return elevated_volume
    
    # Use first two points to establish baseline
    baseline_avg = volume_series.iloc[:2].mean()
    cum_sum = volume_series.iloc[:2].sum()
    cum_count = 2
    
    # Analyze remaining points
    for i in range(2, len(volume_series)):
        # Current volume
        current_volume = volume_series.iloc[i]
        
        # Get cumulative average up to the previous point
        cum_avg = cum_sum / cum_count
        
        # Check if current volume is elevated
        if current_volume >= cum_avg * threshold_multiplier:
            elevated_volume.iloc[i] = True
        
        # Update cumulative metrics for next iteration
        cum_sum += current_volume
        cum_count += 1
    
    return elevated_volume

def calculate_volume_vector(df):
    """
    Calculate total option volume vector for each timestamp.
    
    Parameters:
    - df: DataFrame containing options data with MultiIndex columns
    
    Returns:
    - numpy array containing total option volume for each timestamp
    """
    option_objects = df.columns.get_level_values(0).unique()
    volume_v = np.zeros(len(df.index))
    for i, (idx, row) in enumerate(tqdm(df.iterrows(), total=len(df))):
        total_volume = 0
        for option in option_objects:
            volume = df.loc[idx, (option, 'volume')]
            if volume == 0:
                continue
            option_price = df.loc[idx, (option, 'vwap')]
            stock_price = df.loc[idx, (option, 'stock_vwap')]
            iv = option.calculate_implied_volatility(
                option_price=option_price,
                stock_price=stock_price,
                curr_date=idx 
            )
            if not iv:
                continue
            total_volume += volume
        volume_v[i] = total_volume

    return volume_v

def calculate_iv_skew(call_df, put_df):
    """
    Calculate various IV skew metrics from options data.
    
    Parameters:
    - call_df: DataFrame containing call options data
    - put_df: DataFrame containing put options data
    
    Returns:
    - DataFrame with various skew measurements
    """
    # Create result DataFrame
    result = pd.DataFrame(
        index=call_df.index,
        columns=['downside_skew', 'upside_skew', 'smile_curvature', 'put_call_skew', 'wings_skew']
    )
    
    # Get option objects
    call_options = call_df.columns.get_level_values(0).unique()
    put_options = put_df.columns.get_level_values(0).unique()
    
    # Process each timestamp
    for idx in tqdm(call_df.index, desc="Calculating IV Skew"):
        # Get stock price for this timestamp
        stock_price = None
        for option in call_options:
            if not pd.isna(call_df.loc[idx, (option, 'stock_vwap')]):
                stock_price = call_df.loc[idx, (option, 'stock_vwap')]
                break
        
        if stock_price is None or stock_price == 0:
            result.loc[idx] = [np.nan, np.nan, np.nan, np.nan, np.nan]
            continue
        
        # 1. Create IV curve for calls and puts
        call_strikes = []
        call_ivs = []
        for option in call_options:
            # Get option IV from your calculate_iv_matrix function or existing data
            # This assumes you already have IV values
            if option in call_df.columns.levels[0]:
                iv = call_df.loc[idx, (option, 'iv')] if 'iv' in call_df[option].columns else None
                
                # If IV doesn't exist as a column, calculate it
                if iv is None:
                    volume = call_df.loc[idx, (option, 'volume')]
                    if volume == 0:
                        continue
                    option_price = call_df.loc[idx, (option, 'vwap')]
                    iv = option.calculate_implied_volatility(
                        option_price=option_price,
                        stock_price=stock_price,
                        curr_date=idx 
                    )
                
                if not pd.isna(iv) and iv > 0:
                    call_strikes.append(option.strike_price)
                    call_ivs.append(iv)
        
        put_strikes = []
        put_ivs = []
        for option in put_options:
            if option in put_df.columns.levels[0]:
                iv = put_df.loc[idx, (option, 'iv')] if 'iv' in put_df[option].columns else None
                
                # If IV doesn't exist as a column, calculate it
                if iv is None:
                    volume = put_df.loc[idx, (option, 'volume')]
                    if volume == 0:
                        continue
                    option_price = put_df.loc[idx, (option, 'vwap')]
                    iv = option.calculate_implied_volatility(
                        option_price=option_price,
                        stock_price=stock_price,
                        curr_date=idx 
                    )
                
                if not pd.isna(iv) and iv > 0:
                    put_strikes.append(option.strike_price)
                    put_ivs.append(iv)
        
        # If insufficient data, skip this timestamp
        if len(call_strikes) < 3 or len(put_strikes) < 3:
            result.loc[idx] = [np.nan, np.nan, np.nan, np.nan, np.nan]
            continue
        
        # 2. Calculate various skew metrics
        
        # A. Downside Skew: OTM Put IV - ATM Put IV
        # Find ATM put (closest to spot)
        atm_put_idx = np.abs(np.array(put_strikes) - stock_price).argmin()
        atm_put_iv = put_ivs[atm_put_idx]
        atm_put_strike = put_strikes[atm_put_idx]
        
        # Find 10% OTM put (approximately)
        target_otm_put_strike = stock_price * 0.9
        otm_put_idx = np.abs(np.array(put_strikes) - target_otm_put_strike).argmin()
        otm_put_iv = put_ivs[otm_put_idx]
        
        downside_skew = otm_put_iv - atm_put_iv
        
        # B. Upside Skew: OTM Call IV - ATM Call IV
        # Find ATM call
        atm_call_idx = np.abs(np.array(call_strikes) - stock_price).argmin()
        atm_call_iv = call_ivs[atm_call_idx]
        atm_call_strike = call_strikes[atm_call_idx]
        
        # Find 10% OTM call
        target_otm_call_strike = stock_price * 1.1
        otm_call_idx = np.abs(np.array(call_strikes) - target_otm_call_strike).argmin()
        otm_call_iv = call_ivs[otm_call_idx]
        
        upside_skew = otm_call_iv - atm_call_iv
        
        # C. Smile Curvature: How much IV increases as you move away from ATM
        smile_curvature = downside_skew + upside_skew
        
        # D. Put-Call Skew: ATM Put IV - ATM Call IV
        put_call_skew = atm_put_iv - atm_call_iv
        
        # E. Wings Skew: Far OTM Put IV vs. Far OTM Call IV (typically 25-delta options)
        # Here approximating with 20% OTM options
        far_otm_put_strike = stock_price * 0.8
        far_otm_put_idx = np.abs(np.array(put_strikes) - far_otm_put_strike).argmin()
        far_otm_put_iv = put_ivs[far_otm_put_idx]
        
        far_otm_call_strike = stock_price * 1.2
        far_otm_call_idx = np.abs(np.array(call_strikes) - far_otm_call_strike).argmin()
        far_otm_call_iv = call_ivs[far_otm_call_idx]
        
        wings_skew = far_otm_put_iv - far_otm_call_iv
        
        # Store results
        result.loc[idx] = [
            downside_skew,
            upside_skew,
            smile_curvature,
            put_call_skew,
            wings_skew
        ]
    
    return result

def analyze_and_save_iv_skew(ticker_filter=None, verbose=False, force=False):
    """
    Process all option data file pairs, calculate IV skew metrics,
    and save results to the option_analysis folder.
    
    Parameters:
    - ticker_filter: Optional ticker to process only specific files
    - verbose: Print detailed information if True
    - force: Re-run analysis even if output files exist
    """
    output_dir = "option_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all option data files
    all_files = glob.glob("option_data/*.pkl")
    
    # Apply ticker filter if specified
    if ticker_filter:
        ticker_filter = ticker_filter.lower()
        all_files = [f for f in all_files if f.lower().startswith(f"option_data/{ticker_filter}_")]
        
        if not all_files:
            print(f"No files found for ticker: {ticker_filter}")
            return
    
    # Print sample filenames to understand the pattern
    if verbose:
        print(f"Found {len(all_files)} files in option_data directory")
        for i, file in enumerate(all_files[:5]):  # Print first 5 files
            print(f"Sample file {i+1}: {os.path.basename(file)}")
    else:
        print(f"Found {len(all_files)} files in option_data directory")
    
    # Create a dictionary to map from call files to their corresponding put files
    processed_pairs = []
    
    # Identify call files and try to find their put counterparts
    call_files = [f for f in all_files if "_call_" in os.path.basename(f)]
    
    # Try to find matching put files for each call file
    for call_file in call_files:
        call_filename = os.path.basename(call_file)
        
        # Extract ticker and date for naming output files
        parts = call_filename.split('_')
        ticker = parts[0]
        date_str = '_'.join(parts[1:3])  # Join date and time parts
        file_key = f"{ticker}_{date_str}"
        
        # Check if output already exists
        output_check = os.path.join(output_dir, f"{file_key}_iv_skew.pkl")
        
        if os.path.exists(output_check) and not force:
            if verbose:
                print(f"Skipping {file_key} - output already exists. Use --force to re-run.")
            continue
        
        # Replace "_call_" with "_put_" to find the corresponding put file
        put_filename = call_filename.replace("_call_", "_put_")
        put_file = os.path.join(os.path.dirname(call_file), put_filename)
        
        # Check if the corresponding put file exists
        if os.path.exists(put_file):
            if verbose:
                print(f"Found matching pair: {call_filename} and {put_filename}")
            
            try:
                # Load data
                call_df = pd.read_pickle(call_file)
                put_df = pd.read_pickle(put_file)
                
                # Calculate IV skew
                skew_df = calculate_iv_skew(call_df, put_df)
                
                # Save results
                output_filename = f"{file_key}_iv_skew.pkl"
                output_path = os.path.join(output_dir, output_filename)
                skew_df.to_pickle(output_path)
                
                # Also save as CSV for easier inspection
                csv_path = os.path.join(output_dir, f"{file_key}_iv_skew.csv")
                skew_df.to_csv(csv_path)
                
                # Count valid data points
                valid_count = skew_df['downside_skew'].notna().sum()
                print(f"Successfully calculated skew for {file_key}: found {valid_count} valid points")
                processed_pairs.append((call_file, put_file))
                
                # Create plot if we have valid data
                if valid_count > 0:
                    try:
                        # Plot time series of skew metrics
                        plt.figure(figsize=(14, 8))
                        
                        plt.plot(skew_df.index, skew_df['downside_skew'], label='Downside Skew', color='red')
                        plt.plot(skew_df.index, skew_df['upside_skew'], label='Upside Skew', color='green')
                        plt.plot(skew_df.index, skew_df['put_call_skew'], label='Put-Call Skew', color='blue')
                        plt.plot(skew_df.index, skew_df['wings_skew'], label='Wings Skew', color='purple')
                        
                        plt.title(f'IV Skew Metrics - {file_key}', fontsize=14)
                        plt.xlabel('Date', fontsize=12)
                        plt.ylabel('Skew Value', fontsize=12)
                        plt.axhline(y=0, color='gray', linestyle='--')
                        plt.xticks(rotation=45)
                        plt.legend()
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        
                        plot_path = os.path.join(output_dir, f"{file_key}_iv_skew_timeseries.png")
                        plt.savefig(plot_path, dpi=300)
                        plt.close()
                    except Exception as e:
                        print(f"Error plotting skew for {file_key}: {str(e)}")
                        
            except Exception as e:
                print(f"Error processing pair {call_filename} and {put_filename}: {str(e)}")
        else:
            if verbose:
                print(f"No matching put file found for {call_filename}")
    
    print(f"IV skew analysis complete. Processed {len(processed_pairs)} pairs. Results saved to {output_dir}/")

def calculate_call_put_ratio_dict():
    """
    Calculate call/put volume ratio dictionary for all available option pairs.
    
    Returns:
    - Dictionary mapping ticker/date keys to call/put volume ratio arrays
    """
    d_vol = {}
    d_call_put_vol = {}
    
    for filepath in glob.glob("option_data/*.pkl"):
        df = pd.read_pickle(filepath)
        ticker_parts = os.path.basename(filepath).split('_')[:4]
        volume_v = calculate_volume_vector(df)
        d_vol[tuple(ticker_parts)] = volume_v


    for key1 in d_vol:
        for key2 in d_vol:
            if (key1[0] == key2[0] and 
                key1[1] == key2[1] and 
                key1[2] == 'call' and 
                key2[2] == 'put'):
                d_call_put_vol[f'{key1[0]}_{key1[1]}_{key1[3]}_cp'] = d_vol[key1] / d_vol[key2]

    return d_call_put_vol

def calculate_and_save_call_put_ratio(ticker_filter=None, verbose=False, force=False):
    """
    Calculate call/put volume ratio for all option data file pairs and
    save results to the option_analysis folder.
    
    Parameters:
    - ticker_filter: Optional ticker to process only specific files
    - verbose: Print detailed information if True
    - force: Re-run analysis even if output files exist
    """
    output_dir = "option_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all option data files
    all_files = glob.glob("option_data/*.pkl")
    
    # Apply ticker filter if specified
    if ticker_filter:
        ticker_filter = ticker_filter.lower()
        all_files = [f for f in all_files if f.lower().startswith(f"option_data/{ticker_filter}_")]
        
        if not all_files:
            print(f"No files found for ticker: {ticker_filter}")
            return
            
    if verbose:
        print(f"Found {len(all_files)} files in option_data directory")
        # Print a few examples to understand the pattern
        for i, file in enumerate(all_files[:3]):
            print(f"Example file {i+1}: {os.path.basename(file)}")
    else:
        print(f"Found {len(all_files)} files in option_data directory")
    
    # Identify call files and find their put counterparts
    call_files = [f for f in all_files if "_call_" in os.path.basename(f)]
    if verbose:
        print(f"Found {len(call_files)} potential call files")
    
    processed_pairs = []
    
    # Process each call file and its corresponding put file
    for call_file in all_files:
        call_filename = os.path.basename(call_file)
        
        # Skip if not a call file
        if "_call_" not in call_filename:
            continue
        
        # Extract ticker and date for output filename
        parts = call_filename.split('_')
        ticker = parts[0]
        date_str = '_'.join(parts[1:3])
        file_key = f"{ticker}_{date_str}"
        
        # Check if output already exists
        output_check = os.path.join(output_dir, f"{file_key}_cp_ratio.pkl")
        
        if os.path.exists(output_check) and not force:
            if verbose:
                print(f"Skipping {file_key} - output already exists. Use --force to re-run.")
            continue
        
        # Generate the corresponding put filename
        put_filename = call_filename.replace("_call_", "_put_")
        put_file = os.path.join("option_data", put_filename)
        
        # Check if the put file exists
        if os.path.exists(put_file):
            if verbose:
                print(f"Found matching pair: {call_filename} and {put_filename}")
            
            try:
                # Load data
                call_df = pd.read_pickle(call_file)
                put_df = pd.read_pickle(put_file)
                
                # Calculate call/put ratio
                cp_ratio_df = calculate_call_put_ratio(call_df, put_df)
                
                # Save results
                output_filename = f"{file_key}_cp_ratio.pkl"
                output_path = os.path.join(output_dir, output_filename)
                cp_ratio_df.to_pickle(output_path)
                
                # Also save as CSV for easier inspection
                csv_path = os.path.join(output_dir, f"{file_key}_cp_ratio.csv")
                cp_ratio_df.to_csv(csv_path)
                
                # Count valid data points
                valid_count = cp_ratio_df['cp_ratio'].notna().sum()
                print(f"Processed {file_key}: found {valid_count} valid call/put ratios")
                processed_pairs.append((call_file, put_file))
                
                # Create plot if we have data
                if valid_count > 0:
                    plot_call_put_ratio(cp_ratio_df, file_key, output_dir)
                    
            except Exception as e:
                print(f"Error processing pair {call_filename} and {put_filename}: {str(e)}")
        else:
            if verbose:
                print(f"No matching put file found for {call_filename}")
    
    print(f"Call/Put ratio analysis complete. Processed {len(processed_pairs)} pairs.")
    print(f"Results saved to {output_dir}/")

def calculate_call_put_ratio(call_df, put_df):
    """
    Calculate the call/put volume ratio for each timestamp.
    
    Parameters:
    - call_df: DataFrame containing call options data
    - put_df: DataFrame containing put options data
    
    Returns:
    - DataFrame with call/put volume ratio and raw volumes
    """
    # Create result DataFrame
    result = pd.DataFrame(
        index=call_df.index,
        columns=['call_volume', 'put_volume', 'cp_ratio', 'cp_ratio_ma5']
    )
    
    # Get option objects
    call_options = call_df.columns.get_level_values(0).unique()
    put_options = put_df.columns.get_level_values(0).unique()
    
    # Process each timestamp
    for idx in call_df.index:
        # Calculate total call volume for this timestamp
        call_volume = 0
        for option in call_options:
            volume = call_df.loc[idx, (option, 'volume')]
            if not pd.isna(volume):
                call_volume += volume
        
        # Calculate total put volume for this timestamp
        put_volume = 0
        for option in put_options:
            volume = put_df.loc[idx, (option, 'volume')]
            if not pd.isna(volume):
                put_volume += volume
        
        # Calculate ratio (avoid division by zero)
        cp_ratio = call_volume / put_volume if put_volume > 0 else np.nan
        
        # Store results
        result.loc[idx, 'call_volume'] = call_volume
        result.loc[idx, 'put_volume'] = put_volume
        result.loc[idx, 'cp_ratio'] = cp_ratio
    
    # Calculate 5-period moving average for smoother analysis
    result['cp_ratio_ma5'] = result['cp_ratio'].rolling(window=5).mean()
    
    return result

def plot_call_put_ratio(cp_ratio_df, key, output_dir):
    """
    Create visualization of call/put ratio over time with volume on right axis.
    
    Parameters:
    - cp_ratio_df: DataFrame with call/put ratio data
    - key: String identifier for the ticker/date
    - output_dir: Directory to save the plot
    """
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Create a second y-axis for volumes
    ax2 = ax1.twinx()
    
    # Plot call/put ratio on left axis
    ratio_line = ax1.plot(cp_ratio_df.index, cp_ratio_df['cp_ratio'], 
                         label='C/P Ratio', color='blue', linewidth=2)
    
    # Plot volume on right axis
    call_bars = ax2.bar(cp_ratio_df.index, cp_ratio_df['call_volume'], 
                       alpha=0.3, color='green', label='Call Volume')
    put_bars = ax2.bar(cp_ratio_df.index, -cp_ratio_df['put_volume'], 
                      alpha=0.3, color='red', label='Put Volume')
    
    # Set labels and title
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('C/P Ratio', fontsize=12)
    ax2.set_ylabel('Volume', fontsize=12)
    plt.title(f'Call/Put Volume Ratio - {key}', fontsize=14)
    
    # Format dates
    plt.xticks(rotation=45)
    fig.autofmt_xdate()
    
    # Combine legends from both axes
    handles = ratio_line + [call_bars, put_bars]
    labels = ['C/P Ratio', 'Call Volume', 'Put Volume']
    ax1.legend(handles, labels, loc='upper left')
    
    # Add grid
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f"{key}_cp_ratio.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

def compute_implied_move(call_df, put_df):
    """
    Compute implied move using ATM straddle method.
    
    Parameters:
    - call_df: DataFrame containing call options data
    - put_df: DataFrame containing put options data
    
    Returns:
    - DataFrame with implied move in dollars and percentage
    """
    # Create result DataFrame using the index from call_df
    result = pd.DataFrame(
        index=call_df.index,
        columns=['implied_move_dollar', 'implied_move_percent', 'atm_strike', 'expiration_date']
    )
    
    # Get option objects
    call_options = call_df.columns.get_level_values(0).unique()
    put_options = put_df.columns.get_level_values(0).unique()
    
    # Group options by strike price and expiration date
    call_by_strike = {}
    for option in call_options:
        strike = option.strike_price
        expiration = option.expiration_date
        if strike not in call_by_strike:
            call_by_strike[strike] = {}
        call_by_strike[strike][expiration] = option
    
    put_by_strike = {}
    for option in put_options:
        strike = option.strike_price
        expiration = option.expiration_date
        if strike not in put_by_strike:
            put_by_strike[strike] = {}
        put_by_strike[strike][expiration] = option
    
    # Process each timestamp
    for idx in tqdm(call_df.index):
        # Get stock price for this timestamp
        stock_price = None
        for option in call_options:
            if not pd.isna(call_df.loc[idx, (option, 'stock_vwap')]):
                stock_price = call_df.loc[idx, (option, 'stock_vwap')]
                break
        
        if stock_price is None or stock_price == 0:
            result.loc[idx] = [np.nan, np.nan, np.nan, np.nan]
            continue
        
        # Find valid straddles (matching strikes and expirations with non-zero volume)
        valid_straddles = []
        
        # Find common strike prices between calls and puts
        common_strikes = set(call_by_strike.keys()).intersection(set(put_by_strike.keys()))
        for strike in common_strikes:
            # Find common expiration dates for this strike
            common_expirations = set(call_by_strike[strike].keys()).intersection(set(put_by_strike[strike].keys()))
            
            for expiration in sorted(common_expirations):  # Sort by expiration date
                call_option = call_by_strike[strike][expiration]
                put_option = put_by_strike[strike][expiration]
                
                # Check volumes
                call_volume = call_df.loc[idx, (call_option, 'volume')]
                put_volume = put_df.loc[idx, (put_option, 'volume')]
                
                if call_volume > 0 and put_volume > 0:
                    call_price = call_df.loc[idx, (call_option, 'vwap')]
                    put_price = put_df.loc[idx, (put_option, 'vwap')]
                    
                    # Skip if prices are invalid
                    if pd.isna(call_price) or pd.isna(put_price) or call_price == 0 or put_price == 0:
                        continue
                    
                    # Calculate strike distance from current price
                    strike_diff = abs(strike - stock_price)
                    
                    valid_straddles.append({
                        'strike_diff': strike_diff,
                        'strike': strike,
                        'expiration': expiration,
                        'call_price': call_price,
                        'put_price': put_price
                    })
        
        if not valid_straddles:
            result.loc[idx] = [np.nan, np.nan, np.nan, np.nan]
            continue
        
        # Find the ATM straddle (closest strike to current price)
        valid_straddles.sort(key=lambda x: x['strike_diff'])
        atm_straddle = valid_straddles[0]
        
        # Calculate implied move
        implied_move_dollar = atm_straddle['call_price'] + atm_straddle['put_price']
        implied_move_percent = (implied_move_dollar / stock_price) * 100
        
        result.loc[idx] = [
            implied_move_dollar,
            implied_move_percent,
            atm_straddle['strike'],
            atm_straddle['expiration']
        ]
    
    return result

def save_iv_matrix(ticker_filter=None, verbose=False, force=False):
    """
    Calculate and save IV matrix for all option data files.
    Also calculates and saves the call/put volume ratio dictionary.
    
    Parameters:
    - ticker_filter: Optional ticker to process only specific files
    - verbose: Print detailed information if True
    - force: Re-run analysis even if output files exist
    
    The IV matrix contains implied volatility values for each option strike at each timestamp.
    """
    output_dir = "option_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save call/put volume ratio dictionary (unless filtered)
    if not ticker_filter:
        d_call_put_vol = calculate_call_put_ratio_dict()
        cp_vol_path = os.path.join(output_dir, 'd_call_put_vol.pkl')
        
        if not os.path.exists(cp_vol_path) or force:
            with open(cp_vol_path, 'wb') as file:
                pickle.dump(d_call_put_vol, file)
            print(f"Saved call/put volume dictionary to {cp_vol_path}")
        else:
            print(f"Skipping call/put volume dictionary - file already exists. Use --force to re-run.")
    
    # Find all option data files
    all_files = glob.glob("option_data/*.pkl")
    
    # Apply ticker filter if specified
    if ticker_filter:
        ticker_filter = ticker_filter.lower()
        all_files = [f for f in all_files if f.lower().startswith(f"option_data/{ticker_filter}_")]
        
        if not all_files:
            print(f"No files found for ticker: {ticker_filter}")
            return
    
    print(f"Processing {len(all_files)} files for IV matrix calculation")

    for filepath in all_files:
        base_filename = os.path.basename(filepath)
        new_filename = f'{os.path.splitext(base_filename)[0]}_ivmatrix.pkl'
        output_path = os.path.join(output_dir, new_filename)
        
        # Check if output already exists
        if os.path.exists(output_path) and not force:
            if verbose:
                print(f"Skipping {base_filename} - output already exists. Use --force to re-run.")
            continue
            
        try:
            df = pd.read_pickle(filepath)
            iv_matrix_df = calculate_iv_matrix(df)
            
            if verbose:
                print(f"Calculated IV matrix for {base_filename}")
                print(iv_matrix_df)
            
            iv_matrix_df.to_pickle(output_path)
            print(f"Saved IV matrix to {output_path}")
        except Exception as e:
            print(f"Error processing {base_filename}: {str(e)}")

def save_implied_move(ticker_filter=None, verbose=False, force=False):
    """
    Calculate and save implied move data for all option pairs.
    
    Parameters:
    - ticker_filter: Optional ticker to process only specific files
    - verbose: Print detailed information if True
    - force: Re-run analysis even if output files exist
    
    Implied move is calculated using ATM straddles (closest strike to current stock price)
    and represents the expected price move in the underlying stock based on options pricing.
    """
    output_dir = "option_analysis"
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all option data files
    all_files = glob.glob("option_data/*.pkl")
    
    # Apply ticker filter if specified
    if ticker_filter:
        ticker_filter = ticker_filter.lower()
        all_files = [f for f in all_files if f.lower().startswith(f"option_data/{ticker_filter}_")]
        
        if not all_files:
            print(f"No files found for ticker: {ticker_filter}")
            return
    
    if verbose:
        print(f"Found {len(all_files)} files in option_data directory:")
        for file in all_files:
            print(f"  - {os.path.basename(file)}")
    else:
        print(f"Found {len(all_files)} files in option_data directory")
    
    # Group files by their base name (without call/put and version number)
    file_pairs = {}
    
    for filepath in all_files:
        filename = os.path.basename(filepath)
        
        # Split filename by underscores
        parts = filename.split('_')
        
        # Check if 'call' or 'put' is in parts
        option_type = None
        if 'call' in parts:
            option_type = 'call'
            idx = parts.index('call')
        elif 'put' in parts:
            option_type = 'put'
            idx = parts.index('put')
        else:
            if verbose:
                print(f"Skipping {filename} - can't identify as call or put")
            continue
            
        # Reconstruct base key by joining parts before call/put
        base_key = '_'.join(parts[:idx])
        
        if verbose:
            print(f"Parsed {filename} -> base_key: {base_key}, type: {option_type}")
        
        if base_key not in file_pairs:
            file_pairs[base_key] = {}
        
        file_pairs[base_key][option_type] = filepath
    
    # Print what pairs were found
    if verbose:
        print("\nIdentified the following potential pairs:")
        for key, files in file_pairs.items():
            print(f"{key}: {list(files.keys())}")
    
    # Process each complete pair (has both call and put)
    processed_count = 0
    for key, files in file_pairs.items():
        # Check if output already exists
        output_path = os.path.join(output_dir, f"{key}_implied_move.pkl")
        
        if os.path.exists(output_path) and not force:
            if verbose:
                print(f"Skipping {key} - output already exists. Use --force to re-run.")
            continue
            
        if 'call' in files and 'put' in files:
            if verbose:
                print(f"\nProcessing complete pair for {key}")
                print(f"  Call file: {os.path.basename(files['call'])}")
                print(f"  Put file: {os.path.basename(files['put'])}")
            else:
                print(f"Processing {key}")
            
            try:
                # Load data
                call_df = pd.read_pickle(files['call'])
                put_df = pd.read_pickle(files['put'])
                
                # Compute implied move
                im_df = compute_implied_move(call_df, put_df)
                
                # Save result
                output_filename = f"{key}_implied_move.pkl"
                output_path = os.path.join(output_dir, output_filename)
                im_df.to_pickle(output_path)
                
                # Also save as CSV
                csv_path = os.path.join(output_dir, f"{key}_implied_move.csv")
                im_df.to_csv(csv_path)
                
                valid_moves = im_df['implied_move_percent'].notna().sum()
                print(f"  Success! Found {valid_moves} valid implied moves")
                processed_count += 1
                
                # Create plot if we have data
                if valid_moves > 0:
                    plt.figure(figsize=(12, 6))
                    plt.plot(im_df.index, im_df['implied_move_percent'])
                    plt.title(f'Implied Move % - {key}', fontsize=14)
                    plt.xlabel('Date', fontsize=12)
                    plt.ylabel('Implied Move %', fontsize=12)
                    plt.xticks(rotation=45)
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    
                    plot_path = os.path.join(output_dir, f"{key}_implied_move.png")
                    plt.savefig(plot_path, dpi=300)
                    plt.close()
            
            except Exception as e:
                print(f"  Error processing {key}: {str(e)}")
        else:
            if verbose:
                print(f"\nIncomplete pair for {key} - missing: {set(['call', 'put']) - set(files.keys())}")
    
    print(f"\nImplied move analysis complete. Processed {processed_count} ticker/date pairs.")
    print(f"Results saved to {output_dir}/")


def plot_iv_skew_smiles(ticker_filter=None, verbose=False, force=False):
    """
    Plots smile curvature from IV skew data stored in pickle files.
    
    Parameters:
    - ticker_filter: Optional ticker to process only specific files
    - verbose: Print detailed information if True
    - force: Re-run analysis even if output files exist
    
    This function:
    1. Searches the 'option_analysis' folder for files ending with '_iv_skew.pkl'
    2. For each matching file, loads the data and plots the smile_curvature
    3. Saves the plot in the same folder with an appropriate name
    """
    # Path to the option_analysis folder
    folder_path = 'option_analysis'
    
    # Make sure the folder exists
    if not os.path.exists(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return
    
    # Get list of IV skew files
    iv_skew_files = [f for f in os.listdir(folder_path) if f.endswith('_iv_skew.pkl')]
    
    # Apply ticker filter if specified
    if ticker_filter:
        ticker_filter = ticker_filter.lower()
        iv_skew_files = [f for f in iv_skew_files if f.lower().startswith(f"{ticker_filter}_")]
        
        if not iv_skew_files:
            print(f"No IV skew files found for ticker: {ticker_filter}")
            return
    
    print(f"Found {len(iv_skew_files)} IV skew files to plot")
    
    # Loop through all matching files
    for filename in iv_skew_files:
        # Parse the filename
        name_parts = filename.split('_')
        
        # The format is expected to be: ticker_datetime_optiontype_iv_skew.pkl
        if len(name_parts) >= 4:
            ticker = name_parts[0]
            option_type = name_parts[-3]  # The part before 'iv_skew.pkl'
            
            # Full path to the file
            file_path = os.path.join(folder_path, filename)
            
            # Output path
            plot_filename = f"{ticker}_{option_type}_smile_curve.png"
            plot_path = os.path.join(folder_path, plot_filename)
            
            # Check if output already exists
            if os.path.exists(plot_path) and not force:
                if verbose:
                    print(f"Skipping {filename} - plot already exists. Use --force to re-run.")
                continue
                
            try:
                # Load the data from pickle file
                df = pd.read_pickle(file_path)
                
                # Check if smile_curvature column exists
                if 'smile_curvature' in df.columns:
                    # Create the plot
                    plt.figure(figsize=(12, 6))
                    
                    # Convert index to datetime if it's not already
                    if not pd.api.types.is_datetime64_any_dtype(df.index):
                        df.index = pd.to_datetime(df.index)
                    
                    # Plot smile_curvature
                    plt.plot(df.index, df['smile_curvature'], marker='o', linestyle='-', linewidth=2)
                    
                    # Set title and labels
                    plt.title(f'IV Smile Curvature for {ticker} ({option_type})')
                    plt.xlabel('Date')
                    plt.ylabel('Smile Curvature')
                    
                    # Format x-axis to show dates nicely
                    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
                    
                    # Rotate date labels for better readability
                    plt.xticks(rotation=45)
                    
                    # Add grid
                    plt.grid(True, alpha=0.3)
                    
                    # Tight layout for better appearance
                    plt.tight_layout()
                    
                    # Save the plot with an appropriate name
                    plt.savefig(plot_path)
                    plt.close()
                    
                    print(f"Plot saved: {plot_path}")
                else:
                    print(f"Warning: 'smile_curvature' column not found in {filename}")
            
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
        else:
            print(f"Warning: Could not parse filename format for {filename}")

def run_all_analyses(ticker_filter=None, verbose=False, force=False):
    """
    Run all available analyses in sequence.
    
    Parameters:
    - ticker_filter: Optional ticker to process only specific files
    - verbose: Print detailed information if True
    - force: Re-run analysis even if output files exist
    """
    print("\n========= RUNNING ALL ANALYSES =========\n")
    
    print("\n1. Calculating Implied Volatility...")
    analyze_and_save_all_implied_volatility(ticker_filter, verbose, force)
    
    print("\n2. Calculating IV Skew Metrics...")
    analyze_and_save_iv_skew(ticker_filter, verbose, force)
    
    print("\n3. Calculating Call/Put Ratio...")
    calculate_and_save_call_put_ratio(ticker_filter, verbose, force)
    
    print("\n4. Calculating Implied Move...")
    save_implied_move(ticker_filter, verbose, force)
    
    print("\n5. Calculating IV Matrix...")
    save_iv_matrix(ticker_filter, verbose, force)
    
    print("\n6. Plotting IV Skew Smiles...")
    plot_iv_skew_smiles(ticker_filter, verbose, force)
    
    print("\n========= ALL ANALYSES COMPLETE =========")


def main():
    """
    Main function that handles command-line arguments and runs the appropriate analysis.
    """
    # Create argument parser
    parser = argparse.ArgumentParser(description="Biotech Options Analysis Toolkit")
    
    # Add function argument group
    function_group = parser.add_argument_group("Analysis Functions")
    function_group.add_argument("--analyze-iv", action="store_true", help="Calculate and save implied volatility")
    function_group.add_argument("--analyze-skew", action="store_true", help="Calculate IV skew metrics")
    function_group.add_argument("--cp-ratio", action="store_true", help="Calculate call/put volume ratio")
    function_group.add_argument("--implied-move", action="store_true", help="Calculate expected price moves")
    function_group.add_argument("--iv-matrix", action="store_true", help="Calculate IV matrix by strike")
    function_group.add_argument("--plot-smiles", action="store_true", help="Plot volatility smile curves")
    function_group.add_argument("--all", action="store_true", help="Run all analyses")
    
    # Add options
    parser.add_argument("--ticker", type=str, help="Process only the specified ticker")
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress information")
    parser.add_argument("--force", action="store_true", help="Re-run analysis even if output files exist")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if any function was specified, default to --all if not
    functions_specified = (args.analyze_iv or args.analyze_skew or args.cp_ratio or 
                          args.implied_move or args.iv_matrix or args.plot_smiles or args.all)
    
    if not functions_specified:
        print("No analysis function specified, defaulting to --all")
        args.all = True
    
    # Run the selected functions
    if args.all:
        run_all_analyses(args.ticker, args.verbose, args.force)
    else:
        if args.analyze_iv:
            print("\nCalculating Implied Volatility...")
            analyze_and_save_all_implied_volatility(args.ticker, args.verbose, args.force)
            
        if args.analyze_skew:
            print("\nCalculating IV Skew Metrics...")
            analyze_and_save_iv_skew(args.ticker, args.verbose, args.force)
            
        if args.cp_ratio:
            print("\nCalculating Call/Put Ratio...")
            calculate_and_save_call_put_ratio(args.ticker, args.verbose, args.force)
            
        if args.implied_move:
            print("\nCalculating Implied Move...")
            save_implied_move(args.ticker, args.verbose, args.force)
            
        if args.iv_matrix:
            print("\nCalculating IV Matrix...")
            save_iv_matrix(args.ticker, args.verbose, args.force)
            
        if args.plot_smiles:
            print("\nPlotting IV Skew Smiles...")
            plot_iv_skew_smiles(args.ticker, args.verbose, args.force)
    
    print("\nAnalysis complete!")

if __name__ == '__main__':
    main()