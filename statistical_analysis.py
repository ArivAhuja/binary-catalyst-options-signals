"""
Direct Biotech File Analysis

This script takes a direct approach to finding and analyzing biotech options files.

"""

import os
import pandas as pd
import numpy as np
import glob

def main():
    """Main function with direct file analysis"""
    print("\n===== DIRECT BIOTECH FILE ANALYSIS =====\n")
    
    # STEP 1: Find the cp_ratio files directly
    print("Searching for cp_ratio files:")
    cp_ratio_files = glob.glob('option_analysis/*cp_ratio.pkl')
    
    if not cp_ratio_files:
        print("ERROR: No cp_ratio files found in option_analysis directory!")
        print("Make sure the directory and files exist.")
        return
    
    print(f"Found {len(cp_ratio_files)} cp_ratio files:")
    for f in cp_ratio_files:
        print(f"  {os.path.basename(f)}")
    
    # STEP 2: Load biotech catalysts data
    print("\nLoading biotech catalysts data...")
    try:
        catalysts_df = pd.read_pickle('biotech_catalysts/biotech_catalysts_data.pkl')
        print(f"Loaded {len(catalysts_df)} catalyst events")
        
        # Print catalyst events with their result values
        print("\nCatalyst events with results:")
        for idx, row in catalysts_df.iterrows():
            date_str = idx.strftime('%Y-%m-%d %H:%M:%S')
            print(f"  {row['ticker']} - {date_str} - Result: {row['result']}")
            
        # Count outcomes
        positive_count = (catalysts_df['result'] == 1).sum()
        negative_count = (catalysts_df['result'] == -1).sum()
        pending_count = (catalysts_df['result'] == 0).sum()
        print(f"\nEvent outcomes: {positive_count} positive, {negative_count} negative, {pending_count} pending")
        
    except Exception as e:
        print(f"Error loading catalysts data: {e}")
        return
    
    # STEP 3: Try to match catalysts with cp_ratio files directly
    print("\nAttempting to match catalyst events with cp_ratio files:")
    
    matched_files = []
    historical_files = []
    
    for idx, row in catalysts_df.iterrows():
        ticker = row['ticker'].lower()
        date_str = idx.strftime('%Y-%m-%d %H:%M:%S')
        
        # Try to find any matching file for this ticker/date
        found_matches = []
        for file_path in cp_ratio_files:
            file_name = os.path.basename(file_path)
            if ticker in file_name.lower() and date_str in file_name:
                found_matches.append(file_path)
        
        if found_matches:
            for match in found_matches:
                print(f"  MATCH: {row['ticker']} ({date_str}) -> {os.path.basename(match)}")
                matched_files.append((match, row['result']))
                
                # If this is a historical event (result is 1 or -1), add to historical files
                if row['result'] in [1, -1]:
                    historical_files.append((match, row['result']))
        else:
            print(f"  NO MATCH: {row['ticker']} ({date_str})")
    
    print(f"\nMatched {len(matched_files)} files total, {len(historical_files)} historical events")
    
    # STEP 4: Analyze the cp_ratio data for historical events
    if historical_files:
        print("\n===== ANALYZING CP RATIO DATA =====\n")
        
        positive_events = []
        negative_events = []
        
        for file_path, result in historical_files:
            try:
                # Extract ticker from filename
                file_name = os.path.basename(file_path)
                ticker = file_name.split('_')[0].upper()
                
                print(f"Loading data for {ticker} from {file_name}")
                cp_ratio_df = pd.read_pickle(file_path)
                
                if cp_ratio_df.empty:
                    print(f"  Empty DataFrame for {ticker}")
                    continue
                
                print(f"  DataFrame loaded: {cp_ratio_df.shape} with columns {cp_ratio_df.columns.tolist()}")
                
                # Print first few rows to verify data format
                print("  First 2 rows of data:")
                print(cp_ratio_df.head(2))
                
                # Calculate metrics
                avg_cp_ratio = cp_ratio_df['cp_ratio'].mean()
                final_cp_ratio = cp_ratio_df['cp_ratio'].iloc[-1]
                
                event_data = {
                    'ticker': ticker,
                    'avg_cp_ratio': avg_cp_ratio,
                    'final_cp_ratio': final_cp_ratio
                }
                
                # Add to appropriate outcome list
                if result == 1:
                    positive_events.append(event_data)
                    print(f"  Added to POSITIVE outcomes: Avg CP = {avg_cp_ratio:.4f}, Final CP = {final_cp_ratio:.4f}")
                else:
                    negative_events.append(event_data)
                    print(f"  Added to NEGATIVE outcomes: Avg CP = {avg_cp_ratio:.4f}, Final CP = {final_cp_ratio:.4f}")
                
            except Exception as e:
                print(f"  Error processing {file_path}: {e}")
        
        # Output summary
        print("\n===== CP RATIO ANALYSIS RESULTS =====\n")
        
        if positive_events:
            pos_avg = sum(item['avg_cp_ratio'] for item in positive_events) / len(positive_events)
            pos_final = sum(item['final_cp_ratio'] for item in positive_events) / len(positive_events)
            
            print(f"POSITIVE OUTCOMES (n={len(positive_events)}):")
            print(f"  Average CP ratio: {pos_avg:.4f}")
            print(f"  Average final CP ratio: {pos_final:.4f}")
            print("  Individual events:")
            for item in positive_events:
                print(f"    {item['ticker']}: Avg CP = {item['avg_cp_ratio']:.4f}, Final CP = {item['final_cp_ratio']:.4f}")
        else:
            print("No CP ratio data for positive outcomes")
        
        if negative_events:
            neg_avg = sum(item['avg_cp_ratio'] for item in negative_events) / len(negative_events)
            neg_final = sum(item['final_cp_ratio'] for item in negative_events) / len(negative_events)
            
            print(f"\nNEGATIVE OUTCOMES (n={len(negative_events)}):")
            print(f"  Average CP ratio: {neg_avg:.4f}")
            print(f"  Average final CP ratio: {neg_final:.4f}")
            print("  Individual events:")
            for item in negative_events:
                print(f"    {item['ticker']}: Avg CP = {item['avg_cp_ratio']:.4f}, Final CP = {item['final_cp_ratio']:.4f}")
        else:
            print("\nNo CP ratio data for negative outcomes")
            
        # Compare positive vs negative if both have data
        if positive_events and negative_events:
            pos_avg = sum(item['avg_cp_ratio'] for item in positive_events) / len(positive_events)
            neg_avg = sum(item['avg_cp_ratio'] for item in negative_events) / len(negative_events)
            diff = pos_avg - neg_avg
            
            print(f"\nDIFFERENCE (Positive - Negative):")
            print(f"  Average CP ratio: {diff:.4f}")
            
            if pos_avg > neg_avg:
                print(f"  Positive outcomes had HIGHER CP ratios on average")
            else:
                print(f"  Positive outcomes had LOWER CP ratios on average")
    else:
        print("\nNo historical files to analyze.")

    # Also save results to a file
    with open('results/direct_analysis.txt', 'w') as f:
        if positive_events:
            pos_avg = sum(item['avg_cp_ratio'] for item in positive_events) / len(positive_events)
            pos_final = sum(item['final_cp_ratio'] for item in positive_events) / len(positive_events)
            
            f.write(f"POSITIVE OUTCOMES (n={len(positive_events)}):\n")
            f.write(f"  Average CP ratio: {pos_avg:.4f}\n")
            f.write(f"  Average final CP ratio: {pos_final:.4f}\n")
            f.write("  Individual events:\n")
            for item in positive_events:
                f.write(f"    {item['ticker']}: Avg CP = {item['avg_cp_ratio']:.4f}, Final CP = {item['final_cp_ratio']:.4f}\n")
        else:
            f.write("No CP ratio data for positive outcomes\n")
        
        if negative_events:
            neg_avg = sum(item['avg_cp_ratio'] for item in negative_events) / len(negative_events)
            neg_final = sum(item['final_cp_ratio'] for item in negative_events) / len(negative_events)
            
            f.write(f"\nNEGATIVE OUTCOMES (n={len(negative_events)}):\n")
            f.write(f"  Average CP ratio: {neg_avg:.4f}\n")
            f.write(f"  Average final CP ratio: {neg_final:.4f}\n")
            f.write("  Individual events:\n")
            for item in negative_events:
                f.write(f"    {item['ticker']}: Avg CP = {item['avg_cp_ratio']:.4f}, Final CP = {item['final_cp_ratio']:.4f}\n")
        else:
            f.write("\nNo CP ratio data for negative outcomes\n")
            
        if positive_events and negative_events:
            pos_avg = sum(item['avg_cp_ratio'] for item in positive_events) / len(positive_events)
            neg_avg = sum(item['avg_cp_ratio'] for item in negative_events) / len(negative_events)
            diff = pos_avg - neg_avg
            
            f.write(f"\nDIFFERENCE (Positive - Negative):\n")
            f.write(f"  Average CP ratio: {diff:.4f}\n")
            
            if pos_avg > neg_avg:
                f.write(f"  Positive outcomes had HIGHER CP ratios on average\n")
            else:
                f.write(f"  Positive outcomes had LOWER CP ratios on average\n")

if __name__ == "__main__":
    main()