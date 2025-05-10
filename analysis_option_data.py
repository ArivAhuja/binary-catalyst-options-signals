import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import glob
import os
from pathlib import Path

def create_implied_volatility_vector(df) -> list:
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

    return iv_v

if __name__ == '__main__': 
    output_dir = "option_analysis"
    for filepath in glob.glob("option_data/*.pkl"):
        df = pd.read_pickle(filepath)
        iv_vector = create_implied_volatility_vector(df)
        base_filename = os.path.basename(filepath)
        output_filename = base_filename.replace('_option_data.pkl', '_iv_data.npy')
        output_path = os.path.join(output_dir, output_filename)
        np.save(output_path, iv_vector)
        print(f"Saved IV vector to {output_path}")

