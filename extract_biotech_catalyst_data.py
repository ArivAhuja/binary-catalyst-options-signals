from datetime import datetime
from typing import Literal, List, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime

selected_events = [
    {'date': '2025-03-20 10:15:00', 'ticker': 'ALNY', 'event_type': 'FDA Approval', 'result': 1, 
     'details': 'FDA approved AMVUTTRA (vutrisiran) as the first RNAi therapeutic to reduce cardiovascular death, hospitalizations, and urgent heart failure visits in adults with ATTR amyloidosis with cardiomyopathy.'},
    
    {'date': '2024-11-22 16:00:00', 'ticker': 'BBIO', 'event_type': 'FDA Approval', 'result': 1, 
     'details': 'FDA approved Attruby (acoramidis), a near-complete TTR stabilizer, to reduce cardiovascular death and cardiovascular-related hospitalization in adult patients with ATTR-CM.'}
]

if __name__ == '__main__':
    df = pd.DataFrame(selected_events)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    df.to_pickle('biotech_catalysts/biotech_catalysts_data.pkl')