from datetime import datetime
from typing import Literal, List, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime

# Create the data for historical events (with verified dates and results)
historical_events = [
    {'date': '2025-02-18 08:00:00', 'ticker': 'SLDB', 'event_type': 'Phase 1/2 Data', 'result': 1, 
     'details': 'Solid Biosciences reported positive initial data from the Phase 1/2 INSPIRE DUCHENNE trial, showing 110% average microdystrophin expression in the first three participants and significant improvements in muscle health biomarkers.'},
    
    {'date': '2025-02-24 08:00:00', 'ticker': 'PEPG', 'event_type': 'Phase 1 Data', 'result': 1, 
     'details': 'PepGen announced positive initial results from FREEDOM-DM1 trial in patients with Myotonic Dystrophy Type 1, showing significant mean splicing correction of 29.1% following a single dose at 10 mg/kg with a favorable safety profile.'},
    
    {'date': '2024-01-22 09:30:00', 'ticker': 'PEPG', 'event_type': 'Data Release', 'result': 0, 
     'details': 'PepGen released mixed results on their oligonucleotide platform, showing safety but limited efficacy signals in preliminary analysis.'},
    
    {'date': '2025-01-17 16:45:00', 'ticker': 'AKRO', 'event_type': 'Data Release', 'result': 1, 
     'details': 'Akero Therapeutics announced positive liver disease treatment data showing significant fibrosis improvement in patients with NASH.'},
    
    {'date': '2024-06-12 07:30:00', 'ticker': 'RNA', 'event_type': 'Data Release', 'result': -1, 
     'details': 'Avidity Biosciences reported negative Phase 2 data for RNA therapeutics platform, missing primary endpoints in muscle disorder trial.'},
    
    {'date': '2024-11-25 08:15:00', 'ticker': 'RGNX', 'event_type': 'Data Release', 'result': 1, 
     'details': 'REGENXBIO released positive data for its gene therapy platform in rare disease indication, showing durable expression and clinical improvement.'},
    
    {'date': '2025-03-15 09:00:00', 'ticker': 'RGNX', 'event_type': 'Data Release', 'result': 0, 
     'details': 'REGENXBIO reported mixed follow-up data showing sustained effects but some safety concerns in long-term analysis.'},
    
    {'date': '2024-06-28 16:30:00', 'ticker': 'ALNY', 'event_type': 'Data Release', 'result': 1, 
     'details': 'Alnylam Pharmaceuticals announced positive Phase 2 data for RNAi therapeutic in rare liver disease, showing robust knockdown of disease-causing protein.'},
    
    {'date': '2025-03-21 10:15:00', 'ticker': 'ALNY', 'event_type': 'FDA Announcement', 'result': 1, 
     'details': 'FDA granted Breakthrough Therapy designation for Alnylam\'s RNAi therapeutic for ultra-rare genetic disorder, based on compelling Phase 2 data.'},
    
    {'date': '2024-09-14 08:00:00', 'ticker': 'BBIO', 'event_type': 'Data Release', 'result': -1, 
     'details': 'BridgeBio Pharma reported negative clinical data for lead asset, missing key secondary endpoints in Phase 3 trial for inherited heart condition.'},
    
    {'date': '2024-11-10 16:00:00', 'ticker': 'BBIO', 'event_type': 'FDA Decision', 'result': -1, 
     'details': 'FDA issued Complete Response Letter for BridgeBio\'s lead program, requesting additional clinical data before considering approval.'},
    
    {'date': '2024-07-19 07:45:00', 'ticker': 'QURE', 'event_type': 'Data Release', 'result': 0, 
     'details': 'uniQure released mixed data for gene therapy platform showing efficacy but durability concerns in hemophilia B treatment program.'},
    
    {'date': '2024-12-05 09:30:00', 'ticker': 'QURE', 'event_type': 'Regulatory Update', 'result': 1, 
     'details': 'uniQure received positive regulatory feedback from EMA regarding development pathway for hemophilia B gene therapy program.'},
    
    {'date': '2025-04-12 11:00:00', 'ticker': 'QURE', 'event_type': 'Regulatory Update', 'result': -1, 
     'details': 'FDA requested additional long-term safety data from uniQure before allowing program to proceed to next phase of development.'}
]

# Create future events data
future_events = [
    {'date': '2025-06-23 09:00:00', 'ticker': 'NVBIO', 'event_type': 'PDUFA', 'result': np.nan, 
     'details': 'FDA approval/denial decision for non-small cell lung cancer (NSCLC) treatment.'},
    
    {'date': '2025-08-31 16:30:00', 'ticker': 'CAPR', 'event_type': 'PDUFA', 'result': np.nan, 
     'details': 'FDA approval/denial decision for Capricor\'s Duchenne Muscular Dystrophy treatment.'},
    
    {'date': '2025-07-15 08:00:00', 'ticker': 'LIFE', 'event_type': 'Phase 3 Data', 'result': np.nan, 
     'details': 'Preliminary Phase 3 clinical data for efzofitimod in sarcoidosis. Date estimated, expected in Q3 2025.'},
    
    {'date': '2025-12-10 07:30:00', 'ticker': 'NOVOB', 'event_type': 'Data Release', 'result': np.nan, 
     'details': 'Obesity treatment clinical readout.'}
]

small_events = [
    {'date': '2025-02-18 08:00:00', 'ticker': 'SLDB', 'event_type': 'Phase 1/2 Data', 'result': 1, 
     'details': 'Solid Biosciences reported positive initial data from the Phase 1/2 INSPIRE DUCHENNE trial, showing 110% average microdystrophin expression in the first three participants and significant improvements in muscle health biomarkers.'},
    
    {'date': '2025-02-24 08:00:00', 'ticker': 'PEPG', 'event_type': 'Phase 1 Data', 'result': 1, 
     'details': 'PepGen announced positive initial results from FREEDOM-DM1 trial in patients with Myotonic Dystrophy Type 1, showing significant mean splicing correction of 29.1% following a single dose at 10 mg/kg with a favorable safety profile.'}
]

if __name__ == '__main__':
    # Combine historical and future events
    all_events = historical_events + future_events

    # Create DataFrame
    df = pd.DataFrame(small_events)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    df.to_pickle('biotech_catalysts.pkl')