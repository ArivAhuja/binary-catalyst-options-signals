"""
Biotech Catalyst Events Dataset Creator

This script creates and saves a structured dataset of biotech catalyst events, 
including both historical events with known outcomes and future events with 
pending results. The dataset includes FDA decisions, clinical trial data,
and other significant biotech company events that can impact stock prices.

The data is organized in a pandas DataFrame and saved as a pickle file for 
easy loading in subsequent analysis.

Usage:
    python create_biotech_dataset.py

Output:
    biotech_catalysts/biotech_catalysts_data.pkl - Pickle file containing the DataFrame
"""

import pandas as pd
import os

# Historical biotech catalyst events with known outcomes
# result: 1 = positive outcome, -1 = negative outcome
historical_events = [
    {'date': '2025-02-18 13:00:00', 'ticker': 'SLDB', 'event_type': 'Clinical Data', 'result': 1, 
     'details': 'Positive Phase 1/2 INSPIRE DUCHENNE trial data showed average microdystrophin expression of 110%, and significant improvements in muscle health biomarkers. SGT-003 gene therapy was well-tolerated with no serious adverse events in six participants and early signals of cardiac benefit were observed.'},
    
    {'date': '2025-02-24 13:00:00', 'ticker': 'PEPG', 'event_type': 'Clinical Data', 'result': 1, 
     'details': 'PepGen announced positive initial results from the FREEDOM-DM1 Phase 1 trial evaluating PGN-EDODM1 in myotonic dystrophy type 1 (DM1). Data from the 5 and 10 mg/kg dose cohorts showed significant mean splicing correction of 29.1%, following a single dose at 10 mg/kg, with PGN-EDODM1 demonstrating a favorable emerging safety profile. The company held a conference call to review the data at 8:00 a.m. ET.'},

    {'date': '2024-06-24 12:00:00', 'ticker': 'ALNY', 'event_type': 'Clinical Data', 'result': 1, 
     'details': 'Alnylam announced positive topline results from its HELIOS-B Phase 3 study of vutrisiran in ATTR amyloidosis with cardiomyopathy (ATTR-CM). The study met its primary endpoint, demonstrating a statistically significant reduction in the composite of all-cause mortality and recurrent cardiovascular events in both the overall population and monotherapy population. Vutrisiran also showed statistically significant improvements across all secondary endpoints.'},
    
    {'date': '2024-11-22 22:00:00', 'ticker': 'BBIO', 'event_type': 'FDA', 'result': 1, 
     'details': 'FDA approval of acoramidis for the treatment of ATTR-CM (transthyretin amyloid cardiomyopathy). The FDA had set this PDUFA action date after accepting BridgeBios New Drug Application (NDA) based on positive results from the ATTRibute-CM Phase 3 trial. Upon approval, BridgeBio anticipated receiving a $500 million milestone payment.'},
    
    {'date': '2025-04-17 12:00:00', 'ticker': 'QURE', 'event_type': 'FDA', 'result': 1, 
     'details': 'FDA granted Breakthrough Therapy designation for uniQure\'s Huntington\'s disease treatment in Phase 1/2 development.'},
    
    {'date': '2025-04-03 12:00:00', 'ticker': 'ALDX', 'event_type': 'FDA', 'result': -1,
    'details': 'FDA issued Complete Response Letter (CRL) for Aldeyra\'s dry eye disease treatment. Original PDUFA date was March 4, 2025.'},

    {'date': '2025-03-28 12:00:00', 'ticker': 'MIST', 'event_type': 'FDA', 'result': -1,
    'details': 'FDA issued Complete Response Letter (CRL) for Milestone Pharmaceuticals\' treatment for Paroxysmal supraventricular tachycardia (PSVT).'}
]

# Future biotech catalyst events with pending outcomes
# result: 0 = pending outcome
future_events = [
    {'date': '2025-06-23 12:00:00', 'ticker': 'NUVB', 'event_type': 'FDA', 'result': 0, 
     'details': 'PDUFA date for Nuvation Bio\'s NUV-422 for the treatment of Non-Small Cell Lung Cancer (NSCLC).'},
     
    {'date': '2025-05-26 12:00:00', 'ticker': 'MRK', 'event_type': 'FDA', 'result': 0,
     'details': 'PDUFA Priority Review date for Merck\'s WELIREG (belzutifan).'},
     
    {'date': '2025-05-24 12:00:00', 'ticker': 'LQDA', 'event_type': 'FDA', 'result': 0,
     'details': 'PDUFA date for Liquidia\'s YUTREPIA (treprostinil). CRL previously issued on August 16, 2024; FDA resubmission accepted.'},
     
    {'date': '2025-05-31 12:00:00', 'ticker': 'MRNA', 'event_type': 'FDA', 'result': 0,
     'details': 'PDUFA date for Moderna\'s mRNA-1283 COVID vaccine. Phase 3 met its primary vaccine efficacy endpoint demonstrating non-inferior vaccine efficacy compared to Spikevax.'}
]

def main():
    """
    Process and save biotech catalyst event data.
    
    This function combines historical and future events, converts them to a
    DataFrame, processes date information, and saves the result as a pickle file.
    """
    # Combine both historical and future events into a single list
    all_events = historical_events + future_events

    # Create DataFrame from events data
    df = pd.DataFrame(all_events)
    
    # Convert string dates to datetime objects with UTC timezone
    df['date'] = pd.to_datetime(df['date'], utc=True)
    
    # Set date column as index and sort chronologically
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    # Ensure the directory exists before saving
    os.makedirs('biotech_catalysts', exist_ok=True)
    
    # Save DataFrame as pickle file for later use
    df.to_pickle('biotech_catalysts/biotech_catalysts_data.pkl')
    print(f"Dataset created and saved to 'biotech_catalysts/biotech_catalysts_data.pkl'")
    print(f"Total events: {len(df)} ({len(historical_events)} historical, {len(future_events)} future)")


if __name__ == '__main__':
    main()