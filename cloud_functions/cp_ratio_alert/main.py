"""
Call/Put Ratio Alert Function

This function analyzes call/put ratios from biotech options data and sends alerts 
when a ratio exceeds a specified threshold (default is 15).
"""

import pandas as pd
import numpy as np
import tempfile
import os
import json
import logging
from google.cloud import storage
from google.cloud import pubsub_v1
from google.cloud import secretmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hardcoded configuration values
BUCKET_NAME = 'biotech-options-data'
PROJECT_ID = 'biotech-option-signals'
ALERT_TOPIC = 'biotech-alerts'

def get_polygon_api_key():
    """Get Polygon API key from Secret Manager."""
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{PROJECT_ID}/secrets/polygon_api/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode('UTF-8')

def list_blobs_in_gcs(prefix=None):
    """List all blobs in a Google Cloud Storage bucket with optional prefix."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blobs = bucket.list_blobs(prefix=prefix)
    return [blob.name for blob in blobs]

def read_pickle_from_gcs(blob_name):
    """Read a pickle file directly from Google Cloud Storage."""
    import pickle
    from io import BytesIO
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)
    
    content = blob.download_as_bytes()
    pickle_data = pickle.loads(content)
    return pickle_data

def send_pubsub_message(topic_id, message_data):
    """Send a message to a Pub/Sub topic."""
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(PROJECT_ID, topic_id)
    
    data = json.dumps(message_data).encode('utf-8')
    future = publisher.publish(topic_path, data)
    message_id = future.result()
    
    logger.info(f"Published message to {topic_path} with ID: {message_id}")
    return message_id

def check_call_put_ratio_alerts(request):
    """
    Cloud Function to check call/put ratios and send alerts when > threshold.
    
    This function:
    1. Scans all CP ratio files in the specified GCS bucket
    2. Checks each ticker's most recent call/put volume ratio
    3. Sends an alert via Pub/Sub if the ratio exceeds the threshold
    
    Args:
        request (flask.Request): HTTP request object
        
    Returns:
        dict: Summary of alerts sent
    """
    # Parse request parameters
    request_json = request.get_json(silent=True)
    request_args = request.args
    
    # Check for threshold override in request
    if request_json and 'threshold' in request_json:
        threshold = float(request_json['threshold'])
    elif request_args and 'threshold' in request_args:
        threshold = float(request_args['threshold'])
    else:
        threshold = 15.0  # Default threshold
    
    logger.info(f"Checking for CP ratios > {threshold}")
    
    # Find all CP ratio files in the bucket
    cp_ratio_files = list_blobs_in_gcs(prefix='option_analysis')
    cp_ratio_files = [f for f in cp_ratio_files if '_cp_ratio.pkl' in f]
    
    logger.info(f"Found {len(cp_ratio_files)} CP ratio files to check")
    
    alerts_sent = []
    
    for file_path in cp_ratio_files:
        try:
            # Read the CP ratio data from GCS
            cp_data = read_pickle_from_gcs(file_path)
            
            # Extract ticker and date info from filename
            filename = file_path.split('/')[-1]
            parts = filename.split('_')
            ticker = parts[0]
            
            # Skip empty dataframes
            if cp_data.empty:
                continue
                
            # Check for NaN values
            if 'cp_ratio' not in cp_data.columns:
                logger.warning(f"cp_ratio column not found in {file_path}")
                continue
                
            # Get the latest data point with a valid CP ratio
            latest_data = cp_data[cp_data['cp_ratio'].notna()].iloc[-1] if not cp_data[cp_data['cp_ratio'].notna()].empty else None
            
            if latest_data is not None:
                cp_ratio = latest_data['cp_ratio']
                
                # If CP ratio > threshold, send an alert
                if cp_ratio > threshold:
                    # Get additional data for context
                    call_volume = latest_data.get('call_volume', 0)
                    put_volume = latest_data.get('put_volume', 0)
                    
                    alert_data = {
                        "ticker": ticker,
                        "cp_ratio": float(cp_ratio),
                        "call_volume": int(call_volume),
                        "put_volume": int(put_volume),
                        "timestamp": str(latest_data.name),
                        "alert_type": "HIGH_CP_RATIO",
                        "threshold": threshold,
                        "message": f"HIGH ALERT: {ticker} has call/put ratio of {cp_ratio:.2f} (threshold: {threshold})"
                    }
                    
                    # Send to Pub/Sub
                    message_id = send_pubsub_message(ALERT_TOPIC, alert_data)
                    
                    alerts_sent.append({
                        "ticker": ticker,
                        "cp_ratio": float(cp_ratio),
                        "call_volume": int(call_volume),
                        "put_volume": int(put_volume),
                        "message_id": message_id
                    })
                    
                    logger.info(f"Alert sent for {ticker}: CP ratio = {cp_ratio:.2f}")
        
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
    
    # Return a summary of alerts
    result = {
        "alerts_sent": len(alerts_sent),
        "alert_details": alerts_sent,
        "threshold_used": threshold
    }
    
    return result