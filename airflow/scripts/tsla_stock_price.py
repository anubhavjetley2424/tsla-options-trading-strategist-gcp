import time
import re
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from google.cloud import storage
import os
import requests

BUCKET_NAME = "tsla_options"
PROJECT_ID = "nfl-rag-project"

def upload_to_gcs(bucket_name, local_path):
    if not os.path.exists(local_path):
        print(f"⚠️ File {local_path} not found — skipping upload.")
        return
    try:
        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(os.path.basename(local_path))
        blob.upload_from_filename(local_path)
        print(f"☁️ Uploaded {local_path} → gs://{bucket_name}/{os.path.basename(local_path)}")
    except Exception as e:
        print(f"❌ GCS upload failed: {e}")


def fetch_tsla_history():
    print("📈 Fetching TSLA 2-year history via yfinance...")
    tsla = yf.Ticker("TSLA")
    two_years_ago = datetime.utcnow() - timedelta(days=2*365)
    df_hist = tsla.history(start=two_years_ago.strftime("%Y-%m-%d"),
                             end=datetime.utcnow().strftime("%Y-%m-%d"),
                             interval="1d")
    df_hist.to_csv("tsla_price.csv", encoding="utf-8-sig")
    print("💾 Saved tsla_price.csv")
    upload_to_gcs(BUCKET_NAME, "tsla_price.csv")
    return df_hist
