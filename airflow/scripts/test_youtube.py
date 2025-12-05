# =====================================================
# youtube_transcript_scraper.py (Final)
# =====================================================

import time
import csv
import os
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

from google.cloud import storage


# =====================================================
# USER CONFIG
# =====================================================
BUCKET_NAME = "tsla_options"
PROJECT_ID = "nfl-rag-project"

BASE_URL = "https://www.youtube-transcript.io/"

YOUTUBE_URLS = [
    "https://www.youtube.com/watch?v=z4cfUJH5Itc",
    "https://www.youtube.com/watch?v=PQymGcHtx20",
    "https://www.youtube.com/watch?v=lsf302_91Ko",
    "https://www.youtube.com/watch?v=Iqr4R2K5H6g&t=4s",
    "https://www.youtube.com/watch?v=Ybmv-XkSuGM",
    "https://www.youtube.com/watch?v=MPaRDNxSqYM",
]

PAGE_LOAD_TIMEOUT = 30
TRANSCRIPT_LOAD_TIMEOUT = 45


# =====================================================
# GCP UPLOAD
# =====================================================
def upload_to_gcs(bucket_name, local_path, blob_name=None):
    """Upload output CSV directly to GCS."""
    if not os.path.exists(local_path):
        print(f"⚠️ File not found → {local_path}")
        return

    try:
        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket(bucket_name)

        blob_name = blob_name or os.path.basename(local_path)
        blob = bucket.blob(blob_name)

        blob.upload_from_filename(local_path)
        print(f"☁️ Uploaded → gs://{bucket_name}/{blob_name}")

    except Exception as e:
        print(f"❌ GCS upload failed: {e}")


# =====================================================
# SELENIUM SETUP
# =====================================================
def setup_driver(headless=False):
    chrome_opts = Options()
    if headless:
        chrome_opts.add_argument("--headless=new")

    chrome_opts.add_argument("--disable-gpu")
    chrome_opts.add_argument("--window-size=1400,1000")
    chrome_opts.add_argument("--no-sandbox")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_opts)
    driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)

    return driver


def open_home(driver):
    driver.get(BASE_URL)

    input_box = WebDriverWait(driver, 20).until(
        EC.presence_of_element_located(
            (
                By.XPATH,
                "//input[contains(@class,'w-full') and contains(@class,'h-12')]",
            )
        )
    )

    button = WebDriverWait(driver, 20).until(
        EC.element_to_be_clickable(
            (
                By.XPATH,
                "//button[contains(@class,'inline-flex') and contains(@class,'font-semibold')]",
            )
        )
    )

    return input_box, button


import re
import pandas as pd

def chunk_transcript_text(text, sentences_per_chunk=4):
    """
    Break transcript into chunks of N sentences each.
    Returns a list of chunk strings.
    """
    # Split into sentences using punctuation boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(sentences[i:i + sentences_per_chunk])
        chunks.append(chunk)

    return chunks


def create_chunked_csv(raw_csv_path, chunked_csv_path):
    """
    Read the large CSV and output chunked version.
    """
    df = pd.read_csv(raw_csv_path)
    rows = []

    for _, row in df.iterrows():
        url = row["video_url"]
        transcript = row["transcript"]

        chunks = chunk_transcript_text(transcript, sentences_per_chunk=4)

        for idx, chunk in enumerate(chunks):
            rows.append({
                "video_url": url,
                "chunk_index": idx,
                "text": chunk
            })

    chunk_df = pd.DataFrame(rows)
    chunk_df.to_csv(chunked_csv_path, index=False, encoding="utf-8")

    print(f"📄 Chunked transcript CSV created → {chunked_csv_path}")
    return chunked_csv_path


def go_to_transcript(driver, video_url):
    input_box, button = open_home(driver)

    input_box.clear()
    input_box.send_keys(video_url)

    button.click()

    time.sleep(5)


# =====================================================
# COPY TRANSCRIPT METHOD (Best Method)
# =====================================================
def copy_transcript(driver):
    """Wait for transcript page → click Copy → read clipboard → return text."""

    # Wait for transcript container to load
    WebDriverWait(driver, TRANSCRIPT_LOAD_TIMEOUT).until(
        EC.presence_of_element_located(
            (
                By.XPATH,
                "//div[contains(@class,'rounded-xl') and contains(@class,'bg-card')]",
            )
        )
    )

    # Find Copy Transcript button
    copy_btn = WebDriverWait(driver, 20).until(
        EC.element_to_be_clickable(
            (
                By.XPATH,
                "//button[contains(@class,'bg-primary') and contains(text(),'Copy')]",
            )
        )
    )

    copy_btn.click()
    time.sleep(1)

    # Read clipboard (browser-side)
    transcript = driver.execute_script("return navigator.clipboard.readText();")

    return transcript


# =====================================================
# MAIN PIPELINE
# =====================================================
def main():
    output_path = "youtube_transcripts.csv"

    # Create initial large CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_url", "transcript"])

        driver = setup_driver(headless=False)

        for url in YOUTUBE_URLS:
            try:
                print(f"\n🎬 Processing: {url}")
                go_to_transcript(driver, url)

                text = copy_transcript(driver)
                print(f"✅ Transcript length: {len(text)}")

                writer.writerow([url, text])

            except Exception as e:
                print(f"❌ Error scraping {url}: {e}")

        driver.quit()

    print(f"\n💾 Saved CSV → {output_path}")

    # -----------------------------------------------
    # NEW STEP: Create chunked version for RAG
    # -----------------------------------------------
    chunked_path = "youtube_transcripts_chunked.csv"
    create_chunked_csv(output_path, chunked_path)

    # Upload both CSVs
 
    upload_to_gcs(BUCKET_NAME, chunked_path, blob_name="youtube_transcripts_chunked.csv")


# =====================================================
# ENTRYPOINT
# =====================================================
if __name__ == "__main__":
    main()
