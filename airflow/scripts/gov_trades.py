import time
import os
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from google.cloud import storage

URL = "https://trendspider.com/markets/symbols/TSLA/government-trades/"
BUCKET_NAME = "tsla_options"
PROJECT_ID = "nfl-rag-project"

# =====================================================
# Chrome setup
# =====================================================
def init_driver():
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1920,1080")
    return webdriver.Chrome(options=opts)


# =====================================================
# GCS Upload Helper
# =====================================================

def upload_to_gcs(bucket_name, local_path, blob_name=None):
    if not os.path.exists(local_path):
        print(f"⚠️ File {local_path} not found — skipping upload.")
        return
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(bucket_name)
    blob_name = blob_name or os.path.basename(local_path)
    bucket.blob(blob_name).upload_from_filename(local_path)
    print(f"☁️ Uploaded {local_path} → gs://{bucket_name}/{blob_name}")

# =====================================================
# Parse Table
# =====================================================
def parse_table(html):
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_="data-table congresstrading-data-table")
    if not table:
        print("⚠️ No table found.")
        return pd.DataFrame()

    # Extract header
    thead = table.find("thead", class_="data-table__head")
    headers = [th.get_text(strip=True) for th in thead.find_all("th")] if thead else []

    # Extract rows
    tbody = table.find("tbody", class_="data-table__body")
    rows = []
    if tbody:
        for tr in tbody.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if cells:
                rows.append(cells)

    # Normalize inconsistent row lengths
    if not rows:
        print("⚠️ No rows found.")
        return pd.DataFrame(columns=headers)

    max_len = max(len(headers), max((len(r) for r in rows), default=0))
    if not headers:
        headers = [f"col_{j}" for j in range(max_len)]
    elif len(headers) < max_len:
        headers += [f"col_{j}" for j in range(len(headers), max_len)]

    norm_rows = [r + [""] * (max_len - len(r)) for r in rows]
    df = pd.DataFrame(norm_rows, columns=headers)
    return df


# =====================================================
# Main
# =====================================================
def main():
    out = "trendspider_tsla_gov_trades.csv"
    driver = init_driver()
    try:
        driver.get(URL)
        WebDriverWait(driver, 25).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "table.data-table.congresstrading-data-table")
            )
        )
        time.sleep(2)
        df = parse_table(driver.page_source)
        df.to_csv(out, index=False)
        print(f"✅ Saved {len(df)} rows → {out}")

        # Upload to GCS only if data was saved
        upload_to_gcs(BUCKET_NAME, out)

    except Exception as e:
        print(f"❌ Error: {e}")

    finally:
        driver.quit()


if __name__ == "__main__":
    main()
