# =====================================================
# nasdaq_institutional_holdings_scraper.py
# =====================================================
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from google.cloud import storage
from datetime import datetime
import pandas as pd
import time, os

# =====================================================
# CONFIG
# =====================================================
URL = "https://www.nasdaq.com/market-activity/stocks/tsla/institutional-holdings?page=1&rows_per_page=10"
BUCKET_NAME = "tsla_options"
PROJECT_ID = "nfl-rag-project"


# =====================================================
# HELPERS
# =====================================================
def init_driver():
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1920,1080")
    driver = webdriver.Chrome(options=opts)
    driver.set_page_load_timeout(30)
    return driver


def upload_to_gcs(bucket_name, local_path):
    """Upload CSV to GCS with explicit project."""
    if not os.path.exists(local_path):
        print(f"⚠️ File {local_path} not found — skipping upload.")
        return
    try:
        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(os.path.basename(local_path))
        blob.upload_from_filename(local_path)
        print(f"☁️ Uploaded {local_path} → gs://{bucket_name}/{blob.name}")
    except Exception as e:
        print(f"❌ GCS upload failed: {e}")


# =====================================================
# SCRAPER LOGIC
# =====================================================
def scrape_nasdaq_institutional_holdings():
    driver = init_driver()
    driver.get(URL)
    print(f"🌐 Navigating to {URL}")
    time.sleep(5)

    data = {}

    # ==============
    # 1️⃣ OWNERSHIP SUMMARY CARD
    # ==============
    try:
        summary_div = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "div.jupiter22-institutional-holdings__ownership-summary")
            )
        )

        rows = summary_div.find_elements(By.CSS_SELECTOR, "div.table-row")
        summary_data = []
        for row in rows:
            try:
                label = row.find_element(By.CSS_SELECTOR, "div.row-label").text.strip()
                value = row.find_element(By.CSS_SELECTOR, "div.row-value").text.strip()
                summary_data.append({"metric": label, "value": value})
            except:
                continue
        data["ownership_summary"] = pd.DataFrame(summary_data)
        print(f"✅ Extracted {len(summary_data)} ownership summary rows.")
    except Exception as e:
        print(f"⚠️ Error extracting ownership summary: {e}")
        data["ownership_summary"] = pd.DataFrame()

    # ==============
    # 2️⃣ ACTIVE POSITIONS TABLE (Shadow DOM)
    # ==============
    try:
        active_section = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "div.jupiter22-institutional-holdings__active-positions")
            )
        )

        # The nsdq-table component lives inside this section
        shadow_host = active_section.find_element(By.CSS_SELECTOR, "nsdq-table")
        table_html = driver.execute_script("return arguments[0].shadowRoot.innerHTML", shadow_host)

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(table_html, "html.parser")

        # Extract header cells
        header_cells = [div.get_text(strip=True)
                        for div in soup.select("div.table-header-cell") if div.get_text(strip=True)]

        # Extract table rows
        rows = soup.select("div.table-row")
        row_data = []
        for r in rows:
            cells = [c.get_text(strip=True)
                     for c in r.select("div.table-cell") if c.get_text(strip=True)]
            if cells:
                row_data.append(cells)

        df_active = pd.DataFrame(row_data, columns=header_cells[:len(row_data[0])] if row_data else [])
        data["active_positions"] = df_active
        print(f"✅ Extracted {len(df_active)} active position rows.")
    except Exception as e:
        print(f"⚠️ Error extracting active positions: {e}")
        data["active_positions"] = pd.DataFrame()

    driver.quit()

    # ==============
    # 3️⃣ EXPORT + UPLOAD
    # ==============
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    summary_path = f"nasdaq_tsla_summary_{timestamp}.csv"
    active_path = f"nasdaq_tsla_active_positions_{timestamp}.csv"

    if not data["ownership_summary"].empty:
        data["ownership_summary"].to_csv(summary_path, index=False)
        upload_to_gcs(BUCKET_NAME, summary_path)

    if not data["active_positions"].empty:
        data["active_positions"].to_csv(active_path, index=False)
        upload_to_gcs(BUCKET_NAME, active_path)

    print("\n✅ Scraping completed successfully.")
    return data


# =====================================================
# ENTRYPOINT
# =====================================================
if __name__ == "__main__":
    scrape_nasdaq_institutional_holdings()
