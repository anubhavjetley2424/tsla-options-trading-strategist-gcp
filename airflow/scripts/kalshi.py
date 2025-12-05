# =====================================================
# kalshi_forecast_scraper.py (Final Minimal Version)
# =====================================================
from seleniumwire import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime
from google.cloud import storage
import pandas as pd
import requests
import time, os

# =====================================================
# CONFIG
# =====================================================
SEARCH_TERMS = ["Tesla", "TSLA"]
BUCKET_NAME = "tsla_agent"
PROJECT_ID = "nfl-rag-project"


# =====================================================
# HELPERS
# =====================================================
def init_driver():
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=opts)
    driver.set_page_load_timeout(25)
    return driver


def upload_to_gcs(bucket_name, local_path, blob_name=None, project_id=None):
    """Upload CSV to Google Cloud Storage."""
    if not os.path.exists(local_path):
        print(f"⚠️ File {local_path} not found — skipping upload.")
        return
    try:
        client = storage.Client(project=project_id)
        bucket = client.bucket(bucket_name)
        blob_name = blob_name or os.path.basename(local_path)
        bucket.blob(blob_name).upload_from_filename(local_path)
        print(f"☁️ Uploaded {local_path} → gs://{bucket_name}/{blob_name}")
    except Exception as e:
        print(f"❌ GCS upload failed: {e}")


def extract_volume(driver):
    """Extract total trading volume."""
    try:
        el = driver.find_element(By.CSS_SELECTOR, "div.flex.items-center.pt-1.pl-1 span")
        return el.text.strip()
    except:
        return None


def extract_rules_summary(driver):
    """Extract rules or description summary."""
    try:
        el = driver.find_element(By.CSS_SELECTOR, "div.flex.flex-col.gap-2.relative span")
        return el.text.strip()
    except:
        return None


def extract_current_value_type(driver):
    """Extracts the type label ('forecast', 'odd', 'chance')."""
    try:
        container = driver.find_element(By.CSS_SELECTOR, "span.inline-flex.items-center")
        spans = container.find_elements(By.TAG_NAME, "span")
        texts = [s.text.strip() for s in spans if s.text.strip()]
        combined_text = " ".join(texts).replace("\n", " ").strip().lower()

        if "forecast" in combined_text:
            return "forecast"
        elif "odd" in combined_text:
            return "odd"
        elif "chance" in combined_text:
            return "chance"
        else:
            return "unknown"
    except Exception as e:
        print(f"⚠️ extract_current_value_type failed: {e}")
        return "unknown"


# =====================================================
# SCRAPER
# =====================================================
def scrape_kalshi_topic(search_term):
    driver = init_driver()
    all_data = []

    try:
        driver.get(f"https://kalshi.com/?search={search_term}")
        time.sleep(6)

        cards = driver.find_elements(By.CSS_SELECTOR, "a.text-text-x10.w-full")
        hrefs = [c.get_attribute("href") for c in cards[:30] if c.get_attribute("href")]
        print(f"🔍 Found {len(hrefs)} markets for {search_term}")

        for href in hrefs:
            try:
                driver.requests.clear()
                driver.get(href)
                WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "h1"))
                )

                title = driver.find_element(By.CSS_SELECTOR, "h1").text.strip()
                vol = extract_volume(driver)
                rules = extract_rules_summary(driver)
                ticker = href.rstrip("/").split("/")[-1].upper().split("-")[0]
                val_type = extract_current_value_type(driver)

                # Capture forecast_history JSON
                forecast_url = None
                start = time.time()
                while time.time() - start < 15:
                    for req in driver.requests:
                        if req.response and "/forecast_history" in req.url and ticker in req.url.upper():
                            forecast_url = req.url
                            break
                    if forecast_url:
                        break
                    time.sleep(1)

                if not forecast_url:
                    print(f"⚠️ No forecast_history for {ticker}")
                    continue

                resp = requests.get(forecast_url)
                if not resp.ok:
                    continue

                data = resp.json().get("forecast_history", [])
                for entry in data:
                    dt = datetime.utcfromtimestamp(entry["end_period_ts"]).strftime("%Y-%m-%d %H:%M")
                    odds = entry.get("numerical_forecast") or entry.get("mean_price")

                    # --- textualise numeric odds / forecasts ---
                    if odds is not None:
                        try:
                            odds_val = float(odds)
                            if odds_val > 1_000_000:
                                odds_str = f"{odds_val:,.0f} points"
                            elif odds_val > 100:
                                odds_str = f"{odds_val:,.1f} forecast value"
                            else:
                                odds_str = f"{odds_val:.2f}% chance"
                        except:
                            odds_str = str(odds)
                    else:
                        odds_str = "unknown odds"

                    # --- build record with odds textualised ---
                    all_data.append({
                        "topic": search_term,
                        "question": title,
                        "ticker": ticker,
                        "datetime": dt,
                        "odds_text": odds_str,
                        "odds_value": odds,
                        "volume": vol,
                        "rules_summary": rules,
                        "current_value_type": val_type
                    })
                    print(f"✅ {search_term} | {title[:60]}... | {odds_str} ({val_type})")

            except Exception as e:
                print(f"❌ Error scraping {href}: {e}")

    finally:
        driver.quit()

    if all_data:
        df = pd.DataFrame(all_data)
        csv_name = f"kalshi_{search_term.lower()}_forecasts.csv"
        df.to_csv(csv_name, index=False, encoding="utf-8")
        print(f"💾 Saved {len(df)} rows → {csv_name}")
        upload_to_gcs(BUCKET_NAME, csv_name, project_id=PROJECT_ID)
        return df
    else:
        print(f"⚠️ No data collected for {search_term}")
        return None


# =====================================================
# ENTRYPOINT
# =====================================================
if __name__ == "__main__":
    for term in SEARCH_TERMS:
        scrape_kalshi_topic(term)
