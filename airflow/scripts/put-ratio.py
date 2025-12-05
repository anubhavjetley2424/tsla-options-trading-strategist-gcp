import time
import re
import pandas as pd
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from google.cloud import storage
import os

# ======================================
# CONFIG
# ======================================
URL = "https://optioncharts.io/options/TSLA"
BUCKET_NAME = "tsla_agent"        # already correct
PROJECT_ID = "nfl-rag-project"
METRICS_FILE = "tsla_option_metrics.csv"
TABLE_FILE = "tsla_put_call_ratios.csv"
METRICS_ANNOT = "tsla_option_metrics_annotation.txt"
TABLE_ANNOT = "tsla_put_call_ratios_annotation.txt"


# ======================================
# INIT DRIVER
# ======================================
def init_driver():
    opts = Options()
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    opts.page_load_strategy = "eager"
    driver = webdriver.Chrome(options=opts)
    driver.set_page_load_timeout(60)
    return driver


# ======================================
# GCS UPLOAD FILE
# ======================================
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


# ======================================
# GCS UPLOAD TEXT
# ======================================
def upload_text_to_gcs(bucket_name, text, blob_name):
    try:
        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_string(text, content_type="text/plain")
        print(f"📝 Uploaded annotation → gs://{bucket_name}/{blob_name}")
    except Exception as e:
        print(f"❌ Text upload failed: {e}")


# ======================================
# STRING → NUMERIC
# ======================================
def to_number(val: str):
    if val is None or val == "":
        return None

    val = val.replace(",", "").replace("%", "").replace("±", "").strip()

    match = re.search(r"-?\d+\.?\d*", val)
    if match:
        try:
            num = float(match.group())
            return int(num) if num.is_integer() else num
        except:
            return val
    return val


# ======================================
# METRICS SCRAPER
# ======================================
def scrape_metrics(driver):
    print("🌐 Navigating to OptionCharts TSLA page...")
    driver.get(URL)
    time.sleep(3)
    driver.execute_script("window.scrollBy(0, 400);")
    time.sleep(2)

    WebDriverWait(driver, 20).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.col-lg-4"))
    )

    metric_data = []
    metric_blocks = driver.find_elements(By.CSS_SELECTOR, "div.col-lg-4")

    for block in metric_blocks:
        try:
            headers = block.find_elements(By.CSS_SELECTOR, "div.tw-text-sm.tw-text-gray-500")
            values = block.find_elements(By.CSS_SELECTOR, "div.tw-font-semibold")
            for h, v in zip(headers, values):
                metric_data.append({
                    "Metric": h.text.strip(),
                    "Value": v.text.strip()
                })
        except:
            continue

    df = pd.DataFrame(metric_data)
    df["scrape_timestamp"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    df.to_csv(METRICS_FILE, index=False, encoding="utf-8-sig")
    print(f"💾 Saved metrics → {METRICS_FILE}")
    upload_to_gcs(BUCKET_NAME, METRICS_FILE)

    return df


# ======================================
# METRICS ANNOTATION
# ======================================
def annotate_metrics(df):
    lines = []
    lines.append("TSLA OptionCharts — Metrics Annotation\n")
    lines.append("This file explains each scraped TSLA options metric in natural language.\n")

    for _, row in df.iterrows():
        metric = row["Metric"]
        val = row["Value"]

        lines.append(f"Metric: {metric}")
        lines.append(f"Value: {val}")

        # Simple LLM-friendly reasoning
        lines.append(
            f"Interpretation: '{metric}' describes a key TSLA option market indicator. "
            f"The current reading '{val}' represents the latest value as shown on OptionCharts.\n"
        )

    return "\n".join(lines)


# ======================================
# SCRAPE PUT/CALL RATIO TABLE
# ======================================
def scrape_table(driver):
    print("📊 Scraping Put/Call Ratio Table...")

    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.CSS_SELECTOR,
            "table.table.table-sm.table-hover.table-bordered.table-responsive.optioncharts-table-styling"
        ))
    )

    table = driver.find_element(
        By.CSS_SELECTOR,
        "table.table.table-sm.table-hover.table-bordered.table-responsive.optioncharts-table-styling"
    )

    # HEADER ROWS
    header_rows = table.find_elements(By.CSS_SELECTOR, "thead tr")
    top_headers = [th.text.strip() for th in header_rows[0].find_elements(By.TAG_NAME, "th")]
    bottom_headers = [th.text.strip() for th in header_rows[1].find_elements(By.TAG_NAME, "th")]

    merged_headers = []
    for h in top_headers:
        if h in ("Volume", "Open Interest"):
            start_idx = 0 if h == "Volume" else 3
            merged_headers.extend([f"{h} {sub}" for sub in bottom_headers[start_idx:start_idx + 3]])
        elif h:
            merged_headers.append(h)

    # ROWS
    rows = []
    for tr in table.find_elements(By.CSS_SELECTOR, "tbody tr"):
        cells = [td.text.strip() for td in tr.find_elements(By.TAG_NAME, "td")]
        if cells:
            clean_cells = [
                to_number(c) if idx > 0 else c
                for idx, c in enumerate(cells)
            ]
            rows.append(clean_cells)

    df = pd.DataFrame(rows, columns=merged_headers[:len(rows[0])])
    df["scrape_timestamp"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    df.to_csv(TABLE_FILE, index=False, encoding="utf-8-sig")
    print(f"💾 Saved Put/Call Ratio Table → {TABLE_FILE}")
    upload_to_gcs(BUCKET_NAME, TABLE_FILE)

    return df


# ======================================
# TABLE ANNOTATION
# ======================================
def annotate_table(df):
    lines = []
    lines.append("TSLA OptionCharts — Put/Call Ratio Table Annotation\n")
    lines.append("This text describes the meaning of each column in the put/call ratio dataset.\n")

    expiries = df.iloc[:, 0].unique()

    for exp in expiries:
        exp_df = df[df.iloc[:, 0] == exp]
        lines.append(f"\nExpiration: {exp}")
        lines.append(f"- Total rows: {len(exp_df)}")

        # Example: volume call/put reasoning
        cols = df.columns.tolist()
        if "Volume Calls" in cols and "Volume Puts" in cols:
            try:
                call_vol = exp_df["Volume Calls"].astype(float).sum()
                put_vol = exp_df["Volume Puts"].astype(float).sum()
                ratio = call_vol / put_vol if put_vol else None

                lines.append(f"- Call Volume: {call_vol}")
                lines.append(f"- Put Volume: {put_vol}")
                lines.append(f"- Put/Call Ratio: {ratio:.3f}")
            except:
                pass

    return "\n".join(lines)


# ======================================
# MAIN
# ======================================
if __name__ == "__main__":
    driver = init_driver()

    try:
        metrics_df = scrape_metrics(driver)
        metrics_text = annotate_metrics(metrics_df)
        upload_text_to_gcs(BUCKET_NAME, metrics_text, METRICS_ANNOT)

        table_df = scrape_table(driver)
        table_text = annotate_table(table_df)
        upload_text_to_gcs(BUCKET_NAME, table_text, TABLE_ANNOT)

    finally:
        driver.quit()
