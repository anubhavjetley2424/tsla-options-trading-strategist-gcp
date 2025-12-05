import time
import pandas as pd
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
URL = "https://finviz.com/quote.ashx?t=TSLA&ty=si&p=d"
BUCKET_NAME = "tsla_agent"
PROJECT_ID = "nfl-rag-project"
OUT_FILE = "tsla_short_interest.csv"
ANNOT_FILE = "tsla_short_interest_annotation.txt"


# ======================================
# INIT DRIVER
# ======================================
def init_driver():
    opts = Options()
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--user-agent=Mozilla/5.0")
    opts.page_load_strategy = "eager"
    driver = webdriver.Chrome(options=opts)
    driver.set_page_load_timeout(30)
    return driver


# ======================================
# UPLOAD
# ======================================
def upload_to_gcs(bucket_name, local_path):
    if not os.path.exists(local_path):
        return
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(os.path.basename(local_path))
    blob.upload_from_filename(local_path)
    print(f"☁️ Uploaded {local_path} → gs://{bucket_name}/{blob.name}")


def upload_text_to_gcs(bucket_name, text, blob_name):
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(text, content_type="text/plain")
    print(f"📝 Uploaded annotation → gs://{bucket_name}/{blob.name}")


# ======================================
# SCRAPE FINVIZ SHORT INTEREST
# ======================================
def scrape_short_interest(driver):
    print("🌐 Navigating to Finviz Short Interest page...")
    driver.get(URL)

    time.sleep(5)
    driver.execute_script("window.scrollBy(0, 200);")

    table_xpath = "//table[contains(@class,'financials-table') and contains(@class,'table-fixed')]"
    table_elem = WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.XPATH, table_xpath))
    )

    header_cells = table_elem.find_elements(By.XPATH, ".//thead//tr//*[self::th or self::td]")
    headers = [th.text.strip() for th in header_cells]

    rows = []
    for tr in table_elem.find_elements(By.XPATH, ".//tbody//tr"):
        cells = tr.find_elements(By.XPATH, ".//*[self::td or self::th]")
        row = [td.text.strip() for td in cells]
        if any(row):
            rows.append(row)

    max_cols = max(len(r) for r in rows)
    if len(headers) < max_cols:
        headers += [f"Col_{i}" for i in range(len(headers), max_cols)]

    df = pd.DataFrame(rows, columns=headers)
    df.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    print(f"💾 Saved {OUT_FILE}")

    upload_to_gcs(BUCKET_NAME, OUT_FILE)
    return df


# ======================================
# ANNOTATION
# ======================================
def annotate_short_interest(df):
    lines = []
    lines.append("TSLA Finviz Short Interest — LLM Annotation\n")
    lines.append("This dataset represents short interest metrics scraped from Finviz.\n")

    for _, row in df.iterrows():
        metric = row.iloc[0]
        values = row.iloc[1:].tolist()

        lines.append(f"Metric: {metric}")
        lines.append(f"Values: {values}")
        lines.append(
            f"Interpretation: '{metric}' describes a short interest KPI such as float short %, "
            f"days-to-cover, or shares short. Values {values} represent the current reading."
        )
        lines.append("")

    return "\n".join(lines)


# ======================================
# MAIN
# ======================================
if __name__ == "__main__":
    driver = init_driver()
    try:
        df = scrape_short_interest(driver)

        annotation = annotate_short_interest(df)
        with open(ANNOT_FILE, "w", encoding="utf-8") as f:
            f.write(annotation)

        upload_text_to_gcs(BUCKET_NAME, annotation, ANNOT_FILE)

    finally:
        driver.quit()
