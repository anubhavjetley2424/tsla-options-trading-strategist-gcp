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

URL = "https://optioncharts.io/options/TSLA/expected-move?expiration_dates=2025-11-07%3Aw&option_type=all&strike_range=all"
BUCKET_NAME = "tsla_agent"
PROJECT_ID = "nfl-rag-project"


def init_driver():
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    return webdriver.Chrome(options=opts)


def upload_to_gcs(bucket_name, local_path, blob_name=None):
    if not os.path.exists(local_path):
        print(f"⚠️ File {local_path} not found — skipping upload.")
        return
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(bucket_name)
    blob_name = blob_name or os.path.basename(local_path)
    bucket.blob(blob_name).upload_from_filename(local_path)
    print(f"☁️ Uploaded {local_path} → gs://{bucket_name}/{blob_name}")


def parse_table(html):
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_="table table-sm table-hover table-responsive optioncharts-table-styling")
    if not table:
        return pd.DataFrame()

    # headers
    thead = table.find("thead")
    headers = []
    if thead:
        trs = thead.find_all("tr")
        last = trs[-1] if trs else None
        headers = [th.get_text(strip=True) for th in (last.find_all("th") if last else [])]

    # body rows
    rows = []
    tbody = table.find("tbody")
    if tbody:
        for tr in tbody.find_all("tr"):
            cols = [td.get_text(strip=True) for td in tr.find_all("td")]
            if cols:
                rows.append(cols)

    return pd.DataFrame(rows, columns=headers or None)


def create_textual_annotation(df):
    """Convert expected-move table into LLM-friendly textual annotation."""
    lines = []
    lines.append("TSLA Expected Move — LLM Friendly Annotation")
    lines.append("This file contains natural-language reasoning for each expected-move expiry.\n")

    for _, row in df.iterrows():
        exp = row.get("Expiration")
        em_d = row.get("Expected_Move_Dollars")
        em_p = row.get("Expected_Move_Percentage")
        low = row.get("Lower_Price")
        high = row.get("Upper_Price")
        iv = row.get("Implied_Volatility")

        lines.append(f"Expiration {exp}:")
        lines.append(f"- Expected move: {em_d} ({em_p})")
        lines.append(f"- Expected TSLA range: ${low} → ${high}")
        lines.append(f"- Implied volatility: {iv}%")

        # LLM interpretation
        lines.append(
            f"Interpretation: The market is pricing a move of {em_p} up or down by expiry. "
            f"IV at {iv}% suggests {'high' if float(iv) > 50 else 'moderate'} uncertainty. "
            f"Price is expected to stay between {low} and {high}.\n"
        )

    return "\n".join(lines)


def main():
    driver = init_driver()

    csv_out = "optioncharts_tsla_expected_move.csv"
    text_out = "optioncharts_tsla_expected_move_annotation.txt"

    try:
        driver.get(URL)
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "table.optioncharts-table-styling"))
        )
        time.sleep(1)

        df = parse_table(driver.page_source)

        # Save CSV
        df.to_csv(csv_out, index=False)
        print(f"✅ Saved CSV with {len(df)} rows → {csv_out}")

        # Create annotation txt
        annotation_text = create_textual_annotation(df)
        with open(text_out, "w", encoding="utf-8") as f:
            f.write(annotation_text)
        print(f"📝 Saved textual annotation → {text_out}")

        # Upload both files
        upload_to_gcs(BUCKET_NAME, csv_out)
        upload_to_gcs(BUCKET_NAME, text_out)

    except Exception as e:
        print(f"❌ Error: {e}")

    finally:
        driver.quit()


if __name__ == "__main__":
    main()
