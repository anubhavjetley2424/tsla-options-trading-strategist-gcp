# =====================================================
# optioncharts_option_chain_scraper.py (Direct GCS Upload + Annotation)
# =====================================================
import time, os
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from google.cloud import storage
from io import StringIO

# =====================================================
# CONFIG
# =====================================================
BUCKET_NAME = "tsla_agent"   # UPDATED to your new bucket
PROJECT_ID = "nfl-rag-project"


# =====================================================
# UPLOAD DF DIRECTLY TO GCS
# =====================================================
def upload_df_to_gcs(bucket_name, df, blob_name, project_id):
    if df.empty:
        print(f"⚠️ Empty DataFrame — skipping {blob_name}")
        return

    try:
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        client = storage.Client(project=project_id)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_string(csv_buffer.getvalue(), content_type="text/csv")

        print(f"☁️ Uploaded {blob_name} → gs://{bucket_name}/{blob_name}")

    except Exception as e:
        print(f"❌ GCS upload failed: {e}")


# =====================================================
# UPLOAD STRING (TEXT ANNOTATION)
# =====================================================
def upload_text_to_gcs(bucket_name, text, blob_name, project_id):
    try:
        client = storage.Client(project=project_id)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_string(text, content_type="text/plain")

        print(f"📝 Uploaded annotation → gs://{bucket_name}/{blob_name}")

    except Exception as e:
        print(f"❌ Text upload failed: {e}")


# =====================================================
# DRIVER
# =====================================================
def init_driver():
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1920,1080")
    return webdriver.Chrome(options=opts)


# =====================================================
# LOGIN
# =====================================================
def login(driver, email, password):
    driver.get("https://optioncharts.io/users/sign_in")
    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, "user_email")))
    driver.find_element(By.ID, "user_email").send_keys(email)
    pw = driver.find_element(By.ID, "user_password")
    pw.send_keys(password)
    pw.submit()

    WebDriverWait(driver, 20).until_not(EC.presence_of_element_located((By.ID, "user_email")))

    for _ in range(10):
        if driver.get_cookie("_optioncharts_session"):
            print("✅ Logged in successfully.")
            return True
        time.sleep(1)

    print("⚠️ Login cookie missing — continuing anyway.")
    return False


# =====================================================
# PARSE TABLE
# =====================================================
def parse_table(html, table_id, opt_type):
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", {"id": table_id})
    if not table:
        return pd.DataFrame()

    headers = [th.get_text(strip=True) for th in table.select("thead th")]
    rows = [
        [td.get_text(strip=True) for td in tr.find_all("td")]
        for tr in table.select("tbody tr")
    ]

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=headers)
    df["type"] = opt_type
    return df


# =====================================================
# CLEAN
# =====================================================
def clean_dataframe(df):
    if df.empty:
        return df

    df = df[df["Strike"] != "Strike"].copy()

    numeric_cols = [
        "Strike", "Last Price", "Bid", "Mid", "Ask", "Volume",
        "Open Interest", "Implied Volatility", "Delta", "Gamma", "Theta", "Vega"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace("-", "", regex=False)
                .str.replace("%", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.replace("—", "", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="ignore")

    return df


# =====================================================
# TEXTUAL ANNOTATION (LLM READY)
# =====================================================
def create_annotation(df):
    lines = []
    lines.append("TSLA Option Chain — LLM-Friendly Annotation\n")
    lines.append("This text describes the structure of the option chain data:\n")
    lines.append("- Calls and puts for all strikes and expiries\n")
    lines.append("- Includes Greeks: Delta, Gamma, Theta, Vega\n")
    lines.append("- Includes liquidity metrics: Volume, OI\n")
    lines.append("- Includes pricing: Bid, Ask, Mid, Last Price\n\n")

    # Summary statistics per expiry
    expiries = df["expiry"].unique()
    for exp in expiries:
        exp_df = df[df["expiry"] == exp]
        lines.append(f"Expiration: {exp}\n")
        lines.append(f"- Total contracts: {len(exp_df)}")
        lines.append(f"- Calls: {len(exp_df[exp_df['type']=='call'])}")
        lines.append(f"- Puts: {len(exp_df[exp_df['type']=='put'])}")

        try:
            avg_iv = pd.to_numeric(exp_df["Implied Volatility"], errors="coerce").mean()
            lines.append(f"- Avg Implied Volatility: {avg_iv:.2f}%\n")
        except:
            lines.append("- Avg Implied Volatility: unknown\n")

    return "\n".join(lines)


# =====================================================
# SCRAPER
# =====================================================
def scrape_expiries(driver, expiries):
    all_data = []

    for exp in expiries:
        url = (
            f"https://optioncharts.io/options/TSLA/option-chain?"
            f"option_type=&expiration_dates={exp}&view=list&strike_range=all"
        )
        print(f"\n🌐 Fetching {exp}")
        driver.get(url)

        try:
            WebDriverWait(driver, 25).until(
                EC.presence_of_element_located((By.ID, "option_chain_table_id-call"))
            )
            time.sleep(1.5)

            call_df = parse_table(driver.page_source, "option_chain_table_id-call", "call")
            put_df = parse_table(driver.page_source, "option_chain_table_id-put", "put")

            df = pd.concat([call_df, put_df], ignore_index=True)
            if df.empty:
                continue

            df["expiry"] = exp
            df = clean_dataframe(df)
            all_data.append(df)

            print(f"✅ Scraped {len(df)} rows for {exp}")

        except Exception as e:
            print(f"❌ Error fetching {exp}: {e}")

    if not all_data:
        print("⚠️ No data collected.")
        return None, None

    combined = pd.concat(all_data, ignore_index=True)
    annotation_text = create_annotation(combined)
    return combined, annotation_text


# =====================================================
# MAIN
# =====================================================
def main():
    email = "YOUR_EMAIL"
    password = "YOUR_PASSWORD"

    expiries = [
        "2025-11-07:w","2025-11-14:w","2025-11-21:m","2025-11-28:w",
        "2025-12-05:w","2025-12-12:w","2025-12-19:m","2026-01-16:m",
        "2026-02-20:m","2026-03-20:m","2026-04-17:m","2026-05-15:m",
        "2026-06-18:w","2026-07-17:m","2026-08-21:m","2026-09-18:m",
        "2026-12-18:m","2027-01-15:m","2027-06-17:w","2027-12-17:m",
        "2028-01-21:m"
    ]

    driver = init_driver()

    try:
        login(driver, email, password)
        df, annotation = scrape_expiries(driver, expiries)

        if df is not None:
            upload_df_to_gcs(BUCKET_NAME, df, "tsla_option_chain_all.csv", PROJECT_ID)
            upload_text_to_gcs(BUCKET_NAME, annotation, "tsla_option_chain_all_annotation.txt", PROJECT_ID)

    finally:
        driver.quit()


if __name__ == "__main__":
    main()
