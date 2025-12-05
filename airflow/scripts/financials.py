import time, re, os
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from google.cloud import storage

URL = "https://finance.yahoo.com/quote/TSLA/key-statistics/"
BUCKET_NAME = "tsla_agent"   # UPDATED
PROJECT_ID = "nfl-rag-project"


# -------------------------------
# GCS UPLOAD
# -------------------------------
def upload_to_gcs(bucket_name, local_path, blob_name=None):
    if not os.path.exists(local_path):
        print(f"⚠️ File {local_path} not found — skipping upload.")
        return
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(bucket_name)
    blob_name = blob_name or os.path.basename(local_path)
    bucket.blob(blob_name).upload_from_filename(local_path)
    print(f"☁️ Uploaded {local_path} → gs://{bucket_name}/{blob_name}")


# -------------------------------
# CHROME DRIVER
# -------------------------------
def init_driver():
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument("--lang=en-US")
    return webdriver.Chrome(options=opts)


# -------------------------------
# SCROLL FOR CARDS
# -------------------------------
def scroll_all(driver, times=10, pause=1.2):
    last_h = 0
    for _ in range(times):
        driver.execute_script("window.scrollBy(0, document.body.scrollHeight);")
        time.sleep(pause)
        h = driver.execute_script("return document.body.scrollHeight;")
        if h == last_h:
            break
        last_h = h


# -------------------------------
# PARSE TOP TABLE
# -------------------------------
def parse_simple_table(tbl):
    rows, headers = [], []
    thead_tr = tbl.find("tr", class_="yf-kbx2lo")
    if thead_tr:
        headers = [th.get_text(strip=True) for th in thead_tr.find_all(["th", "td"])]

    body_trs = tbl.find_all("tr", class_=lambda c: c and c.startswith("yf-kbx2lo"))
    if body_trs and thead_tr and body_trs[0] == thead_tr:
        body_trs = body_trs[1:]

    for tr in body_trs:
        tds = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
        if tds:
            rows.append(tds)

    max_len = max(len(headers), max((len(r) for r in rows), default=0))
    if not headers:
        headers = [f"col_{j}" for j in range(max_len)]
    elif len(headers) < max_len:
        headers += [f"col_{j}" for j in range(len(headers), max_len)]

    rows = [r + [""] * (max_len - len(r)) for r in rows]
    return pd.DataFrame(rows, columns=headers)


# -------------------------------
# PARSE INDIVIDUAL CARD TABLES
# -------------------------------
def parse_card_table(section):
    title_el = section.find(["h2", "h3", "header"])
    title = title_el.get_text(strip=True) if title_el else "card"
    table = section.find("table", class_="table yf-vaowmx")
    if not table:
        return title, pd.DataFrame()

    rows = []
    body = table.find("tbody")
    tr_list = body.find_all("tr") if body else table.find_all("tr")

    for tr in tr_list:
        tds = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
        if tds:
            rows.append(tds)

    thead = table.find("thead")
    headers = [th.get_text(strip=True) for th in thead.find_all("th")] if thead else ["Metric", "Value"]

    max_len = max(len(headers), max((len(r) for r in rows), default=0))
    if not headers:
        headers = [f"col_{j}" for j in range(max_len)]
    elif len(headers) < max_len:
        headers += [f"col_{j}" for j in range(len(headers), max_len)]

    rows = [r + [""] * (max_len - len(r)) for r in rows]
    return title, pd.DataFrame(rows, columns=headers)


# -------------------------------
# TEXTUAL ANNOTATION
# -------------------------------
def create_textual_annotation(title, df):
    """Create LLM-friendly textual annotation for the card/table."""
    lines = []
    lines.append(f"Yahoo Finance TSLA Statistics — {title}")
    lines.append("This annotation describes the financial metrics in natural language.\n")

    for _, row in df.iterrows():
        metric = row.iloc[0]
        vals = row.iloc[1:].tolist()

        # Human-readable reasoning
        lines.append(f"Metric: {metric}")
        lines.append(f"Values: {vals}")

        # Quick interpretation
        lines.append(
            f"Interpretation: '{metric}' shows how Tesla's performance is evolving over recent quarters. "
            f"The values {vals} represent historical trends or current standings for this metric."
        )
        lines.append("")

    return "\n".join(lines)


# -------------------------------
# MAIN
# -------------------------------
def main():
    driver = init_driver()
    try:
        driver.get(URL)
        WebDriverWait(driver, 25).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "table.yf-kbx2lo"))
        )
        time.sleep(1.5)

        # ---------------------------
        # Parse top financial metrics table
        # ---------------------------
        soup = BeautifulSoup(driver.page_source, "html.parser")
        top_tbl = soup.find("table", class_="yf-kbx2lo")
        top_df = parse_simple_table(top_tbl) if top_tbl else pd.DataFrame()

        top_csv = "yahoo_tsla_key_statistics_top.csv"
        top_txt = "yahoo_tsla_key_statistics_top_annotation.txt"

        top_df.to_csv(top_csv, index=False)

        # create annotation
        top_annot = create_textual_annotation("Top Financial Metrics", top_df)
        with open(top_txt, "w", encoding="utf-8") as f:
            f.write(top_annot)

        upload_to_gcs(BUCKET_NAME, top_csv)
        upload_to_gcs(BUCKET_NAME, top_txt)

        # ---------------------------
        # Scroll + parse card sections
        scroll_all(driver, times=14, pause=1)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        container = soup.find("div", class_="container yf-14j5zka") or soup
        sections = container.find_all("section", class_="card small tw-p-0 yf-1eotcqa sticky noBackGround")

        for i, sec in enumerate(sections, 1):
            title, df = parse_card_table(sec)

            safe = re.sub(r"[^A-Za-z0-9_-]+", "_", title)[:60] or f"card_{i}"

            csv_out = f"yahoo_tsla_card_{i:02d}_{safe}.csv"
            txt_out = f"yahoo_tsla_card_{i:02d}_{safe}_annotation.txt"

            df.to_csv(csv_out, index=False)

            annot = create_textual_annotation(title, df)
            with open(txt_out, "w", encoding="utf-8") as f:
                f.write(annot)

            upload_to_gcs(BUCKET_NAME, csv_out)
            upload_to_gcs(BUCKET_NAME, txt_out)

            print(f"🗂️ Uploaded card {i}: {title} + annotation")

    finally:
        driver.quit()


if __name__ == "__main__":
    main()
