# =====================================================
# x_all_topics_scraper.py — STABLE, SCROLLING VERSION
# =====================================================
import os
import time
import pandas as pd
from datetime import datetime

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from google.cloud import storage


# =====================================================
# CONFIG
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SELENIUM_PROFILE = os.path.join(BASE_DIR, "chrome_profile_selenium")
os.makedirs(SELENIUM_PROFILE, exist_ok=True)

TOPICS = ["Tesla", "TSLA", "Tesla Robotaxi", "Tesla Energy","Tesla Optimus", "TSLA News"]

TARGET_POSTS = 60           # minimum posts per tab
MAX_SCROLLS = 65            # prevent infinite loops

BUCKET_NAME = "tsla_agent"
PROJECT_ID = "nfl-rag-project"


# =====================================================
# UPLOAD TO GCS
# =====================================================
def upload_to_gcs(bucket_name, local_path, blob_name=None):
    try:
        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name or os.path.basename(local_path))
        blob.upload_from_filename(local_path)
        print(f"☁️ Uploaded to gs://{bucket_name}/{blob.name}")
    except Exception as e:
        print(f"❌ Upload failed: {e}")


# =====================================================
# DRIVER (CRASH-PROOF)
# =====================================================
def init_driver():
    options = webdriver.ChromeOptions()

    options.add_argument(f"--user-data-dir={SELENIUM_PROFILE}")
    options.add_argument("--profile-directory=Profile1")

    # Stability controls
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--remote-debugging-port=9222")
    options.add_argument("--window-size=1400,950")

    # Make Chrome less bot-detectable
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    service = Service()
    driver = webdriver.Chrome(service=service, options=options)

    print("✅ Chrome launched successfully.")
    return driver


# =====================================================
# EXTRACT POSTS FROM CURRENT VIEW
# =====================================================
def extract_posts(driver, seen):
    out = []

    articles = driver.find_elements(By.CSS_SELECTOR, "article[data-testid='tweet']")
    for art in articles:
        try:
            text_el = art.find_element(By.CSS_SELECTOR, "div[data-testid='tweetText']")
            text = text_el.text.strip()

            time_el = art.find_element(By.TAG_NAME, "time")
            dt_iso = time_el.get_attribute("datetime")
            dt = datetime.fromisoformat(dt_iso.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M")

            key = dt + "|" + text[:40]
            if key not in seen:
                seen.add(key)
                out.append((dt, text))

        except:
            continue

    return out


# =====================================================
# SCROLL + COLLECT LOOP
# =====================================================
def scroll_collect(driver, topic, tab_name):
    print(f"\n🔍 {topic} [{tab_name}] — collecting tweets...")

    seen = set()
    posts = []
    scrolls = 0

    while len(posts) < TARGET_POSTS and scrolls < MAX_SCROLLS:
        batch = extract_posts(driver, seen)
        posts.extend(batch)

        print(f"  ↳ Scroll {scrolls+1} | Posts: {len(posts)}")

        # Scroll down to load more
        driver.execute_script("window.scrollBy(0, 2000);")
        time.sleep(1.8)

        scrolls += 1

    print(f"📌 Final count for {topic} [{tab_name}] → {len(posts)} posts")
    return posts


# =====================================================
# SCRAPE A SINGLE TOPIC
# =====================================================
def scrape_topic(driver, topic):
    topic_rows = []

    # --------------------------------------------
    # TOP TAB
    # --------------------------------------------
    search_url = f"https://x.com/search?q={topic.replace(' ', '%20')}&src=typed_query"
    driver.get(search_url)
    time.sleep(3)

    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "article"))
        )
    except:
        print("⚠️ Failed to load results.")
        return []

    top_posts = scroll_collect(driver, topic, "Top")

    for dt, text in top_posts:
        topic_rows.append({
            "datetime": dt,
            "topic": topic,
            "tab": "Top",
            "context": text
        })

    # --------------------------------------------
    # LATEST TAB
    # --------------------------------------------
    try:
        latest = WebDriverWait(driver, 8).until(
            EC.element_to_be_clickable((By.XPATH, "//a[contains(@href,'f=live')]"))
        )
        latest.click()
        time.sleep(3)

        latest_posts = scroll_collect(driver, topic, "Latest")

        for dt, text in latest_posts:
            topic_rows.append({
                "datetime": dt,
                "topic": topic,
                "tab": "Latest",
                "context": text
            })

    except:
        print("⚠️ Couldn't open Latest tab.")

    return topic_rows


# =====================================================
# MASTER SCRAPER
# =====================================================
def scrape_all_topics():
    driver = init_driver()
    driver.get("https://x.com/home")
    time.sleep(4)

    all_rows = []

    for topic in TOPICS:
        rows = scrape_topic(driver, topic)
        all_rows.extend(rows)

    driver.quit()

    if not all_rows:
        print("⚠️ No data collected.")
        return

    df = pd.DataFrame(all_rows)
    csv_path = "x_topics_scraped.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")

    print(f"\n💾 Saved {len(df)} rows → {csv_path}")
    upload_to_gcs(BUCKET_NAME, csv_path)



# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    scrape_all_topics()
# =====================================================
# x_all_topics_scraper.py — STABLE, SCROLLING VERSION
# =====================================================
import os
import time
import pandas as pd
from datetime import datetime

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from google.cloud import storage


# =====================================================
# CONFIG
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SELENIUM_PROFILE = os.path.join(BASE_DIR, "chrome_profile_selenium")
os.makedirs(SELENIUM_PROFILE, exist_ok=True)

TOPICS = [
    "Tesla", "TSLA", "Tesla Robotaxi", "Tesla Energy",
    "Tesla Optimus", "TSLA News"
]

TARGET_POSTS = 30           # minimum posts per tab
MAX_SCROLLS = 35            # prevent infinite loops

BUCKET_NAME = "tsla_agent"
PROJECT_ID = "nfl-rag-project"


# =====================================================
# UPLOAD TO GCS
# =====================================================
def upload_to_gcs(bucket_name, local_path, blob_name=None):
    try:
        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name or os.path.basename(local_path))
        blob.upload_from_filename(local_path)
        print(f"☁️ Uploaded to gs://{bucket_name}/{blob.name}")
    except Exception as e:
        print(f"❌ Upload failed: {e}")


# =====================================================
# DRIVER (CRASH-PROOF)
# =====================================================
def init_driver():
    options = webdriver.ChromeOptions()

    options.add_argument(f"--user-data-dir={SELENIUM_PROFILE}")
    options.add_argument("--profile-directory=Profile1")

    # Stability controls
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--remote-debugging-port=9222")
    options.add_argument("--window-size=1400,950")

    # Make Chrome less bot-detectable
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    service = Service()
    driver = webdriver.Chrome(service=service, options=options)

    print("✅ Chrome launched successfully.")
    return driver


# =====================================================
# EXTRACT POSTS FROM CURRENT VIEW
# =====================================================
def extract_posts(driver, seen):
    out = []

    articles = driver.find_elements(By.CSS_SELECTOR, "article[data-testid='tweet']")
    for art in articles:
        try:
            text_el = art.find_element(By.CSS_SELECTOR, "div[data-testid='tweetText']")
            text = text_el.text.strip()

            time_el = art.find_element(By.TAG_NAME, "time")
            dt_iso = time_el.get_attribute("datetime")
            dt = datetime.fromisoformat(dt_iso.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M")

            key = dt + "|" + text[:40]
            if key not in seen:
                seen.add(key)
                out.append((dt, text))

        except:
            continue

    return out


# =====================================================
# SCROLL + COLLECT LOOP
# =====================================================
def scroll_collect(driver, topic, tab_name):
    print(f"\n🔍 {topic} [{tab_name}] — collecting tweets...")

    seen = set()
    posts = []
    scrolls = 0

    while len(posts) < TARGET_POSTS and scrolls < MAX_SCROLLS:
        batch = extract_posts(driver, seen)
        posts.extend(batch)

        print(f"  ↳ Scroll {scrolls+1} | Posts: {len(posts)}")

        # Scroll down to load more
        driver.execute_script("window.scrollBy(0, 2000);")
        time.sleep(1.8)

        scrolls += 1

    print(f"📌 Final count for {topic} [{tab_name}] → {len(posts)} posts")
    return posts


# =====================================================
# SCRAPE A SINGLE TOPIC
# =====================================================
def scrape_topic(driver, topic):
    topic_rows = []

    # --------------------------------------------
    # TOP TAB
    # --------------------------------------------
    search_url = f"https://x.com/search?q={topic.replace(' ', '%20')}&src=typed_query"
    driver.get(search_url)
    time.sleep(3)

    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "article"))
        )
    except:
        print("⚠️ Failed to load results.")
        return []

    top_posts = scroll_collect(driver, topic, "Top")

    for dt, text in top_posts:
        topic_rows.append({
            "datetime": dt,
            "topic": topic,
            "tab": "Top",
            "context": text
        })

    # --------------------------------------------
    # LATEST TAB
    # --------------------------------------------
    try:
        latest = WebDriverWait(driver, 8).until(
            EC.element_to_be_clickable((By.XPATH, "//a[contains(@href,'f=live')]"))
        )
        latest.click()
        time.sleep(3)

        latest_posts = scroll_collect(driver, topic, "Latest")

        for dt, text in latest_posts:
            topic_rows.append({
                "datetime": dt,
                "topic": topic,
                "tab": "Latest",
                "context": text
            })

    except:
        print("⚠️ Couldn't open Latest tab.")

    return topic_rows


# =====================================================
# MASTER SCRAPER
# =====================================================
def scrape_all_topics():
    driver = init_driver()
    driver.get("https://x.com/home")
    time.sleep(4)

    all_rows = []

    for topic in TOPICS:
        rows = scrape_topic(driver, topic)
        all_rows.extend(rows)

    driver.quit()

    if not all_rows:
        print("⚠️ No data collected.")
        return

    df = pd.DataFrame(all_rows)
    csv_path = "x_topics_scraped.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")

    print(f"\n💾 Saved {len(df)} rows → {csv_path}")
    upload_to_gcs(BUCKET_NAME, csv_path)


# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    scrape_all_topics()
