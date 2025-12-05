# =====================================================
# nasdaq_tesla_news_scraper.py
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
BUCKET_NAME = "tsla_options"
PROJECT_ID = "nfl-rag-project"
BASE_URL = "https://www.nasdaq.com"
NEWS_URL = f"{BASE_URL}/market-activity/stocks/tsla/news-headlines"


# =====================================================
# HELPERS
# =====================================================
def init_driver():
    """Initialize headless Chrome."""
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
    """Upload CSV to GCS with explicit project fallback."""
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
# SCRAPER
# =====================================================
def scrape_nasdaq_tesla_news(max_articles=5):
    """Scrape latest Tesla news articles from Nasdaq."""
    driver = init_driver()
    driver.get(NEWS_URL)
    print(f"🌐 Navigating to {NEWS_URL}")
    time.sleep(5)

    # Wait for news list container
    try:
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.jupiter22-c-article-list"))
        )
    except Exception:
        print("⚠️ Could not find news list container.")
        driver.quit()
        return

    soup_items = driver.find_elements(By.CSS_SELECTOR, "li.jupiter22-c-article-list__item.article a.jupiter22-c-article-list__item_title_wrapper")
    links = [a.get_attribute("href") for a in soup_items if a.get_attribute("href")]
    links = links[:max_articles]

    print(f"📰 Found {len(links)} article links to scrape.")

    all_articles = []
    for i, link in enumerate(links, 1):
        print(f"\n➡️ Scraping article {i}/{len(links)}: {link}")
        try:
            driver.get(link)
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "h1.jupiter22-c-hero-article-title"))
            )

            # Title
            title = driver.find_element(By.CSS_SELECTOR, "h1.jupiter22-c-hero-article-title").text.strip()

            # Body paragraphs
            try:
                body_section = driver.find_element(By.CSS_SELECTOR, "section.jupiter22-c-article-body div.body__content")
                paragraphs = [p.text.strip() for p in body_section.find_elements(By.TAG_NAME, "p") if p.text.strip()]
                article_text = " ".join(paragraphs)
            except:
                article_text = ""

            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

            all_articles.append({
                "timestamp": timestamp,
                "title": title,
                "url": link,
                "article_text": article_text
            })
            print(f"✅ Scraped: {title[:80]}...")

            # Return to main news page
            driver.get(NEWS_URL)
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.jupiter22-c-article-list"))
            )
            time.sleep(2)

        except Exception as e:
            print(f"⚠️ Error scraping {link}: {e}")
            try:
                driver.get(NEWS_URL)
                time.sleep(3)
            except:
                pass

    driver.quit()

    # Save and upload results
    if all_articles:
        df = pd.DataFrame(all_articles)
        csv_name = "nasdaq_tesla_news.csv"
        #df.to_csv(csv_name, index=False, encoding="utf-8")
        #print(f"\n💾 Saved {len(df)} articles → {csv_name}")
        upload_to_gcs(BUCKET_NAME, csv_name)
    else:
        print("⚠️ No articles scraped.")


# =====================================================
# ENTRYPOINT
# =====================================================
if __name__ == "__main__":
    scrape_nasdaq_tesla_news(max_articles=5)
