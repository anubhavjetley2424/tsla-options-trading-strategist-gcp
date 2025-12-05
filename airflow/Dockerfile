FROM apache/airflow:2.9.0-python3.10

USER root

# ---------------------------------------------------------
# Install system deps for Chrome + Selenium
# ---------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl unzip gnupg2 ca-certificates \
    fonts-liberation libu2f-udev xdg-utils \
    libxss1 libgbm1 libasound2 libnss3 libxshmfence1 \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------
# Install Chrome for Testing (official stable)
# ---------------------------------------------------------
RUN CHROME_URL=$(curl -s https://googlechromelabs.github.io/chrome-for-testing/last-known-good-versions.json \
    | grep -oP '"Stable":.*?{' \
    | grep -oP '"version": "\K[0-9.]+' ) \
    && echo "Installing Chrome $CHROME_URL" \
    && wget -q https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/$CHROME_URL/linux64/chrome-linux64.zip \
    && unzip chrome-linux64.zip -d /opt/chrome \
    && rm chrome-linux64.zip \
    && ln -s /opt/chrome/chrome-linux64/chrome /usr/bin/google-chrome

# ---------------------------------------------------------
# Install ChromeDriver (matching version)
# ---------------------------------------------------------
RUN CHROME_VER=$(google-chrome --version | awk '{print $3}') && \
    wget -q https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/$CHROME_VER/linux64/chromedriver-linux64.zip \
    && unzip chromedriver-linux64.zip -d /usr/local/bin/ \
    && rm chromedriver-linux64.zip \
    && chmod +x /usr/local/bin/chromedriver-linux64/chromedriver \
    && ln -s /usr/local/bin/chromedriver-linux64/chromedriver /usr/bin/chromedriver

# ---------------------------------------------------------
# Install Python dependencies
# ---------------------------------------------------------
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt \
    && pip install --no-cache-dir google-cloud-storage selenium-wire

# ---------------------------------------------------------
# Ensure scripts folder exists
# ---------------------------------------------------------
RUN mkdir -p /opt/airflow/scripts \
    && chown -R airflow:airflow /opt/airflow/scripts

USER airflow
WORKDIR /opt/airflow
