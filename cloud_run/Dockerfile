FROM python:3.10-slim

ENV PIP_NO_CACHE_DIR=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy requirements first (cache key)
COPY requirements.txt .

# Force reinstall EVERYTHING without cache
RUN pip install --upgrade --no-cache-dir pip setuptools wheel
RUN pip install --force-reinstall --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8080
EXPOSE 8080

CMD ["gunicorn", "--bind", ":8080", "--workers=1", "--threads=8", "--timeout=0", "main:app"]
