# Use an official lightweight Python image
FROM python:3.11-slim

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

# Set working directory inside container
WORKDIR /app

# Copy dependency file first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI application code
COPY . .

# Expose the default FastAPI port
EXPOSE 8080

# Start FastAPI with Uvicorn when container runs
CMD ["uvicorn", "market_service:app", "--host", "0.0.0.0", "--port", "8080"]
