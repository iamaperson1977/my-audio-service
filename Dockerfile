FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY audioProcessingService.py .

# Expose port (Railway will use $PORT)
EXPOSE $PORT

# Run with gunicorn using Railway's PORT
CMD gunicorn -w 4 -b 0.0.0.0:$PORT --timeout 300 audioProcessingService:app
