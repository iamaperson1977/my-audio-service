FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app files
COPY . .

# Expose port (Railway automatically injects $PORT)
EXPOSE $PORT

# Start Flask app with Gunicorn
CMD ["gunicorn", "-k", "gthread", "-w", "2", "-t", "600", "-b", "0.0.0.0:${PORT}", "audioProcessingService:app"]
