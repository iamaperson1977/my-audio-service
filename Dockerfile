FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python service
COPY audioProcessingService.py .

# Expose port
EXPOSE 5000

# Run the service
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "--timeout", "7200", "audioProcessingService:app"]

