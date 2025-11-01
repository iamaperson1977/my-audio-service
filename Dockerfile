FROM python:3.9-slim

# System deps for Demucs / audio
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app
COPY . .

# Expose a concrete port at build-time (Railway injects $PORT at runtime)
EXPOSE 8000

# Start with Gunicorn; use a shell so $PORT expands. 
# ⚠️ CHANGE the module name on the right if your file isn't named exactly like this.
# If your file is audio_service.py, change to "audio_service:app".
CMD ["sh", "-c", "gunicorn -k gthread -w 2 -t 600 -b 0.0.0.0:${PORT:-8000} audioProcessingService:app"]

