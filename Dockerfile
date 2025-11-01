FROM python:3.9-slim

# System deps for audio I/O and Demucs
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose a concrete port at build time; Railway still injects $PORT at runtime
EXPOSE 8000

# Start with Gunicorn; shell expands $PORT, default to 8000 if missing
CMD ["sh", "-c", "gunicorn -k gthread -w 2 -t 600 -b 0.0.0.0:${PORT:-8000} audioProcessingService:app"]


