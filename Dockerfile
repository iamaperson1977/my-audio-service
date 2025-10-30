# Dockerfile for AI Audio Processing Service

# ==============================================================================
# Stage 1: Base Image & System Dependencies
# ==============================================================================
# Use a Python 3.11 base image (non-slim for better compatibility)
FROM python:3.11

# Set environment variables to prevent interactive prompts during apt-get install
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# Install necessary system dependencies for audio processing libraries
# - build-essential & python3-dev: For compiling some Python packages
# - ffmpeg: Required by pydub and demucs
# - libsndfile1: Required by the soundfile library for WAV/FLAC I/O
# - libportaudio2: May be needed by some audio libraries
RUN apt-get update --fix-missing && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 \
    libportaudio2 \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ==============================================================================
# Stage 2: Python Environment Setup
# ==============================================================================
# Set the working directory inside the container
WORKDIR /app

# Upgrade pip, setuptools, and wheel first for smoother installs
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy ONLY the requirements file first to leverage Docker's build cache
# If requirements.txt doesn't change, Docker won't need to reinstall packages
COPY requirements.txt .

# Install Python dependencies specified in requirements.txt
# Using --no-cache-dir keeps the image smaller
# This MUST include torch and demucs
RUN pip install --no-cache-dir -r requirements.txt

# ==============================================================================
# Stage 3: Application Code & Configuration
# ==============================================================================
# Copy the rest of your application code (audioProcessingService.py, etc.)
COPY . .

# NOTE: No EXPOSE line is needed. 
# Railway automatically injects the $PORT environment variable
# and automatically handles exposing the correct port.

# ==============================================================================
# Stage 4: Runtime Command
# ==============================================================================
# Command to run the application using Gunicorn (production WSGI server)
# - Binds to all network interfaces (0.0.0.0)
# - Uses the PORT environment variable provided by Railway
# - Uses 'sh -c' to correctly substitute the $PORT variable
# - Sets a worker timeout of 120 seconds for potentially long audio processing (like demucs)
# - Starts with 1 worker (can be adjusted based on server resources and load)
# - Assumes your Flask app instance is named 'app' in 'audioProcessingService.py'
CMD sh -c 'gunicorn --bind 0.0.0.0:$PORT --timeout 900 --workers 1 audioProcessingService:app'

