# Use a Python base image with a specific version (non-slim)
FROM python:3.11

# Install necessary system dependencies for audio processing libraries (ffmpeg, build tools)
RUN apt-get update --fix-missing && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 \
    libportaudio2 \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Upgrade pip, setuptools, and wheel FIRST
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install your Python dependencies directly in one step
RUN pip install --no-cache-dir \
    Flask==3.0.0 \
    numpy==1.24.3 \
    soundfile==0.12.1 \
    scipy==1.11.4 \
    gunicorn==21.2.0 \
    pydub==0.25.1 \
    librosa==0.10.1 \
    numba==0.58.1

# Copy the rest of your application code
COPY . .

# Expose the port your Flask app will run on
# Railway provides the PORT environment variable automatically
EXPOSE $PORT

# Define the command to run your application using Gunicorn
# Railway automatically uses the PORT variable
CMD sh -c 'gunicorn --bind 0.0.0.0:$PORT audioProcessingService:app'
