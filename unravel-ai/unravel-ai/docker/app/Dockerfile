
# Dockerfile for API Service
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
    curl \
    git \
    gnupg \
    ca-certificates \
    nodejs \
    npm \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Docker CLI for sandbox
RUN curl -fsSL https://get.docker.com -o get-docker.sh && sh get-docker.sh

# Install JS-Beautify
RUN npm install -g js-beautify

# Set up working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .
# Create necessary directories
RUN mkdir -p data/source data/decompiled data/specs data/reconstructed

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose port for web interface
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
