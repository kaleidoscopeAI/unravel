FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Additional dependencies for graph analysis
RUN pip install --no-cache-dir networkx matplotlib

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/graph_analysis

# Set environment variables
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Default command (can be overridden)
CMD ["python", "-m", "src.core.code_graph_analyzer"]
