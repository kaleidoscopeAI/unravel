
# Dockerfile for Code Graph Analyzer
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends     build-essential gcc     && rm -rf /var/lib/apt/lists/*

# Install Docker CLI
RUN curl -fsSL https://get.docker.com -o get-docker.sh && sh get-docker.sh

# Set up working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --upgrade pip &&     pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (if needed)
EXPOSE 80

# Command to run the graph analyzer (adjust as needed)
CMD ["python", "src/core/code_graph_analyzer.py", "-i", "/app/code", "-o", "/app/output"]
