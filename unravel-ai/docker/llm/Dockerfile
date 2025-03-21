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

# Additional LLM-specific dependencies
RUN pip install --no-cache-dir huggingface_hub llama-cpp-python>=0.1.77

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/cache
RUN mkdir -p ~/.config/unravel-ai
RUN mkdir -p ~/.cache/unravel-ai/models

# Set up LLM configuration
RUN python -c "from src.utils.config import Config; Config()"

# Run LLM setup script
RUN chmod +x scripts/setup_llm.sh
RUN ./scripts/setup_llm.sh --skip-deps

# Expose port
EXPOSE 8100

# Run the LLM service
CMD ["python", "-m", "src.core.llm_service"]
