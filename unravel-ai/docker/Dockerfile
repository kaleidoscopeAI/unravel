# Stage 1: Build dependencies
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS builder

# Set up environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

# Install basic dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3-dev build-essential \
    libpq-dev git curl wget software-properties-common \
    file radare2 nodejs npm unzip && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install js-beautify
RUN npm install -g js-beautify

# Create app directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Stage 2: Final image
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Set up environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

# Install basic dependencies (less than builder)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip libpq-dev curl && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy only necessary files from builder
COPY --from=builder /app /app

# Mount persistent directory for models (assuming /mnt/data on Render)
# This will depend on your Render disk setup
#VOLUME /mnt/data
ENV MODEL_DIR=/mnt/data/models  #Or /data

# Download and cache the model (if needed - consider using Render's persistent storage)
# RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
#     model_name = 'mistralai/Mistral-7B-Instruct-v0.2'; \
#     tokenizer = AutoTokenizer.from_pretrained(model_name); \
#     tokenizer.save_pretrained('$MODEL_DIR/mistral7b_tokenizer'); \
#     print('Tokenizer downloaded successfully'); \
#     from optimum.onnxruntime import ORTModelForCausalLM; \
#     model = ORTModelForCausalLM.from_pretrained(model_name, export=True); \
#     model.save_pretrained('$MODEL_DIR/mistral7b_onnx'); \
#     print('Model downloaded and converted to ONNX successfully')"

# Expose port
EXPOSE 8000

# Set health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set the entrypoint
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
