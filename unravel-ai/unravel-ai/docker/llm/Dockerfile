
# Dockerfile for LLM Service
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends     build-essential gcc     && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install huggingface_hub requests numpy psutil setuptools wheel

# Install llama-cpp-python
RUN pip install llama-cpp-python

# Create necessary directories
RUN mkdir -p /app/models

# Copy the LLM interface module
COPY src/core/llm_interface.py /app/llm_interface.py

# Download the model (this might take a while)
# You may need to adjust the MODEL_ID and MODEL_FILE based on your needs
ARG MODEL_ID=TheBloke/Mistral-7B-Instruct-v0.2-GGUF
ARG MODEL_FILE=mistral-7b-instruct-v0.2.Q4_K_M.gguf

RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='$MODEL_ID', filename='$MODEL_FILE', local_dir='/app/models')"

# Expose port
EXPOSE 8100

# Command to run the LLM service
CMD ["python", "src/services/llm_service.py"]
