#!/bin/bash
# LLM setup script for Unravel AI

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
CACHE_DIR="${HOME}/.cache/unravel-ai"
MODEL_DIR="${CACHE_DIR}/models"
CONFIG_DIR="${HOME}/.config/unravel-ai"

# Process arguments
SKIP_DEPS=0
FORCE=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-deps) SKIP_DEPS=1; shift ;;
        --force) FORCE=1; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Install Python dependencies if needed
if [ $SKIP_DEPS -eq 0 ]; then
    echo "Installing LLM Python dependencies..."
    pip install huggingface_hub requests numpy psutil
    pip install llama-cpp-python
fi

# Check for LLM config
if [ ! -f "${CONFIG_DIR}/llm_config.json" ]; then
    echo "LLM configuration not found at ${CONFIG_DIR}/llm_config.json"
    exit 1
fi

# Extract model info from config
MODEL_ID=$(grep -o '"model_id": *"[^"]*"' ${CONFIG_DIR}/llm_config.json | cut -d'"' -f4)
MODEL_FILE=$(grep -o '"model_file": *"[^"]*"' ${CONFIG_DIR}/llm_config.json | cut -d'"' -f4)

# Download model if needed
if [ -n "$MODEL_ID" ] && [ -n "$MODEL_FILE" ]; then
    MODEL_DIR_SAFE="${MODEL_DIR}/${MODEL_ID//\//_}"
    mkdir -p "${MODEL_DIR_SAFE}"
    
    if [ ! -f "${MODEL_DIR_SAFE}/${MODEL_FILE}" ] || [ $FORCE -eq 1 ]; then
        echo "Downloading model ${MODEL_ID}/${MODEL_FILE}..."
        python -c "
from huggingface_hub import hf_hub_download
import os

try:
    model_path = hf_hub_download(
        repo_id='${MODEL_ID}', 
        filename='${MODEL_FILE}',
        local_dir='${MODEL_DIR_SAFE}',
        force_download=${FORCE}
    )
    print(f'Model downloaded to: {model_path}')
except Exception as e:
    print(f'Error downloading model: {str(e)}')
    exit(1)
"
    else
        echo "Model already exists at ${MODEL_DIR_SAFE}/${MODEL_FILE}"
    fi
fi

echo "LLM setup completed!"
