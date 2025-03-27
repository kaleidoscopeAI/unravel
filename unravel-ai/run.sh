#!/bin/bash
# Unravel AI Runner Script

# Set script directory as the base directory
cd "$(dirname "$0")"

# Ensure python dependencies
./unravel --help > /dev/null 2>&1 || pip install -r requirements.txt

# Run unravel with passed arguments
./unravel "$@"
