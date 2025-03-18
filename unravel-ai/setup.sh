#!/bin/bash
# Unravel AI Setup Script

set -e

# Check for required dependencies
echo "Checking dependencies..."
DEPS=("python3" "pip" "virtualenv" "git" "docker")
MISSING_DEPS=()

for dep in "${DEPS[@]}"; do
  if ! command -v $dep &> /dev/null; then
    MISSING_DEPS+=($dep)
  fi
done

if [ ${#MISSING_DEPS[@]} -ne 0 ]; then
  echo "Error: Missing dependencies: ${MISSING_DEPS[@]}"
  echo "Please install these dependencies and try again."
  exit 1
fi

# Set up virtual environment
echo "Setting up virtual environment..."
python3 -m virtualenv venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Set up environment
if [ ! -f .env ]; then
  echo "Creating .env file..."
  cp .env.example .env
  echo "Please edit .env file with your settings."
fi

# Initialize the database if DATABASE_URL is set
if grep -q "DATABASE_URL" .env; then
  DB_URL=$(grep "DATABASE_URL" .env | cut -d '=' -f2-)
  if [ ! -z "$DB_URL" ]; then
    echo "Initializing database..."
    python scripts/db_init.py --db-url "$DB_URL"
  fi
fi

# Create data directories
echo "Creating data directories..."
mkdir -p data/{source,decompiled,specs,reconstructed}

echo "Setup complete!"
echo "To start the API server, run: uvicorn src.api.main:app --reload"
