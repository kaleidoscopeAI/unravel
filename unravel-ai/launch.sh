#!/bin/bash
# Unravel AI - Launch Script

set -e

# Load environment variables
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

# Check if virtualenv is activated
if [ -z "$VIRTUAL_ENV" ]; then
  # Check if venv directory exists
  if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
  else
    echo "Warning: Virtual environment not found. Create one with './setup.sh'"
  fi
fi

# Function to check if required environment variables are set
check_env_vars() {
  local missing=false
  
  if [ -z "$DATABASE_URL" ]; then
    echo "Error: DATABASE_URL is not set"
    missing=true
  fi
  
  if [ -z "$SECRET_KEY" ]; then
    echo "Warning: SECRET_KEY is not set. Generating a random one..."
    export SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
  fi
  
  if [ "$missing" = true ]; then
    echo "Please set the required environment variables or create a .env file"
    exit 1
  fi
}

# Function to initialize database
init_database() {
  echo "Initializing database..."
  python scripts/db_init.py --db-url "$DATABASE_URL"
}

# Function to start the API server
start_api() {
  local port=${API_PORT:-8000}
  local host=${API_HOST:-0.0.0.0}
  local reload=${API_DEBUG:-False}
  
  echo "Starting API server on $host:$port..."
  
  if [ "$reload" = "True" ] || [ "$reload" = "true" ]; then
    echo "Running in debug mode with auto-reload..."
    uvicorn src.api.main:app --host $host --port $port --reload
  else
    uvicorn src.api.main:app --host $host --port $port
  fi
}

# Main entry point
case "$1" in
  setup)
    ./setup.sh
    ;;
  init-db)
    check_env_vars
    init_database
    ;;
  start)
    check_env_vars
    start_api
    ;;
  deploy)
    if [ -z "$2" ]; then
      echo "Usage: $0 deploy <api_key>"
      exit 1
    fi
    
    echo "Deploying to Render..."
    python scripts/deploy_render.py --api-key "$2" --repo-url "$(git remote get-url origin)" --env-file .env
    ;;
  *)
    echo "Unravel AI - Launch Script"
    echo "Usage: $0 [setup|init-db|start|deploy]"
    echo ""
    echo "Commands:"
    echo "  setup    - Set up the project (create virtual environment, install dependencies)"
    echo "  init-db  - Initialize the database"
    echo "  start    - Start the API server"
    echo "  deploy   - Deploy to Render (requires Render API key)"
    exit 1
    ;;
esac

exit 0
