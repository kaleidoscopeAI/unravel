#!/bin/bash
# Unravel AI - Render Setup Script
# Run this script to set up and deploy Unravel AI on Render

set -e

echo "===================================================="
echo "Unravel AI - Render Setup and Deployment"
echo "===================================================="

# Configuration
REPO_URL="https://github.com/yourusername/unravel-ai.git"
RENDER_API_KEY="$1"
UNRAVEL_DIR="unravel-ai"

if [ -z "$RENDER_API_KEY" ]; then
  echo "Usage: $0 <RENDER_API_KEY>"
  echo "Please provide your Render API key as an argument"
  exit 1
fi

# Clone repository
if [ ! -d "$UNRAVEL_DIR" ]; then
  echo "Cloning Unravel AI repository..."
  git clone $REPO_URL $UNRAVEL_DIR
  cd $UNRAVEL_DIR
else
  echo "Repository already exists, updating..."
  cd $UNRAVEL_DIR
  git pull
fi

# Set up environment
echo "Setting up environment..."
if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env file from template. Please edit as needed."
else
  echo ".env file already exists."
fi

# Run setup script
echo "Running setup script..."
./setup.sh

# Initialize database schema (using PostgreSQL URL from Render)
echo "Initializing database schema..."
DATABASE_URL="postgresql://unravel_admin:aTY6nh1Pj8SWxtyrFX0S1uydNRIfOxIN@dpg-cvajmjaj1k6c738ta6ug-a.ohio-postgres.render.com/unravel_ai"
python scripts/db_init.py --db-url "$DATABASE_URL"

# Deploy to Render
echo "Deploying to Render..."
python scripts/deploy_render.py --api-key "$RENDER_API_KEY" --repo-url "$REPO_URL" --env-file .env

echo "===================================================="
echo "Setup and deployment complete!"
echo "===================================================="
echo "Your Unravel AI instance will be available at the URL provided by Render."
echo "Check the Render dashboard for deployment status."
