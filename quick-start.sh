#!/bin/bash
# Unravel AI - Quick Start Script
# This script provides a quick start for testing Unravel AI locally

set -e

echo "===================================================="
echo "Unravel AI - Quick Start"
echo "===================================================="

# Configuration
UNRAVEL_DIR="unravel-ai"
DATABASE_URL="postgresql://unravel_admin:aTY6nh1Pj8SWxtyrFX0S1uydNRIfOxIN@dpg-cvajmjaj1k6c738ta6ug-a.ohio-postgres.render.com/unravel_ai"
SECRET_KEY="development_secret_key_for_testing_only"

# Create project directory from unravel-setup.sh
if [ ! -d "$UNRAVEL_DIR" ]; then
  echo "Setting up Unravel AI project structure..."
  bash unravel-setup.sh
else
  echo "Project directory already exists."
fi

cd $UNRAVEL_DIR

# Set up environment variables
echo "Configuring environment..."
cat > .env << EOL
# Unravel AI Environment Configuration - Quick Start

# Database settings
DATABASE_URL=${DATABASE_URL}

# API settings
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=True

# Security settings
SECRET_KEY=${SECRET_KEY}
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# File storage settings
STORAGE_TYPE=local
STORAGE_PATH=./data

# Sandbox settings
SANDBOX_ENABLED=True
SANDBOX_TIMEOUT=3600
EOL

# Create a simple test file
echo "Creating test file..."
mkdir -p tests/samples
cat > tests/samples/hello.py << EOL
#!/usr/bin/env python3
"""
Simple Hello World program for testing Unravel AI
"""

def greet(name="World"):
    """
    Greets the specified name
    
    Args:
        name: Name to greet
        
    Returns:
        Greeting message
    """
    return f"Hello, {name}!"

class Greeter:
    """Greeter class for object-oriented greetings"""
    
    def __init__(self, prefix="Hello"):
        """
        Initialize greeter
        
        Args:
            prefix: Greeting prefix
        """
        self.prefix = prefix
    
    def greet(self, name="World"):
        """
        Greet someone
        
        Args:
            name: Name to greet
            
        Returns:
            Greeting message
        """
        return f"{self.prefix}, {name}!"

if __name__ == "__main__":
    print(greet())
    
    greeter = Greeter("Greetings")
    print(greeter.greet("Unravel AI User"))
EOL

# Install dependencies and initialize
echo "Setting up environment and dependencies..."
if [ ! -d "venv" ]; then
  python3 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
else
  source venv/bin/activate
fi

# Initialize database
echo "Initializing database..."
export DATABASE_URL
export SECRET_KEY
python scripts/db_init.py --db-url "$DATABASE_URL"

echo "===================================================="
echo "Quick Start Complete!"
echo "===================================================="
echo ""
echo "To start the Unravel AI API server:"
echo "  cd unravel-ai"
echo "  source venv/bin/activate"
echo "  ./launch.sh start"
echo ""
echo "The API will be available at: http://localhost:8000"
echo "Test the API health: http://localhost:8000/health"
echo ""
echo "To test software ingestion (coming in Phase 2):"
echo "  Use the test file at: tests/samples/hello.py"
echo "===================================================="
