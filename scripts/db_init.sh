#!/bin/bash
# db_init.sh

# Database connection details (adjust as needed)
DB_USER="unravel_admin"
DB_PASS="your_db_password" # Replace with actual password
DB_HOST="dpg-cvajmjaj1k6c738ta6ug-a.ohio-postgres.render.com"
DB_PORT="5432"
DB_NAME="unravel_ai"

# Construct the database URL
DATABASE_URL="postgresql://$DB_USER:$DB_PASS@$DB_HOST:$DB_PORT/$DB_NAME"

# Set the environment variable
export DATABASE_URL=$DATABASE_URL

# Run the db_init.py script
python3 unravel-ai/scripts/db_init.py

# Create the vector extension
psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -p "$DB_PORT" -c "CREATE EXTENSION IF NOT EXISTS vector;"

echo "Database initialization complete."
