#!/usr/bin/env python3
"""
Unravel AI - Database Initialization Script
Initialize the PostgreSQL database schema for Unravel AI
"""

import os
import sys
import psycopg2
import argparse
import logging
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# SQL Scripts for database initialization
SQL_CREATE_TABLES = """
-- Users table
CREATE TABLE IF NOT EXISTS users (
    user_id VARCHAR(36) PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    company VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP WITH TIME ZONE,
    account_type VARCHAR(20) NOT NULL DEFAULT 'free',
    api_key VARCHAR(64) UNIQUE,
    is_active BOOLEAN DEFAULT TRUE
);

-- Software table to store ingested software
CREATE TABLE IF NOT EXISTS software (
    software_id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) REFERENCES users(user_id),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    file_type VARCHAR(50),
    original_file_path TEXT,
    work_file_path TEXT,
    status VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Decompiled files table
CREATE TABLE IF NOT EXISTS decompiled_files (
    file_id VARCHAR(36) PRIMARY KEY,
    software_id VARCHAR(36) REFERENCES software(software_id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    file_type VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Specification files table
CREATE TABLE IF NOT EXISTS spec_files (
    spec_id VARCHAR(36) PRIMARY KEY,
    software_id VARCHAR(36) REFERENCES software(software_id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Reconstructed software table
CREATE TABLE IF NOT EXISTS reconstructed_software (
    reconstructed_id VARCHAR(36) PRIMARY KEY,
    software_id VARCHAR(36) REFERENCES software(software_id) ON DELETE CASCADE,
    directory_path TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Mimicked software table
CREATE TABLE IF NOT EXISTS mimicked_software (
    mimicked_id VARCHAR(36) PRIMARY KEY,
    software_id VARCHAR(36) REFERENCES software(software_id) ON DELETE CASCADE,
    target_language VARCHAR(50) NOT NULL,
    directory_path TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Licenses table
CREATE TABLE IF NOT EXISTS licenses (
    license_id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) REFERENCES users(user_id),
    software_id VARCHAR(36) REFERENCES software(software_id),
    license_type VARCHAR(50) NOT NULL,
    license_key TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expiration_date TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT TRUE
);

-- Sandbox executions table
CREATE TABLE IF NOT EXISTS sandbox_executions (
    execution_id VARCHAR(36) PRIMARY KEY,
    software_id VARCHAR(36) REFERENCES software(software_id),
    user_id VARCHAR(36) REFERENCES users(user_id),
    container_id VARCHAR(100),
    start_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP WITH TIME ZONE,
    status VARCHAR(50) NOT NULL,
    log_path TEXT
);

-- API usage tracking
CREATE TABLE IF NOT EXISTS api_usage (
    usage_id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) REFERENCES users(user_id),
    endpoint VARCHAR(100) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    response_time_ms INTEGER,
    status_code INTEGER,
    request_size INTEGER,
    response_size INTEGER,
    ip_address VARCHAR(45)
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_software_user_id ON software(user_id);
CREATE INDEX IF NOT EXISTS idx_decompiled_files_software_id ON decompiled_files(software_id);
CREATE INDEX IF NOT EXISTS idx_spec_files_software_id ON spec_files(software_id);
CREATE INDEX IF NOT EXISTS idx_reconstructed_software_id ON reconstructed_software(software_id);
CREATE INDEX IF NOT EXISTS idx_mimicked_software_id ON mimicked_software(software_id);
CREATE INDEX IF NOT EXISTS idx_licenses_user_id ON licenses(user_id);
CREATE INDEX IF NOT EXISTS idx_licenses_software_id ON licenses(software_id);
CREATE INDEX IF NOT EXISTS idx_sandbox_executions_user_id ON sandbox_executions(user_id);
CREATE INDEX IF NOT EXISTS idx_api_usage_user_id ON api_usage(user_id);
CREATE INDEX IF NOT EXISTS idx_api_usage_timestamp ON api_usage(timestamp);
"""

def init_database(db_url):
    """Initialize the database schema"""
    logger.info("Connecting to database...")
    
    try:
        conn = psycopg2.connect(db_url)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        logger.info("Creating tables...")
        cursor.execute(SQL_CREATE_TABLES)
        
        # Create admin user if it doesn't exist
        cursor.execute("""
        INSERT INTO users (user_id, email, password_hash, account_type)
        VALUES ('admin', 'admin@unravelai.com', '$2b$12$IWGGSEr9r5PXqV7vn9bOXe6tgkr55BPh8sgXaqWJ5lMpnJZ0yc5b6', 'admin')
        ON CONFLICT (email) DO NOTHING;
        """)
        
        logger.info("Database initialization completed successfully.")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize Unravel AI database")
    parser.add_argument("--db-url", help="PostgreSQL connection URL", 
                        default=os.environ.get("DATABASE_URL"))
    
    args = parser.parse_args()
    
    if not args.db_url:
        logger.error("Database URL not provided. Use --db-url or set DATABASE_URL environment variable.")
        sys.exit(1)
    
    init_database(args.db_url)
