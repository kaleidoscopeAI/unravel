#!/usr/bin/env python3
"""
Unravel AI - Main Application Entry
Initializes and launches the Unravel AI system
"""

import os
import sys
import asyncio
import logging
import argparse
from pathlib import Path
import uvicorn
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Import internal modules
from src.api.routes import router as api_router
from src.utils.config import initialize_config, config
from src.data.database import get_database
from src.sandbox.manager import SandboxManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), '../logs/unravel.log'), mode='a'),
    ]
)
logger = logging.getLogger(__name__)

# Create directories
os.makedirs(os.path.join(os.path.dirname(__file__), '../logs'), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), '../data'), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), '../data/uploads'), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), '../data/results'), exist_ok=True)

# Global sandbox manager
sandbox_manager = SandboxManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handle application startup and shutdown events
    """
    # On startup
    logger.info("Starting Unravel AI...")
    
    # Initialize database tables
    db = get_database()
    try:
        db.create_tables()
    except Exception as e:
        logger.error(f"Error creating database tables: {str(e)}")
    
    # Load background tasks
    background_tasks = []
    
    try:
        # Start sandbox monitoring task
        async def cleanup_sandboxes():
            while True:
                try:
                    # Check for expired sandboxes
                    logger.debug("Checking for expired sandboxes")
                    sandbox_manager.cleanup_expired()
                except Exception as e:
                    logger.error(f"Error in sandbox cleanup: {str(e)}")
                
                # Check every minute
                await asyncio.sleep(60)
        
        # Start background task
        cleanup_task = asyncio.create_task(cleanup_sandboxes())
        background_tasks.append(cleanup_task)
        
        logger.info("Background tasks started")
    except Exception as e:
        logger.error(f"Error starting background tasks: {str(e)}")
    
    # Yield control to FastAPI
    yield
    
    # On shutdown
    logger.info("Shutting down Unravel AI...")
    
    # Clean up background tasks
    for task in background_tasks:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    
    # Clean up resources
    sandbox_manager.cleanup_all()
    
    logger.info("Shutdown complete")

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application
    """
    # Initialize configuration
    initialize_config()
    
    # Create FastAPI app
    app = FastAPI(
        title="Unravel AI",
        description="Software ingestion and mimicry system",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, replace with specific origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API routers
    app.include_router(api_router)
    
    # Mount static files if UI directory exists
    ui_dir = os.path.join(os.path.dirname(__file__), '../ui/dist')
    if os.path.exists(ui_dir):
        app.mount("/", StaticFiles(directory=ui_dir, html=True), name="ui")
    
    return app

# Create app instance
app = create_app()

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Unravel AI Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", type=str, default="info", 
                       choices=["debug", "info", "warning", "error", "critical"],
                       help="Logging level")
    parser.add_argument("--init-db", action="store_true", help="Initialize database and exit")
    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    # Initialize database if requested
    if args.init_db:
        db = get_database()
        db.create_tables()
        logger.info("Database initialized successfully")
        return
    
    # Get configuration
    host = args.host or config.get("API_HOST", "0.0.0.0")
    port = args.port or int(config.get("API_PORT", 8000))
    reload = args.reload or config.get("API_DEBUG", False)
    
    # Display startup banner
    logger.info("="*60)
    logger.info(f"Starting Unravel AI on http://{host}:{port}")
    logger.info(f"Environment: {'Development' if reload else 'Production'}")
    logger.info(f"Debug mode: {'Enabled' if reload else 'Disabled'}")
    logger.info("="*60)
    
    # Start the server
    uvicorn.run(
        "src.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=args.log_level.lower()
    )

if __name__ == "__main__":
    main()