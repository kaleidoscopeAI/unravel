import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from typing import Optional, AsyncGenerator

Base = declarative_base()

class Database:
    """Database connection manager for Unravel AI"""
    
    def __init__(self, db_url: str):
        """
        Initialize database connection
        
        Args:
            db_url: Database connection URL
        """
        # Convert synchronous URL to async if needed
        if db_url.startswith("postgresql://"):
            self.db_url = db_url.replace("postgresql://", "postgresql+asyncpg://")
        else:
            self.db_url = db_url
        
        self.engine = create_async_engine(
            self.db_url,
            echo=False,
            future=True
        )
        
        self.session_factory = sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
    
    async def connect(self) -> None:
        """Establish connection to database"""
        # This is a no-op as we're using lazy connections
        pass
    
    async def disconnect(self) -> None:
        """Close database connection"""
        await self.engine.dispose()
    
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get a database session
        
        Yields:
            AsyncSession: Database session
        """
        async with self.session_factory() as session:
            try:
                yield session
            finally:
                await session.close()
