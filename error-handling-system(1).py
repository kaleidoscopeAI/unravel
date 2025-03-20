#!/usr/bin/env python3
"""
Kaleidoscope AI - Enhanced Error Handling System
================================================
A comprehensive error handling system for the Kaleidoscope AI software
that provides robust error classification, detailed logging, graceful
degradation, and recovery mechanisms.

This module ensures that errors are properly captured, categorized,
and handled to prevent cascading failures and provide actionable
information.
"""

import os
import sys
import time
import traceback
import logging
import json
import hashlib
import asyncio
from enum import Enum, auto
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, Type, Set
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("kaleidoscope_errors.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Severity levels for errors"""
    DEBUG = auto()      # Minor issues that don't affect operation
    INFO = auto()       # Informational errors, minimal impact
    WARNING = auto()    # Potential problems that should be addressed
    ERROR = auto()      # Serious problems affecting functionality
    CRITICAL = auto()   # Severe errors that prevent operation
    FATAL = auto()      # Catastrophic errors requiring immediate attention

class ErrorCategory(Enum):
    """Categories for different types of errors"""
    SYSTEM = auto()           # Operating system or environment errors
    NETWORK = auto()          # Network connectivity issues
    API = auto()              # API interaction errors
    PARSING = auto()          # Errors parsing files or data
    ANALYSIS = auto()         # Errors during code analysis
    DECOMPILATION = auto()    # Errors in decompilation process
    SPECIFICATION = auto()    # Errors generating specifications
    RECONSTRUCTION = auto()   # Errors during code reconstruction
    MIMICRY = auto()          # Errors during code mimicry
    LLM = auto()              # LLM API or integration errors
    SECURITY = auto()         # Security-related errors
    RESOURCE = auto()         # Resource availability errors
    VALIDATION = auto()       # Input validation errors
    RECOVERY = auto()         # Recovery process errors
    UNKNOWN = auto()          # Unclassified errors

@dataclass
class ErrorContext:
    """Contextual information about an error"""
    operation: str                          # Operation being performed when error occurred
    input_data: Optional[Dict[str, Any]] = None  # Input data related to the error
    file_path: Optional[str] = None         # Path to relevant file
    component: Optional[str] = None         # Component where error occurred
    additional_info: Dict[str, Any] = field(default_factory=dict)  # Additional information
    timestamp: float = field(default_factory=time.time)  # Error timestamp

@dataclass
class EnhancedError:
    """Enhanced error object with detailed information"""
    message: str                            # Error message
    category: ErrorCategory                 # Error category
    severity: ErrorSeverity                 # Error severity
    exception: Optional[Exception] = None   # Original exception
    traceback: Optional[str] = None         # Exception traceback
    context: Optional[ErrorContext] = None  # Error context
    error_id: str = field(default_factory=lambda: hashlib.md5(str(time.time()).encode()).hexdigest())
    timestamp: float = field(default_factory=time.time)  # Error timestamp
    
    def __post_init__(self):
        """Initialize traceback if exception is provided"""
        if self.exception and not self.traceback:
            self.traceback = ''.join(traceback.format_exception(
                type(self.exception), 
                self.exception, 
                self.exception.__traceback__
            ))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = {
            "error_id": self.error_id,
            "message": self.message,
            "category": self.category.name,
            "severity": self.severity.name,
            "timestamp": self.timestamp
        }
        
        if self.traceback:
            result["traceback"] = self.traceback
        
        if self.context:
            result["context"] = asdict(self.context)
        
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    def log(self):
        """Log the error based on severity"""
        log_message = f"[{self.error_id}] {self.message}"
        
        if self.context and self.context.operation:
            log_message += f" (during {self.context.operation})"
        
        if self.severity == ErrorSeverity.DEBUG:
            logger.debug(log_message)
        elif self.severity == ErrorSeverity.INFO:
            logger.info(log_message)
        elif self.severity == ErrorSeverity.WARNING:
            logger.warning(log_message)
        elif self.severity == ErrorSeverity.ERROR:
            logger.error(log_message)
            if self.traceback:
                logger.error(f"Traceback: {self.traceback}")
        elif self.severity == ErrorSeverity.CRITICAL or self.severity == ErrorSeverity.FATAL:
            logger.critical(log_message)
            if self.traceback:
                logger.critical(f"Traceback: {self.traceback}")

class ErrorHandlerRegistry:
    """Registry of error handlers for different categories and severities"""
    
    def __init__(self):
        """Initialize the registry"""
        self.handlers: Dict[Tuple[ErrorCategory, ErrorSeverity], List[Callable]] = {}
        self.default_handlers: List[Callable] = []
    
    def register_handler(
        self, 
        handler: Callable[[EnhancedError], None], 
        category: Optional[ErrorCategory] = None, 
        severity: Optional[ErrorSeverity] = None
    ):
        """
        Register an error handler
        
        Args:
            handler: Error handler function
            category: Error category to handle (None for all)
            severity: Error severity to handle (None for all)
        """
        if category is None or severity is None:
            self.default_handlers.append(handler)
        else:
            key = (category, severity)
            if key not in self.handlers:
                self.handlers[key] = []
            self.handlers[key].append(handler)
    
    def get_handlers(self, error: EnhancedError) -> List[Callable]:
        """
        Get handlers for an error
        
        Args:
            error: Enhanced error object
            
        Returns:
            List of handler functions
        """
        # Get specific handlers
        key = (error.category, error.severity)
        specific_handlers = self.handlers.get(key, [])
        
        # Get category-only handlers
        category_handlers = []
        for (cat, sev), handlers in self.handlers.items():
            if cat == error.category and sev is None:
                category_handlers.extend(handlers)
        
        # Get severity-only handlers
        severity_handlers = []
        for (cat, sev), handlers in self.handlers.items():
            if cat is None and sev == error.severity:
                severity_handlers.extend(handlers)
        
        # Combine all handlers
        all_handlers = specific_handlers + category_handlers + severity_handlers + self.default_handlers
        
        return all_handlers

class ErrorManager:
    """Central error management system"""
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern implementation"""
        if cls._instance is None:
            cls._instance = super(ErrorManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, error_log_path: str = "errors.json"):
        """
        Initialize the error manager
        
        Args:
            error_log_path: Path to the error log file
        """
        # Skip re-initialization for singleton
        if self._initialized:
            return
        
        self.error_log_path = error_log_path
        self.registry = ErrorHandlerRegistry()
        self.recent_errors: List[EnhancedError] = []
        self.max_recent_errors = 100
        self.error_counts: Dict[ErrorCategory, int] = {category: 0 for category in ErrorCategory}
        self.recovery_strategies: Dict[ErrorCategory, List[Callable]] = {}
        self._initialized = True
        
        # Register built-in handlers
        self._register_builtin_handlers()
    
    def _register_builtin_handlers(self):
        """Register built-in error handlers"""
        # Log all errors
        self.registry.register_handler(self._log_error_handler)
        
        # Save critical and fatal errors to file
        self.registry.register_handler(
            self._save_error_handler, 
            category=None, 
            severity=ErrorSeverity.CRITICAL
        )
        self.registry.register_handler(
            self._save_error_handler, 
            category=None, 
            severity=ErrorSeverity.FATAL
        )
    
    def register_recovery_strategy(
        self, 
        strategy: Callable[[EnhancedError], bool], 
        category: ErrorCategory
    ):
        """
        Register a recovery strategy for an error category
        
        Args:
            strategy: Recovery strategy function
            category: Error category to handle
        """
        if category not in self.recovery_strategies:
            self.recovery_strategies[category] = []
        self.recovery_strategies[category].append(strategy)
    
    def handle_error(self, error: EnhancedError) -> bool:
        """
        Handle an error
        
        Args:
            error: Enhanced error object
            
        Returns:
            Whether the error was handled successfully
        """
        # Add to recent errors
        self.recent_errors.append(error)
        if len(self.recent_errors) > self.max_recent_errors:
            self.recent_errors.pop(0)
        
        # Update error counts
        self.error_counts[error.category] = self.error_counts.get(error.category, 0) + 1
        
        # Log the error
        error.log()
        
        # Get and run handlers
        handlers = self.registry.get_handlers(error)
        for handler in handlers:
            try:
                handler(error)
            except Exception as e:
                logger.error(f"Error in handler {handler.__name__}: {str(e)}")
        
        # Try recovery strategies
        if error.category in self.recovery_strategies:
            for strategy in self.recovery_strategies[error.category]:
                try:
                    if strategy(error):
                        return True
                except Exception as e:
                    logger.error(f"Error in recovery strategy: {str(e)}")
        
        return False
    
    def _log_error_handler(self, error: EnhancedError):
        """Log error handler"""
        # Already logged by error.log()
        pass
    
    def _save_error_handler(self, error: EnhancedError):
        """Save error to file handler"""
        try:
            # Read existing errors
            errors = []
            if os.path.exists(self.error_log_path):
                try:
                    with open(self.error_log_path, 'r') as f:
                        errors = json.load(f)
                except json.JSONDecodeError:
                    # If file is corrupted, start fresh
                    errors = []
            
            # Add new error
            errors.append(error.to_dict())
            
            # Write back to file
            with open(self.error_log_path, 'w') as f:
                json.dump(errors, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving error to file: {str(e)}")
    
    def create_error(
        self, 
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        exception: Optional[Exception] = None,
        operation: Optional[str] = None,
        **context_kwargs
    ) -> EnhancedError:
        """
        Create an enhanced error
        
        Args:
            message: Error message
            category: Error category
            severity: Error severity
            exception: Original exception
            operation: Operation being performed
            **context_kwargs: Additional context information
            
        Returns:
            Enhanced error object
        """
        # Create context if needed
        context = None
        if operation or context_kwargs:
            context = ErrorContext(
                operation=operation or "unknown",
                **context_kwargs
            )
        
        # Create error
        error = EnhancedError(
            message=message,
            category=category,
            severity=severity,
            exception=exception,
            context=context
        )
        
        return error
    
    def handle_exception(
        self,
        exception: Exception,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        operation: Optional[str] = None,
        **context_kwargs
    ) -> EnhancedError:
        """
        Handle an exception
        
        Args:
            exception: Exception to handle
            category: Error category
            severity: Error severity
            operation: Operation being performed
            **context_kwargs: Additional context information
            
        Returns:
            Enhanced error object
        """
        # Create error
        error = self.create_error(
            message=str(exception),
            category=category,
            severity=severity,
            exception=exception,
            operation=operation,
            **context_kwargs
        )
        
        # Handle error
        self.handle_error(error)
        
        return error
    
    @contextmanager
    def error_context(
        self,
        operation: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        **context_kwargs
    ):
        """
        Context manager for error handling
        
        Args:
            operation: Operation being performed
            category: Error category
            severity: Error severity
            **context_kwargs: Additional context information
        """
        try:
            yield
        except Exception as e