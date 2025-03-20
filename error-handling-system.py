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
        except Exception as e:
            self.handle_exception(
                exception=e,
                category=category,
                severity=severity,
                operation=operation,
                **context_kwargs
            )
            raise  # Re-raise the exception

class GracefulDegradation:
    """
    Implements graceful degradation strategies for different system components
    when errors occur, allowing the system to continue operation with reduced
    functionality rather than failing completely.
    """
    
    def __init__(self, error_manager: ErrorManager = None):
        """
        Initialize graceful degradation
        
        Args:
            error_manager: Error manager instance
        """
        self.error_manager = error_manager or ErrorManager()
        self.fallback_strategies = {}
        self.degradation_state = {}
        
        # Register built-in strategies
        self._register_builtin_strategies()
    
    def _register_builtin_strategies(self):
        """Register built-in fallback strategies"""
        # LLM API errors - retry with exponential backoff
        self.register_fallback(
            component="llm_integration",
            function="generate_completion",
            fallback=self._llm_api_retry_strategy
        )
        
        # Decompilation errors - try alternative decompiler
        self.register_fallback(
            component="decompilation",
            function="decompile_binary",
            fallback=self._alternative_decompiler_strategy
        )
        
        # Parsing errors - try simplified parsing
        self.register_fallback(
            component="parsing",
            function="parse_code",
            fallback=self._simplified_parsing_strategy
        )
    
    def register_fallback(
        self, 
        component: str, 
        function: str, 
        fallback: Callable
    ):
        """
        Register a fallback strategy for a component function
        
        Args:
            component: Component name
            function: Function name
            fallback: Fallback function
        """
        key = f"{component}.{function}"
        self.fallback_strategies[key] = fallback
    
    def get_fallback(self, component: str, function: str) -> Optional[Callable]:
        """
        Get a fallback strategy for a component function
        
        Args:
            component: Component name
            function: Function name
            
        Returns:
            Fallback function or None
        """
        key = f"{component}.{function}"
        return self.fallback_strategies.get(key)
    
    def _llm_api_retry_strategy(self, *args, **kwargs):
        """
        Fallback strategy for LLM API errors - retry with exponential backoff
        """
        max_retries = kwargs.pop('max_retries', 3)
        initial_delay = kwargs.pop('initial_delay', 1.0)
        
        for retry in range(max_retries):
            try:
                # Try to call the original function
                return args[0](*args[1:], **kwargs)
            except Exception as e:
                # Log retry attempt
                logger.warning(f"LLM API error, retry {retry+1}/{max_retries}: {str(e)}")
                
                # Wait with exponential backoff
                delay = initial_delay * (2 ** retry)
                time.sleep(delay)
        
        # If all retries fail, try with a simpler prompt
        try:
            # Simplify the prompt
            if 'prompt' in kwargs:
                kwargs['prompt'] = self._simplify_prompt(kwargs['prompt'])
            elif len(args) > 1:
                args = list(args)
                args[1] = self._simplify_prompt(args[1])
                args = tuple(args)
            
            return args[0](*args[1:], **kwargs)
        except Exception as e:
            # If still failing, raise the exception
            logger.error(f"All LLM API retries failed: {str(e)}")
            raise
    
    def _simplify_prompt(self, prompt: str) -> str:
        """
        Simplify a prompt by reducing its length and complexity
        
        Args:
            prompt: Original prompt
            
        Returns:
            Simplified prompt
        """
        # Split into lines
        lines = prompt.split('\n')
        
        # If prompt is too long, reduce it
        if len(lines) > 50:
            # Keep first 10 and last 30 lines
            lines = lines[:10] + ["..."] + lines[-30:]
        
        # Rejoin and return
        return '\n'.join(lines)
    
    def _alternative_decompiler_strategy(self, *args, **kwargs):
        """
        Fallback strategy for decompilation errors - try alternative decompiler
        """
        # Try different decompilers in order
        decompilers = ["ghidra", "radare2", "retdec", "binary_ninja"]
        
        # Extract the original decompiler
        original_decompiler = kwargs.get('decompiler', 'unknown')
        
        # Remove the original decompiler from the list
        if original_decompiler in decompilers:
            decompilers.remove(original_decompiler)
        
        # Try each alternative decompiler
        for decompiler in decompilers:
            try:
                # Set the alternative decompiler
                kwargs['decompiler'] = decompiler
                
                # Try to call the original function
                return args[0](*args[1:], **kwargs)
            except Exception as e:
                logger.warning(f"Alternative decompiler {decompiler} failed: {str(e)}")
        
        # If all alternatives fail, try with a basic approach
        try:
            # Use a very basic decompilation approach
            kwargs['basic_mode'] = True
            
            return args[0](*args[1:], **kwargs)
        except Exception as e:
            # If still failing, raise the exception
            logger.error(f"All decompilation attempts failed: {str(e)}")
            raise
    
    def _simplified_parsing_strategy(self, *args, **kwargs):
        """
        Fallback strategy for parsing errors - try simplified parsing
        """
        try:
            # Enable simplified parsing mode
            kwargs['simplified'] = True
            
            # Reduce parsing depth if applicable
            if 'max_depth' in kwargs:
                kwargs['max_depth'] = min(kwargs['max_depth'], 2)
            
            # Try to call the original function
            return args[0](*args[1:], **kwargs)
        except Exception as e:
            # If still failing, try a very basic approach
            logger.warning(f"Simplified parsing failed: {str(e)}")
            
            try:
                # Use regex-based parsing as a last resort
                kwargs['regex_fallback'] = True
                
                return args[0](*args[1:], **kwargs)
            except Exception as e2:
                # If all approaches fail, raise the exception
                logger.error(f"All parsing attempts failed: {str(e2)}")
                raise
    
    @contextmanager
    def degradable_operation(
        self, 
        component: str, 
        function: str, 
        *args, 
        **kwargs
    ):
        """
        Context manager for operations that can degrade gracefully
        
        Args:
            component: Component name
            function: Function name
            *args: Original function arguments
            **kwargs: Original function keyword arguments
        """
        try:
            yield
        except Exception as e:
            # Get fallback strategy
            fallback = self.get_fallback(component, function)
            
            if fallback:
                logger.warning(f"Operation {component}.{function} failed, trying fallback: {str(e)}")
                
                # Mark component as degraded
                self.degradation_state[f"{component}.{function}"] = True
                
                # Apply fallback strategy
                return fallback(*args, **kwargs)
            else:
                # No fallback available, re-raise the exception
                raise

class RetryManager:
    """Manages retrying operations with various strategies"""
    
    def __init__(self):
        """Initialize the retry manager"""
        self.default_max_retries = 3
        self.default_initial_delay = 1.0
        self.default_max_delay = 60.0
        self.default_backoff_factor = 2.0
    
    async def retry_async(
        self, 
        operation: Callable, 
        *args, 
        max_retries: int = None, 
        initial_delay: float = None, 
        max_delay: float = None, 
        backoff_factor: float = None, 
        retry_exceptions: Tuple[Type[Exception], ...] = (Exception,),
        **kwargs
    ):
        """
        Retry an asynchronous operation with exponential backoff
        
        Args:
            operation: Async function to retry
            *args: Function arguments
            max_retries: Maximum number of retries
            initial_delay: Initial delay between retries (seconds)
            max_delay: Maximum delay between retries (seconds)
            backoff_factor: Backoff multiplication factor
            retry_exceptions: Exception types to retry on
            **kwargs: Function keyword arguments
            
        Returns:
            Result of the operation
            
        Raises:
            Exception: If all retries fail
        """
        max_retries = max_retries or self.default_max_retries
        initial_delay = initial_delay or self.default_initial_delay
        max_delay = max_delay or self.default_max_delay
        backoff_factor = backoff_factor or self.default_backoff_factor
        
        last_exception = None
        
        for retry in range(max_retries + 1):  # +1 for the initial attempt
            try:
                return await operation(*args, **kwargs)
            except retry_exceptions as e:
                last_exception = e
                
                if retry >= max_retries:
                    # Last attempt failed, re-raise the exception
                    logger.error(f"All retries failed for {operation.__name__}: {str(e)}")
                    raise
                
                # Calculate delay with exponential backoff
                delay = min(initial_delay * (backoff_factor ** retry), max_delay)
                
                # Log retry
                logger.warning(f"Retry {retry+1}/{max_retries} for {operation.__name__} after {delay:.1f}s: {str(e)}")
                
                # Wait before next attempt
                await asyncio.sleep(delay)
    
    def retry(
        self, 
        operation: Callable, 
        *args, 
        max_retries: int = None, 
        initial_delay: float = None, 
        max_delay: float = None, 
        backoff_factor: float = None, 
        retry_exceptions: Tuple[Type[Exception], ...] = (Exception,),
        **kwargs
    ):
        """
        Retry an operation with exponential backoff
        
        Args:
            operation: Function to retry
            *args: Function arguments
            max_retries: Maximum number of retries
            initial_delay: Initial delay between retries (seconds)
            max_delay: Maximum delay between retries (seconds)
            backoff_factor: Backoff multiplication factor
            retry_exceptions: Exception types to retry on
            **kwargs: Function keyword arguments
            
        Returns:
            Result of the operation
            
        Raises:
            Exception: If all retries fail
        """
        max_retries = max_retries or self.default_max_retries
        initial_delay = initial_delay or self.default_initial_delay
        max_delay = max_delay or self.default_max_delay
        backoff_factor = backoff_factor or self.default_backoff_factor
        
        last_exception = None
        
        for retry in range(max_retries + 1):  # +1 for the initial attempt
            try:
                return operation(*args, **kwargs)
            except retry_exceptions as e:
                last_exception = e
                
                if retry >= max_retries:
                    # Last attempt failed, re-raise the exception
                    logger.error(f"All retries failed for {operation.__name__}: {str(e)}")
                    raise
                
                # Calculate delay with exponential backoff
                delay = min(initial_delay * (backoff_factor ** retry), max_delay)
                
                # Log retry
                logger.warning(f"Retry {retry+1}/{max_retries} for {operation.__name__} after {delay:.1f}s: {str(e)}")
                
                # Wait before next attempt
                time.sleep(delay)

class ErrorMonitor:
    """Monitors errors and provides analytics and alerts"""
    
    def __init__(self, error_manager: ErrorManager = None):
        """
        Initialize the error monitor
        
        Args:
            error_manager: Error manager instance
        """
        self.error_manager = error_manager or ErrorManager()
        self.alert_thresholds = {}
        self.alert_callbacks = {}
        self.error_trends = {}
        self.last_alert_time = {}
        self.alert_cooldown = 300  # 5 minutes between alerts
    
    def set_alert_threshold(
        self, 
        category: ErrorCategory, 
        count_threshold: int, 
        time_window: int = 3600
    ):
        """
        Set an alert threshold for an error category
        
        Args:
            category: Error category
            count_threshold: Number of errors to trigger alert
            time_window: Time window for counting errors (seconds)
        """
        self.alert_thresholds[category] = {
            'count': count_threshold,
            'window': time_window
        }
    
    def register_alert_callback(
        self, 
        category: ErrorCategory, 
        callback: Callable[[ErrorCategory, int], None]
    ):
        """
        Register an alert callback for an error category
        
        Args:
            category: Error category
            callback: Alert callback function
        """
        if category not in self.alert_callbacks:
            self.alert_callbacks[category] = []
        self.alert_callbacks[category].append(callback)
    
    def add_error(self, error: EnhancedError):
        """
        Add an error to monitoring
        
        Args:
            error: Enhanced error object
        """
        category = error.category
        
        # Initialize trend data if needed
        if category not in self.error_trends:
            self.error_trends[category] = []
        
        # Add error timestamp
        self.error_trends[category].append(error.timestamp)
        
        # Check alert threshold
        self._check_alert_threshold(category)
    
    def _check_alert_threshold(self, category: ErrorCategory):
        """
        Check if an alert threshold has been reached
        
        Args:
            category: Error category
        """
        if category not in self.alert_thresholds:
            return
        
        threshold = self.alert_thresholds[category]
        count_threshold = threshold['count']
        time_window = threshold['window']
        
        # Get recent errors in the time window
        now = time.time()
        window_start = now - time_window
        
        recent_errors = [t for t in self.error_trends.get(category, []) if t > window_start]
        
        # Update trend data (remove old errors)
        self.error_trends[category] = recent_errors
        
        # Check if threshold is reached
        if len(recent_errors) >= count_threshold:
            # Check cooldown period
            last_alert = self.last_alert_time.get(category, 0)
            if now - last_alert > self.alert_cooldown:
                # Trigger alert
                self._trigger_alert(category, len(recent_errors))
                
                # Update last alert time
                self.last_alert_time[category] = now
    
    def _trigger_alert(self, category: ErrorCategory, error_count: int):
        """
        Trigger alerts for an error category
        
        Args:
            category: Error category
            error_count: Number of recent errors
        """
        logger.warning(f"Alert threshold reached for {category.name}: {error_count} errors")
        
        # Call alert callbacks
        for callback in self.alert_callbacks.get(category, []):
            try:
                callback(category, error_count)
            except Exception as e:
                logger.error(f"Error in alert callback: {str(e)}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get error statistics
        
        Returns:
            Dictionary of error statistics
        """
        stats = {
            'counts_by_category': {},
            'recent_errors': len(self.error_manager.recent_errors),
            'trends': {}
        }
        
        # Count errors by category
        for category, count in self.error_manager.error_counts.items():
            stats['counts_by_category'][category.name] = count
        
        # Calculate error trends
        now = time.time()
        for category, timestamps in self.error_trends.items():
            # Count errors in different time windows
            last_hour = sum(1 for t in timestamps if t > now - 3600)
            last_day = sum(1 for t in timestamps if t > now - 86400)
            
            stats['trends'][category.name] = {
                'last_hour': last_hour,
                'last_day': last_day,
                'total': len(timestamps)
            }
        
        return stats

def main():
    """Example usage of the error handling system"""
    # Initialize components
    error_manager = ErrorManager()
    graceful_degradation = GracefulDegradation(error_manager)
    retry_manager = RetryManager()
    error_monitor = ErrorMonitor(error_manager)
    
    # Set up alert thresholds
    error_monitor.set_alert_threshold(ErrorCategory.API, 5, 3600)  # 5 API errors in 1 hour
    error_monitor.set_alert_threshold(ErrorCategory.LLM, 3, 1800)  # 3 LLM errors in 30 minutes
    
    # Register alert callback
    def alert_callback(category, count):
        print(f"ALERT: {category.name} errors reached {count}")
    
    error_monitor.register_alert_callback(ErrorCategory.API, alert_callback)
    error_monitor.register_alert_callback(ErrorCategory.LLM, alert_callback)
    
    # Example of error context usage
    def risky_operation():
        with error_manager.error_context("file_processing", category=ErrorCategory.PARSING):
            # Some risky operation
            raise ValueError("Sample error")
    
    # Example of retry usage
    def unreliable_api_call():
        raise ConnectionError("API connection failed")
    
    def test_retry():
        try:
            result = retry_manager.retry(
                unreliable_api_call,
                max_retries=2,
                initial_delay=0.1
            )
        except Exception as e:
            error = error_manager.handle_exception(
                e,
                category=ErrorCategory.API,
                operation="api_call"
            )
            error_monitor.add_error(error)
    
    # Example of graceful degradation
    def process_file():
        with graceful_degradation.degradable_operation("parsing", "parse_file"):
            # File processing that might fail
            raise SyntaxError("Invalid file format")
    
    # Run examples
    try:
        risky_operation()
    except Exception:
        pass
    
    test_retry()
    
    try:
        process_file()
    except Exception:
        pass
    
    # Show error statistics
    print("Error Statistics:")
    print(json.dumps(error_monitor.get_error_statistics(), indent=2))

if __name__ == "__main__":
    main()
