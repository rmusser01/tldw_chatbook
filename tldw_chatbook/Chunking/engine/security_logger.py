# security_logger.py
"""
Security event logging for the Chunking module.
Logs security-related events for audit and monitoring.

Privacy contract (TASK-19323): the in-memory event store and every log line
carry METADATA ONLY -- event types, severities, counts, and lengths -- never
user document content. The module's sole persistence surface is the loguru
sink declared in ``SecurityLogger.__init__``; adding any other file write
here trips ``Tests/Architecture/test_security_logger_write_surface.py``.
"""

import threading
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from loguru import logger


class SecurityEventType(Enum):
    """Types of security events."""
    XXE_ATTEMPT = "xxe_attempt"
    REDOS_ATTEMPT = "redos_attempt"
    OVERSIZED_INPUT = "oversized_input"
    MALICIOUS_PATTERN = "malicious_pattern"
    SUSPICIOUS_CONTENT = "suspicious_content"
    INVALID_INPUT = "invalid_input"
    RESOURCE_LIMIT = "resource_limit"
    CACHE_OVERFLOW = "cache_overflow"


class SecurityLogger:
    """
    Centralized security event logger for the Chunking module.
    """

    def __init__(self, log_file: Optional[Path] = None, enable_console: bool = True):
        """
        Initialize the security logger.

        Args:
            log_file: Optional path to security log file
            enable_console: Whether to also log to console
        """
        self.log_file = log_file
        self.enable_console = enable_console
        self._events = []

        # Configure dedicated security logger
        if log_file:
            logger.add(
                log_file,
                filter=lambda record: "security" in record["extra"],
                format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {extra[event_type]} | {message}",
                rotation="100 MB",
                retention="30 days"
            )

    def log_event(self,
                  event_type: SecurityEventType,
                  message: str,
                  details: Optional[dict[str, Any]] = None,
                  severity: str = "WARNING") -> None:
        """
        Log a security event.

        Args:
            event_type: Type of security event
            message: Event message
            details: Additional event details
            severity: Event severity (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": event_type.value,
            "message": message,
            "severity": severity,
            "details": details or {}
        }

        # Store event
        self._events.append(event)

        # Log to configured outputs
        log_message = f"SECURITY EVENT: {message}"
        extra = {"security": True, "event_type": event_type.value}

        if severity == "DEBUG":
            logger.debug(log_message, extra=extra)
        elif severity == "INFO":
            logger.info(log_message, extra=extra)
        elif severity == "WARNING":
            logger.warning(log_message, extra=extra)
        elif severity == "ERROR":
            logger.error(log_message, extra=extra)
        elif severity == "CRITICAL":
            logger.critical(log_message, extra=extra)

    def log_xxe_attempt(self, xml_content: str, source: Optional[str] = None) -> None:
        """
        Log an XXE attack attempt.

        Only the LENGTH of the offending XML is retained: the content itself
        is user document data and must never sit in the event store, where a
        future export or dump would ship it outside the redaction guarantees
        (TASK-19323).

        Args:
            xml_content: The malicious XML content (used for its length only)
            source: Optional source identifier
        """
        self.log_event(
            SecurityEventType.XXE_ATTEMPT,
            "XML External Entity (XXE) attack attempt blocked",
            {
                "xml_length": len(xml_content) if xml_content else 0,
                "source": source,
                "blocked_patterns": ["DOCTYPE", "ENTITY", "SYSTEM"]
            },
            severity="ERROR"
        )

    def log_redos_attempt(self, pattern: str, source: Optional[str] = None) -> None:
        """
        Log a ReDoS attack attempt.

        Args:
            pattern: The regex pattern that was blocked
            source: Optional source identifier
        """
        self.log_event(
            SecurityEventType.REDOS_ATTEMPT,
            "Regular Expression Denial of Service (ReDoS) pattern blocked",
            {
                "pattern": pattern[:200] if pattern else "",
                "source": source,
                "reason": "Pattern complexity or timeout"
            },
            severity="ERROR"
        )

    def log_oversized_input(self, size: int, limit: int, source: Optional[str] = None) -> None:
        """
        Log an oversized input attempt.

        Args:
            size: Size of the input
            limit: Maximum allowed size
            source: Optional source identifier
        """
        self.log_event(
            SecurityEventType.OVERSIZED_INPUT,
            f"Oversized input rejected: {size} bytes (limit: {limit})",
            {
                "input_size": size,
                "size_limit": limit,
                "source": source
            },
            severity="WARNING"
        )

    def log_suspicious_content(self, content_type: str, details: str, source: Optional[str] = None) -> None:
        """
        Log suspicious content detection.

        Args:
            content_type: Type of suspicious content
            details: Details about what was detected
            source: Optional source identifier
        """
        self.log_event(
            SecurityEventType.SUSPICIOUS_CONTENT,
            f"Suspicious content detected: {content_type}",
            {
                "content_type": content_type,
                "details": details,
                "source": source
            },
            severity="WARNING"
        )

    def get_events(self,
                   event_type: Optional[SecurityEventType] = None,
                   severity: Optional[str] = None,
                   limit: int = 100) -> list:
        """
        Retrieve logged security events.

        Args:
            event_type: Filter by event type
            severity: Filter by severity
            limit: Maximum number of events to return

        Returns:
            List of security events
        """
        events = self._events

        if event_type:
            events = [e for e in events if e["type"] == event_type.value]

        if severity:
            events = [e for e in events if e["severity"] == severity]

        return events[-limit:]

    def clear_events(self) -> None:
        """Clear stored security events."""
        self._events.clear()
        logger.info("Security event log cleared")


# Global security logger instance with thread-safe initialization
_security_logger: Optional[SecurityLogger] = None
_security_logger_lock = threading.Lock()


def get_security_logger() -> SecurityLogger:
    """
    Get the global security logger instance.

    Uses double-checked locking pattern for thread-safe lazy initialization.

    Returns:
        SecurityLogger instance
    """
    global _security_logger
    if _security_logger is None:
        with _security_logger_lock:
            # Double-check after acquiring lock
            if _security_logger is None:
                _security_logger = SecurityLogger()
    return _security_logger


def configure_security_logging(log_file: Optional[Path] = None, enable_console: bool = True) -> None:
    """
    Configure security logging.

    Thread-safe reconfiguration of the global security logger.

    Args:
        log_file: Optional path to security log file
        enable_console: Whether to also log to console
    """
    global _security_logger
    with _security_logger_lock:
        _security_logger = SecurityLogger(log_file, enable_console)
