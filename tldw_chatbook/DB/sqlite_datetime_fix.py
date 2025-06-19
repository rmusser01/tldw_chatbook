"""
SQLite datetime adapter fix for Python 3.12+

This module provides proper datetime adapters to avoid deprecation warnings
in Python 3.12 and later versions.
"""

import sqlite3
from datetime import datetime, timezone


def adapt_datetime(dt):
    """Adapt datetime to ISO 8601 string for SQLite storage."""
    if dt is None:
        return None
    # Ensure we have timezone info (use UTC if naive)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    # Return ISO format string
    return dt.isoformat()


def convert_datetime(s):
    """Convert ISO 8601 string back to datetime object."""
    if s is None or s == '':
        return None
    if isinstance(s, bytes):
        s = s.decode('utf-8')
    # Parse ISO format
    return datetime.fromisoformat(s)


def setup_datetime_adapters():
    """Register datetime adapters and converters for SQLite."""
    # Register adapter for datetime -> string
    sqlite3.register_adapter(datetime, adapt_datetime)
    
    # Register converter for string -> datetime
    sqlite3.register_converter("TIMESTAMP", convert_datetime)
    sqlite3.register_converter("DATETIME", convert_datetime)
    
    # Also handle date and time if needed
    from datetime import date, time
    
    def adapt_date(d):
        if d is None:
            return None
        return d.isoformat()
    
    def adapt_time(t):
        if t is None:
            return None
        return t.isoformat()
    
    def convert_date(s):
        if s is None or s == '':
            return None
        if isinstance(s, bytes):
            s = s.decode('utf-8')
        return date.fromisoformat(s)
    
    def convert_time(s):
        if s is None or s == '':
            return None
        if isinstance(s, bytes):
            s = s.decode('utf-8')
        return time.fromisoformat(s)
    
    sqlite3.register_adapter(date, adapt_date)
    sqlite3.register_adapter(time, adapt_time)
    sqlite3.register_converter("DATE", convert_date)
    sqlite3.register_converter("TIME", convert_time)


# Auto-register on import
setup_datetime_adapters()