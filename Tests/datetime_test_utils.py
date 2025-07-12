"""
DateTime Test Utilities
=====================

This module provides consistent datetime handling utilities for the test suite.
It addresses the common issue where SQLite returns datetime objects when
PARSE_DECLTYPES is used, but some tests expect string representations.

Usage:
    from Tests.datetime_test_utils import (
        normalize_datetime,
        assert_datetime_equal,
        datetime_to_iso_string,
        parse_datetime_flexible
    )
"""

from datetime import datetime
from typing import Union, Optional
import re


def normalize_datetime(dt: Union[str, datetime, None]) -> Optional[str]:
    """
    Normalize a datetime value to ISO format string.
    
    Args:
        dt: DateTime object, ISO string, or None
        
    Returns:
        ISO format string or None
    """
    if dt is None:
        return None
    
    if isinstance(dt, datetime):
        return dt.isoformat()
    
    if isinstance(dt, str):
        # Already a string, ensure it's in ISO format
        try:
            # Parse and re-format to ensure consistency
            parsed = parse_datetime_flexible(dt)
            return parsed.isoformat() if parsed else dt
        except:
            return dt
    
    return str(dt)


def parse_datetime_flexible(dt_str: str) -> Optional[datetime]:
    """
    Parse datetime string flexibly, handling various formats.
    
    Args:
        dt_str: DateTime string in various formats
        
    Returns:
        datetime object or None if parsing fails
    """
    if not dt_str:
        return None
    
    # Common datetime formats to try
    formats = [
        "%Y-%m-%d %H:%M:%S.%f",  # SQLite with microseconds
        "%Y-%m-%d %H:%M:%S",      # SQLite without microseconds
        "%Y-%m-%dT%H:%M:%S.%f",   # ISO with microseconds
        "%Y-%m-%dT%H:%M:%S",      # ISO without microseconds
        "%Y-%m-%dT%H:%M:%S.%fZ",  # ISO with Z suffix
        "%Y-%m-%dT%H:%M:%SZ",     # ISO with Z suffix, no microseconds
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(dt_str, fmt)
        except ValueError:
            continue
    
    # Try parsing with fromisoformat as fallback
    try:
        # Remove 'Z' suffix if present
        cleaned = dt_str.rstrip('Z')
        return datetime.fromisoformat(cleaned)
    except:
        return None


def assert_datetime_equal(
    actual: Union[str, datetime, None],
    expected: Union[str, datetime, None],
    msg: Optional[str] = None
):
    """
    Assert that two datetime values are equal, handling type differences.
    
    Args:
        actual: Actual datetime value
        expected: Expected datetime value
        msg: Optional assertion message
    """
    actual_normalized = normalize_datetime(actual)
    expected_normalized = normalize_datetime(expected)
    
    if actual_normalized != expected_normalized:
        error_msg = f"DateTime mismatch: {actual_normalized} != {expected_normalized}"
        if msg:
            error_msg = f"{msg}: {error_msg}"
        raise AssertionError(error_msg)


def datetime_to_iso_string(dt: Union[str, datetime, None]) -> Optional[str]:
    """
    Convert datetime to ISO format string for JSON serialization.
    
    Args:
        dt: DateTime object or string
        
    Returns:
        ISO format string or None
    """
    return normalize_datetime(dt)


def compare_datetime_dicts(actual_dict: dict, expected_dict: dict, datetime_fields: list):
    """
    Compare two dictionaries, with special handling for datetime fields.
    
    Args:
        actual_dict: Dictionary with actual values
        expected_dict: Dictionary with expected values
        datetime_fields: List of field names that contain datetime values
        
    Raises:
        AssertionError if dictionaries don't match
    """
    # Compare all keys
    assert set(actual_dict.keys()) == set(expected_dict.keys()), \
        f"Dictionary keys mismatch: {set(actual_dict.keys())} != {set(expected_dict.keys())}"
    
    # Compare values
    for key in actual_dict:
        if key in datetime_fields:
            assert_datetime_equal(
                actual_dict[key],
                expected_dict[key],
                msg=f"Field '{key}'"
            )
        else:
            assert actual_dict[key] == expected_dict[key], \
                f"Field '{key}' mismatch: {actual_dict[key]} != {expected_dict[key]}"


def is_recent_datetime(dt: Union[str, datetime], max_seconds_ago: int = 60) -> bool:
    """
    Check if a datetime is recent (within specified seconds from now).
    
    Args:
        dt: DateTime to check
        max_seconds_ago: Maximum seconds in the past to consider "recent"
        
    Returns:
        True if datetime is recent, False otherwise
    """
    if isinstance(dt, str):
        dt_obj = parse_datetime_flexible(dt)
        if not dt_obj:
            return False
    else:
        dt_obj = dt
    
    now = datetime.now()
    diff = (now - dt_obj).total_seconds()
    
    return 0 <= diff <= max_seconds_ago


def extract_datetime_from_json_log(log_entry: str) -> Optional[datetime]:
    """
    Extract datetime from a JSON log entry that might contain datetime strings.
    
    Args:
        log_entry: JSON string that might contain datetime values
        
    Returns:
        First datetime found or None
    """
    # Look for ISO datetime patterns in the string
    iso_pattern = r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z)?'
    matches = re.findall(iso_pattern, log_entry)
    
    for match in matches:
        dt = parse_datetime_flexible(match)
        if dt:
            return dt
    
    return None