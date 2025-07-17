"""
Atomic file operations utility module.

This module provides atomic file write operations to prevent data corruption
from partial writes due to crashes, power failures, or other interruptions.
"""

import os
import tempfile
from pathlib import Path
from typing import Union, Optional
import shutil
from loguru import logger


def atomic_write_text(
    file_path: Union[str, Path],
    content: str,
    encoding: str = 'utf-8',
    mode: int = 0o644
) -> None:
    """
    Write text content to a file atomically.
    
    This function writes content to a temporary file in the same directory as the
    target file, then atomically renames it to replace the target file. This ensures
    that the file is either fully written or not written at all.
    
    Args:
        file_path: Path to the target file
        content: Text content to write
        encoding: Text encoding (default: utf-8)
        mode: File permissions (default: 0o644)
        
    Raises:
        OSError: If the write or rename operation fails
        IOError: If the temporary file cannot be created
    """
    file_path = Path(file_path)
    parent_dir = file_path.parent
    
    # Ensure parent directory exists
    parent_dir.mkdir(parents=True, exist_ok=True)
    
    # Create temporary file in the same directory (for atomic rename)
    fd = None
    temp_path = None
    
    try:
        # Create temp file with secure permissions
        fd, temp_path = tempfile.mkstemp(
            dir=parent_dir,
            prefix=f".{file_path.name}.",
            suffix=".tmp",
            text=True
        )
        
        # Write content to temp file
        with os.fdopen(fd, 'w', encoding=encoding) as f:
            f.write(content)
            # Ensure data is written to disk
            f.flush()
            os.fsync(f.fileno())
        
        # Set file permissions
        os.chmod(temp_path, mode)
        
        # Atomic rename (on POSIX) or replace (cross-platform)
        # os.replace is atomic on POSIX and does best-effort on Windows
        os.replace(temp_path, str(file_path))
        
        logger.debug(f"Atomically wrote {len(content)} chars to {file_path}")
        
    except Exception as e:
        # Clean up temp file if it exists
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass
        logger.error(f"Failed to atomically write to {file_path}: {e}")
        raise


def atomic_write_bytes(
    file_path: Union[str, Path],
    content: bytes,
    mode: int = 0o644
) -> None:
    """
    Write binary content to a file atomically.
    
    This function writes content to a temporary file in the same directory as the
    target file, then atomically renames it to replace the target file.
    
    Args:
        file_path: Path to the target file
        content: Binary content to write
        mode: File permissions (default: 0o644)
        
    Raises:
        OSError: If the write or rename operation fails
        IOError: If the temporary file cannot be created
    """
    file_path = Path(file_path)
    parent_dir = file_path.parent
    
    # Ensure parent directory exists
    parent_dir.mkdir(parents=True, exist_ok=True)
    
    # Create temporary file in the same directory (for atomic rename)
    fd = None
    temp_path = None
    
    try:
        # Create temp file with secure permissions
        fd, temp_path = tempfile.mkstemp(
            dir=parent_dir,
            prefix=f".{file_path.name}.",
            suffix=".tmp",
            text=False
        )
        
        # Write content to temp file
        with os.fdopen(fd, 'wb') as f:
            f.write(content)
            # Ensure data is written to disk
            f.flush()
            os.fsync(f.fileno())
        
        # Set file permissions
        os.chmod(temp_path, mode)
        
        # Atomic rename (on POSIX) or replace (cross-platform)
        os.replace(temp_path, str(file_path))
        
        logger.debug(f"Atomically wrote {len(content)} bytes to {file_path}")
        
    except Exception as e:
        # Clean up temp file if it exists
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass
        logger.error(f"Failed to atomically write to {file_path}: {e}")
        raise


def atomic_write_json(
    file_path: Union[str, Path],
    data: dict,
    encoding: str = 'utf-8',
    mode: int = 0o644,
    indent: Optional[int] = 2
) -> None:
    """
    Write JSON data to a file atomically.
    
    Args:
        file_path: Path to the target file
        data: Dictionary to serialize as JSON
        encoding: Text encoding (default: utf-8)
        mode: File permissions (default: 0o644)
        indent: JSON indentation level (default: 2)
        
    Raises:
        OSError: If the write or rename operation fails
        json.JSONDecodeError: If the data cannot be serialized to JSON
    """
    import json
    
    # Serialize to JSON
    content = json.dumps(data, indent=indent, ensure_ascii=False)
    
    # Write atomically
    atomic_write_text(file_path, content, encoding=encoding, mode=mode)


def atomic_copy(
    src_path: Union[str, Path],
    dst_path: Union[str, Path],
    mode: Optional[int] = None
) -> None:
    """
    Copy a file atomically.
    
    This function copies the source file to a temporary file in the same directory
    as the destination, then atomically renames it to the destination path.
    
    Args:
        src_path: Path to the source file
        dst_path: Path to the destination file
        mode: File permissions for destination (default: preserve source permissions)
        
    Raises:
        OSError: If the copy or rename operation fails
        FileNotFoundError: If the source file doesn't exist
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    parent_dir = dst_path.parent
    
    if not src_path.exists():
        raise FileNotFoundError(f"Source file not found: {src_path}")
    
    # Ensure parent directory exists
    parent_dir.mkdir(parents=True, exist_ok=True)
    
    # Create temporary file in the same directory (for atomic rename)
    fd = None
    temp_path = None
    
    try:
        # Create temp file
        fd, temp_path = tempfile.mkstemp(
            dir=parent_dir,
            prefix=f".{dst_path.name}.",
            suffix=".tmp"
        )
        os.close(fd)  # Close the file descriptor, we'll use shutil.copy2
        
        # Copy file with metadata
        shutil.copy2(str(src_path), temp_path)
        
        # Set permissions if specified
        if mode is not None:
            os.chmod(temp_path, mode)
        
        # Atomic rename
        os.replace(temp_path, str(dst_path))
        
        logger.debug(f"Atomically copied {src_path} to {dst_path}")
        
    except Exception as e:
        # Clean up temp file if it exists
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass
        logger.error(f"Failed to atomically copy {src_path} to {dst_path}: {e}")
        raise


# Backward compatibility aliases
write_text_atomic = atomic_write_text
write_bytes_atomic = atomic_write_bytes
write_json_atomic = atomic_write_json
copy_atomic = atomic_copy