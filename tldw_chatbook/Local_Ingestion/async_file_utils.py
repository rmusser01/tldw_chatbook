# async_file_utils.py
"""
Async file utilities for efficient file processing.
Provides streaming, encoding detection, and async file operations.
"""

import asyncio
from pathlib import Path
from typing import AsyncIterator, Optional, Union, List, Tuple
import chardet
from loguru import logger

try:
    import aiofiles
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False
    logger.warning("aiofiles not available. Async file operations will fall back to sync mode.")


async def detect_encoding_async(
    file_path: Path, 
    sample_size: int = 10000,
    timeout: float = 5.0
) -> str:
    """
    Asynchronously detect file encoding with timeout support.
    
    Args:
        file_path: Path to the file
        sample_size: Number of bytes to sample for detection
        timeout: Maximum time to wait for detection
        
    Returns:
        Detected encoding string
    """
    try:
        if AIOFILES_AVAILABLE:
            async with aiofiles.open(file_path, 'rb') as f:
                sample = await asyncio.wait_for(f.read(sample_size), timeout=timeout)
        else:
            # Fallback to sync
            with open(file_path, 'rb') as f:
                sample = f.read(sample_size)
        
        detected = chardet.detect(sample)
        encoding = detected['encoding'] or 'utf-8'
        confidence = detected['confidence'] or 0.0
        
        logger.debug(f"Detected encoding for {file_path.name}: {encoding} (confidence: {confidence:.2f})")
        
        if confidence < 0.7:
            logger.warning(f"Low confidence encoding detection for {file_path.name}")
            # Try multiple encodings
            for enc in ['utf-8', 'latin-1', 'cp1252', 'utf-16']:
                try:
                    # Test read with encoding
                    if AIOFILES_AVAILABLE:
                        async with aiofiles.open(file_path, 'r', encoding=enc) as f:
                            await f.read(100)
                    else:
                        with open(file_path, 'r', encoding=enc) as f:
                            f.read(100)
                    logger.debug(f"Successfully tested encoding {enc} for {file_path.name}")
                    return enc
                except (UnicodeDecodeError, UnicodeError):
                    continue
        
        return encoding
        
    except asyncio.TimeoutError:
        logger.warning(f"Encoding detection timed out for {file_path.name}, using utf-8")
        return 'utf-8'
    except Exception as e:
        logger.error(f"Error detecting encoding for {file_path.name}: {e}")
        return 'utf-8'


async def stream_file_content(
    file_path: Path, 
    chunk_size: int = 8192,
    encoding: Optional[str] = None,
    timeout: float = 300.0  # 5 minutes default
) -> AsyncIterator[str]:
    """
    Stream file content in chunks for memory-efficient processing.
    
    Args:
        file_path: Path to the file
        chunk_size: Size of each chunk in bytes
        encoding: File encoding (auto-detected if None)
        timeout: Maximum time for the entire operation
        
    Yields:
        String chunks of the file content
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if not encoding:
        encoding = await detect_encoding_async(file_path)
    
    try:
        if AIOFILES_AVAILABLE:
            async with aiofiles.open(file_path, 'r', encoding=encoding, errors='replace') as f:
                while True:
                    chunk = await asyncio.wait_for(f.read(chunk_size), timeout=timeout)
                    if not chunk:
                        break
                    yield chunk
        else:
            # Fallback to sync with async wrapper
            def _read_sync():
                with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        yield chunk
            
            for chunk in _read_sync():
                yield chunk
                await asyncio.sleep(0)  # Allow other tasks to run
                
    except asyncio.TimeoutError:
        logger.error(f"Streaming timed out for {file_path.name}")
        raise
    except Exception as e:
        logger.error(f"Error streaming file {file_path.name}: {e}")
        raise


async def read_file_async(
    file_path: Path, 
    encoding: Optional[str] = None,
    timeout: float = 60.0
) -> str:
    """
    Read entire file asynchronously with timeout support.
    
    Args:
        file_path: Path to the file
        encoding: File encoding (auto-detected if None)
        timeout: Maximum time for reading
        
    Returns:
        File content as string
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if not encoding:
        encoding = await detect_encoding_async(file_path)
    
    try:
        if AIOFILES_AVAILABLE:
            async with aiofiles.open(file_path, 'r', encoding=encoding, errors='replace') as f:
                content = await asyncio.wait_for(f.read(), timeout=timeout)
        else:
            # Fallback to sync
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                content = f.read()
        
        logger.debug(f"Successfully read {len(content)} characters from {file_path.name}")
        return content
        
    except asyncio.TimeoutError:
        logger.error(f"Reading timed out for {file_path.name}")
        raise
    except Exception as e:
        logger.error(f"Error reading file {file_path.name}: {e}")
        raise


async def write_file_async(
    file_path: Path,
    content: str,
    encoding: str = 'utf-8',
    create_parents: bool = True
) -> None:
    """
    Write content to file asynchronously.
    
    Args:
        file_path: Path to write to
        content: Content to write
        encoding: File encoding
        create_parents: Create parent directories if they don't exist
    """
    if create_parents:
        file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if AIOFILES_AVAILABLE:
            async with aiofiles.open(file_path, 'w', encoding=encoding) as f:
                await f.write(content)
        else:
            # Fallback to sync
            with open(file_path, 'w', encoding=encoding) as f:
                f.write(content)
        
        logger.debug(f"Successfully wrote {len(content)} characters to {file_path.name}")
        
    except Exception as e:
        logger.error(f"Error writing file {file_path.name}: {e}")
        raise


async def process_files_concurrently(
    file_paths: List[Path],
    process_func,
    max_concurrent: int = 4,
    **kwargs
) -> List[Tuple[Path, Union[str, Exception]]]:
    """
    Process multiple files concurrently with controlled parallelism.
    
    Args:
        file_paths: List of file paths to process
        process_func: Async function to process each file
        max_concurrent: Maximum concurrent operations
        **kwargs: Additional arguments for process_func
        
    Returns:
        List of (path, result) tuples where result is either processed content or exception
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(file_path: Path):
        async with semaphore:
            try:
                result = await process_func(file_path, **kwargs)
                return (file_path, result)
            except Exception as e:
                logger.error(f"Error processing {file_path.name}: {e}")
                return (file_path, e)
    
    tasks = [process_with_semaphore(fp) for fp in file_paths]
    results = await asyncio.gather(*tasks)
    
    return results


def get_large_file_threshold() -> int:
    """Get threshold for considering a file 'large' (in bytes)."""
    return 10 * 1024 * 1024  # 10MB


async def is_large_file(file_path: Path) -> bool:
    """Check if file should be processed as a large file."""
    try:
        return file_path.stat().st_size > get_large_file_threshold()
    except Exception:
        return False


# Convenience function for sync contexts
def read_file_with_fallback(file_path: Path) -> str:
    """
    Synchronous file reading with encoding detection and fallback.
    For use when async is not available.
    """
    try:
        # Try UTF-8 first
        return file_path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        # Try with chardet
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)
        detected = chardet.detect(raw_data)
        encoding = detected['encoding'] or 'latin-1'
        
        try:
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            # Last resort: ignore errors
            return file_path.read_text(encoding='utf-8', errors='ignore')