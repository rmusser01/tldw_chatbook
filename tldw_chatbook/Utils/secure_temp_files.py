"""
Secure temporary file utilities for safe file handling.
"""
import os
import tempfile
import stat
import logging
from pathlib import Path
from typing import Optional, Union
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@contextmanager
def secure_temp_file(suffix: str = '', prefix: str = 'tmp', dir: Optional[str] = None, 
                    text: bool = True, mode: str = 'w+'):
    """
    Context manager for creating secure temporary files.
    
    Args:
        suffix: File extension/suffix
        prefix: File name prefix  
        dir: Directory to create file in (defaults to secure temp dir)
        text: Whether to open in text mode
        mode: File open mode
        
    Yields:
        Temporary file object
    """
    temp_file = None
    try:
        # Create temporary file with secure permissions
        temp_file = tempfile.NamedTemporaryFile(
            mode=mode,
            suffix=suffix,
            prefix=prefix,
            dir=dir,
            delete=False  # We'll handle deletion ourselves
        )
        
        # Set secure permissions (read/write for owner only)
        os.chmod(temp_file.name, stat.S_IRUSR | stat.S_IWUSR)
        
        yield temp_file
        
    finally:
        if temp_file:
            try:
                temp_file.close()
                # Securely delete the temporary file
                if os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)
                    logger.debug(f"Securely deleted temporary file: {temp_file.name}")
            except Exception as e:
                logger.error(f"Error cleaning up temporary file {temp_file.name}: {e}")


@contextmanager
def secure_temp_dir(suffix: str = '', prefix: str = 'tmp', dir: Optional[str] = None):
    """
    Context manager for creating secure temporary directories.
    
    Args:
        suffix: Directory name suffix
        prefix: Directory name prefix
        dir: Parent directory to create temp dir in
        
    Yields:
        Path to temporary directory
    """
    temp_dir = None
    try:
        # Create temporary directory with secure permissions
        temp_dir = tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=dir)
        
        # Set secure permissions (read/write/execute for owner only)
        os.chmod(temp_dir, stat.S_IRWXU)
        
        yield temp_dir
        
    finally:
        if temp_dir and os.path.exists(temp_dir):
            try:
                # Recursively remove the temporary directory and all contents
                import shutil
                shutil.rmtree(temp_dir)
                logger.debug(f"Securely deleted temporary directory: {temp_dir}")
            except Exception as e:
                logger.error(f"Error cleaning up temporary directory {temp_dir}: {e}")


def create_secure_temp_file(content: Union[str, bytes], suffix: str = '', 
                           prefix: str = 'tmp', dir: Optional[str] = None) -> str:
    """
    Create a secure temporary file with given content and return its path.
    
    Args:
        content: Content to write to the file
        suffix: File extension/suffix
        prefix: File name prefix
        dir: Directory to create file in
        
    Returns:
        Path to the created temporary file
        
    Note:
        Caller is responsible for deleting the file when done.
    """
    mode = 'w' if isinstance(content, str) else 'wb'
    
    temp_file = tempfile.NamedTemporaryFile(
        mode=mode,
        suffix=suffix,
        prefix=prefix,
        dir=dir,
        delete=False
    )
    
    try:
        # Set secure permissions
        os.chmod(temp_file.name, stat.S_IRUSR | stat.S_IWUSR)
        
        # Write content
        temp_file.write(content)
        temp_file.flush()
        
        return temp_file.name
        
    finally:
        temp_file.close()


def secure_delete_file(file_path: Union[str, Path]) -> bool:
    """
    Securely delete a file.
    
    Args:
        file_path: Path to file to delete
        
    Returns:
        True if deletion was successful, False otherwise
    """
    try:
        file_path = Path(file_path)
        if file_path.exists():
            # Overwrite file content with zeros before deletion for extra security
            if file_path.is_file():
                file_size = file_path.stat().st_size
                with open(file_path, "r+b") as f:
                    f.write(b'\x00' * file_size)
                    f.flush()
                    os.fsync(f.fileno())
            
            # Delete the file
            file_path.unlink()
            logger.debug(f"Securely deleted file: {file_path}")
            return True
    except Exception as e:
        logger.error(f"Error securely deleting file {file_path}: {e}")
        return False


class SecureTempFileManager:
    """
    Manager for tracking and cleaning up temporary files.
    """
    
    def __init__(self):
        self._temp_files = set()
        self._temp_dirs = set()
    
    def create_temp_file(self, content: Union[str, bytes], suffix: str = '', 
                        prefix: str = 'tmp', dir: Optional[str] = None) -> str:
        """Create a temporary file and track it for cleanup."""
        temp_path = create_secure_temp_file(content, suffix, prefix, dir)
        self._temp_files.add(temp_path)
        return temp_path
    
    def create_temp_dir(self, suffix: str = '', prefix: str = 'tmp', 
                       dir: Optional[str] = None) -> str:
        """Create a temporary directory and track it for cleanup."""
        temp_dir = tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=dir)
        os.chmod(temp_dir, stat.S_IRWXU)
        self._temp_dirs.add(temp_dir)
        return temp_dir
    
    def cleanup_all(self):
        """Clean up all tracked temporary files and directories."""
        # Clean up files first
        for temp_file in list(self._temp_files):
            if secure_delete_file(temp_file):
                self._temp_files.discard(temp_file)
        
        # Clean up directories
        for temp_dir in list(self._temp_dirs):
            try:
                if os.path.exists(temp_dir):
                    import shutil
                    shutil.rmtree(temp_dir)
                    logger.debug(f"Cleaned up temporary directory: {temp_dir}")
                self._temp_dirs.discard(temp_dir)
            except Exception as e:
                logger.error(f"Error cleaning up temporary directory {temp_dir}: {e}")
    
    def __del__(self):
        """Cleanup on destruction."""
        self.cleanup_all()


# Global temp file manager instance
_global_temp_manager = SecureTempFileManager()


def get_temp_manager() -> SecureTempFileManager:
    """Get the global temporary file manager."""
    return _global_temp_manager


def cleanup_all_temp_files():
    """Clean up all temporary files created by the global manager."""
    _global_temp_manager.cleanup_all()