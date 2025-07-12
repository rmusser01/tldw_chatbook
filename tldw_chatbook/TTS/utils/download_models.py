# download_models.py
# Description: Utilities for downloading TTS model files
#
# Imports
import os
import asyncio
from pathlib import Path
from typing import Optional, Dict, Tuple
import httpx
from loguru import logger

#######################################################################################################################
#
# Model Download Utilities

class ModelDownloader:
    """Handles downloading of TTS model files"""
    
    # Known model URLs
    KOKORO_MODELS = {
        "v0.19": {
            "model": "https://huggingface.co/hexgrad/kLegacy/resolve/main/v0.19/kokoro-v0_19.onnx?download=true",
            "voices": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/v0.1.0/voices.json",
            "size": {
                "model": 311_000_000,  # ~311MB
                "voices": 10_000,      # ~10KB
            }
        },
        "v0.23": {
            "model": "https://huggingface.co/hexgrad/Kokoro-82M-ONNX/resolve/main/kokoro-v0_23-int8.onnx?download=true",
            "voices": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/v0.1.0/voices.json",
            "size": {
                "model": 350_000_000,  # ~350MB
                "voices": 10_000,      # ~10KB
            }
        }
    }
    
    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize downloader.
        
        Args:
            base_dir: Base directory for model storage. Defaults to ~/.config/tldw_cli/models
        """
        if base_dir is None:
            base_dir = Path.home() / ".config" / "tldw_cli" / "models"
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    async def download_file(
        self, 
        url: str, 
        destination: Path, 
        expected_size: Optional[int] = None,
        progress_callback: Optional[callable] = None
    ) -> bool:
        """
        Download a file with progress tracking.
        
        Args:
            url: URL to download from
            destination: Path to save file
            expected_size: Expected file size for validation
            progress_callback: Callback for progress updates (bytes_downloaded, total_bytes)
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Downloading {url} to {destination}")
        
        # Create parent directory if needed
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if file already exists with correct size
        if destination.exists() and expected_size:
            actual_size = destination.stat().st_size
            if actual_size == expected_size:
                logger.info(f"File {destination} already exists with correct size")
                return True
            else:
                logger.warning(f"File {destination} exists but size mismatch "
                             f"(expected: {expected_size}, actual: {actual_size})")
        
        async with httpx.AsyncClient(follow_redirects=True) as client:
            try:
                # Stream download
                async with client.stream("GET", url) as response:
                    response.raise_for_status()
                    
                    # Get total size
                    total_size = int(response.headers.get("content-length", 0))
                    if total_size == 0 and expected_size:
                        total_size = expected_size
                    
                    # Download to temporary file first
                    temp_file = destination.with_suffix(destination.suffix + ".tmp")
                    bytes_downloaded = 0
                    
                    with open(temp_file, "wb") as f:
                        async for chunk in response.aiter_bytes(chunk_size=8192):
                            f.write(chunk)
                            bytes_downloaded += len(chunk)
                            
                            # Progress callback
                            if progress_callback:
                                progress_callback(bytes_downloaded, total_size)
                    
                    # Verify size if expected
                    if expected_size and bytes_downloaded != expected_size:
                        logger.error(f"Downloaded size mismatch: {bytes_downloaded} != {expected_size}")
                        temp_file.unlink(missing_ok=True)
                        return False
                    
                    # Move to final destination
                    temp_file.rename(destination)
                    logger.info(f"Successfully downloaded {destination}")
                    return True
                    
            except Exception as e:
                logger.error(f"Download failed: {e}")
                # Clean up temp file
                temp_file = destination.with_suffix(destination.suffix + ".tmp")
                temp_file.unlink(missing_ok=True)
                return False
    
    async def download_kokoro_model(
        self, 
        version: str = "v0.19",
        progress_callback: Optional[callable] = None
    ) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Download Kokoro model files.
        
        Args:
            version: Model version to download
            progress_callback: Progress callback function
            
        Returns:
            Tuple of (model_path, voices_path) or (None, None) on failure
        """
        if version not in self.KOKORO_MODELS:
            logger.error(f"Unknown Kokoro model version: {version}")
            return None, None
        
        model_info = self.KOKORO_MODELS[version]
        kokoro_dir = self.base_dir / "kokoro"
        kokoro_dir.mkdir(exist_ok=True)
        
        model_path = kokoro_dir / f"kokoro-{version}.onnx"
        voices_path = kokoro_dir / "voices.json"
        
        # Download model
        model_success = await self.download_file(
            model_info["model"],
            model_path,
            model_info["size"]["model"],
            progress_callback
        )
        
        if not model_success:
            return None, None
        
        # Download voices
        voices_success = await self.download_file(
            model_info["voices"],
            voices_path,
            model_info["size"]["voices"],
            progress_callback
        )
        
        if not voices_success:
            return None, None
        
        return model_path, voices_path
    
    def get_model_paths(self, model_name: str = "kokoro", version: str = "v0.19") -> Dict[str, Path]:
        """
        Get paths for model files.
        
        Args:
            model_name: Name of the model
            version: Model version
            
        Returns:
            Dictionary of file paths
        """
        if model_name == "kokoro":
            kokoro_dir = self.base_dir / "kokoro"
            return {
                "model": kokoro_dir / f"kokoro-{version}.onnx",
                "voices": kokoro_dir / "voices.json"
            }
        
        return {}


# Convenience functions
async def download_kokoro_model(
    version: str = "v0.19",
    base_dir: Optional[Path] = None,
    progress_callback: Optional[callable] = None
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Download Kokoro model files.
    
    Args:
        version: Model version to download
        base_dir: Base directory for models
        progress_callback: Progress callback
        
    Returns:
        Tuple of (model_path, voices_path)
    """
    downloader = ModelDownloader(base_dir)
    return await downloader.download_kokoro_model(version, progress_callback)


def get_kokoro_model_paths(
    version: str = "v0.19",
    base_dir: Optional[Path] = None
) -> Dict[str, Path]:
    """
    Get paths for Kokoro model files.
    
    Args:
        version: Model version
        base_dir: Base directory for models
        
    Returns:
        Dictionary with 'model' and 'voices' paths
    """
    downloader = ModelDownloader(base_dir)
    return downloader.get_model_paths("kokoro", version)


# CLI interface for testing
if __name__ == "__main__":
    async def main():
        def progress(downloaded, total):
            if total > 0:
                percent = (downloaded / total) * 100
                print(f"Progress: {downloaded:,} / {total:,} bytes ({percent:.1f}%)", end='\r')
        
        print("Downloading Kokoro model files...")
        model_path, voices_path = await download_kokoro_model(progress_callback=progress)
        
        if model_path and voices_path:
            print(f"\nModel downloaded to: {model_path}")
            print(f"Voices downloaded to: {voices_path}")
        else:
            print("\nDownload failed!")
    
    asyncio.run(main())

#
# End of download_models.py
#######################################################################################################################