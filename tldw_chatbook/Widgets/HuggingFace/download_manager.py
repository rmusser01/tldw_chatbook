# tldw_chatbook/Widgets/HuggingFace/download_manager.py
"""
Download manager for HuggingFace model files with progress tracking.
"""

import asyncio
import queue
import threading
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.widgets import Label, Button, ProgressBar, Static, ListView, ListItem
from textual.reactive import reactive
from textual import work
from loguru import logger


class DownloadItem:
    """Represents a single download."""
    
    def __init__(self, repo_id: str, filename: str, size: int, destination: Path):
        self.repo_id = repo_id
        self.filename = filename
        self.size = size
        self.destination = destination
        self.downloaded = 0
        self.status = "pending"  # pending, downloading, completed, error
        self.error_message = None
        self.start_time = None
        self.end_time = None
    
    @property
    def progress(self) -> float:
        """Get download progress as percentage."""
        if self.size == 0:
            return 0.0
        return (self.downloaded / self.size) * 100
    
    @property
    def speed(self) -> Optional[float]:
        """Get download speed in bytes per second."""
        if not self.start_time or self.status != "downloading":
            return None
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if elapsed == 0:
            return None
        return self.downloaded / elapsed


class DownloadManager(Container):
    """Widget for managing model downloads with progress tracking."""
    
    DEFAULT_CSS = """
    DownloadManager {
        height: 1fr;
        layout: vertical;
        background: $surface;
        border: solid $primary;
    }
    
    DownloadManager .header {
        height: 3;
        padding: 1;
        background: $boost;
        border-bottom: solid $primary;
    }
    
    DownloadManager .header-title {
        text-style: bold;
    }
    
    DownloadManager .downloads-list {
        height: 1fr;
        overflow-y: auto;
        background: $background;
        border: solid $primary-background-darken-1;
    }
    
    DownloadManager .download-item {
        padding: 1;
        margin: 1;
        background: $background;
        border: solid $primary-background-darken-1;
    }
    
    DownloadManager .download-header {
        layout: horizontal;
        height: auto;
        margin-bottom: 1;
    }
    
    DownloadManager .download-title {
        width: 1fr;
        text-style: bold;
    }
    
    DownloadManager .download-status {
        width: auto;
        margin-left: 1;
    }
    
    DownloadManager .download-status.pending {
        color: $text-muted;
    }
    
    DownloadManager .download-status.downloading {
        color: $primary;
    }
    
    DownloadManager .download-status.completed {
        color: $success;
    }
    
    DownloadManager .download-status.error {
        color: $error;
    }
    
    DownloadManager .download-progress {
        height: 1;
        margin: 1 0;
    }
    
    DownloadManager .download-info {
        color: $text-muted;
        height: 1;
    }
    
    DownloadManager .download-actions {
        layout: horizontal;
        height: 3;
        margin-top: 1;
    }
    
    DownloadManager .download-actions Button {
        margin-right: 1;
    }
    
    DownloadManager .summary-bar {
        height: 3;
        padding: 1;
        background: $boost;
        border-top: solid $primary;
    }
    """
    
    # Reactive properties
    downloads: reactive[Dict[str, DownloadItem]] = reactive({})
    active_downloads: reactive[int] = reactive(0)
    
    MAX_CONCURRENT_DOWNLOADS = 2
    
    def __init__(self, download_dir: Optional[Path] = None, **kwargs):
        """
        Initialize download manager.
        
        Args:
            download_dir: Directory to save downloads (default: ~/Downloads/tldw_models)
        """
        super().__init__(**kwargs)
        self.download_dir = download_dir or Path.home() / "Downloads" / "tldw_models"
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.download_queue = queue.Queue()
        self.download_tasks = []
        self._shutdown = threading.Event()
    
    def compose(self) -> ComposeResult:
        """Compose the download manager UI."""
        # Header
        with Container(classes="header"):
            yield Label("Download Manager", classes="header-title")
        
        # Downloads list
        yield ListView(id="downloads-list", classes="downloads-list")
        
        # Summary bar
        with Container(classes="summary-bar"):
            yield Static("", id="summary-text")
    
    def on_mount(self) -> None:
        """Start download workers on mount."""
        logger.info("DownloadManager mounted, starting download workers")
        
        # Start download workers using run_worker with thread=True
        for i in range(self.MAX_CONCURRENT_DOWNLOADS):
            logger.info(f"Starting download worker {i}")
            # Use partial to bind the worker_id
            from functools import partial
            worker_func = partial(self._download_worker_thread, worker_id=i)
            self.run_worker(worker_func, thread=True, name=f"download_worker_{i}")
            logger.info(f"Started download worker {i}")
    
    def watch_downloads(self, downloads: Dict[str, DownloadItem]) -> None:
        """Update UI when downloads change."""
        self.call_later(self._refresh_downloads_list)
        self._update_summary()
    
    def watch_active_downloads(self, count: int) -> None:
        """Update summary when active downloads change."""
        self._update_summary()
    
    async def _refresh_downloads_list(self) -> None:
        """Refresh the downloads list display."""
        downloads_list = self.query_one("#downloads-list", ListView)
        await downloads_list.clear()
        
        for download_id, download in self.downloads.items():
            item = self._create_download_item(download_id, download)
            await downloads_list.append(item)
    
    def _create_download_item(self, download_id: str, download: DownloadItem) -> ListItem:
        """Create a list item for a download."""
        # Build widgets list
        widgets = []
        
        # Header with title and status
        header_widgets = [
            Static(
                f"{download.repo_id}/{download.filename}",
                classes="download-title"
            ),
            Static(
                download.status.upper(),
                classes=f"download-status {download.status}"
            )
        ]
        header = Horizontal(*header_widgets, classes="download-header")
        widgets.append(header)
        
        # Progress bar
        if download.status == "downloading":
            progress_bar = ProgressBar(
                total=100,
                show_eta=True,
                show_percentage=True,
                classes="download-progress"
            )
            progress_bar.update(progress=download.progress)
            widgets.append(progress_bar)
        
        # Info line
        info_text = self._get_download_info(download)
        widgets.append(Static(info_text, classes="download-info"))
        
        # Actions
        if download.status in ["error", "completed"]:
            action_widgets = []
            
            if download.status == "error":
                retry_btn = Button("Retry", variant="primary", classes="retry-button")
                retry_btn.download_id = download_id
                action_widgets.append(retry_btn)
            
            remove_btn = Button("Remove", variant="error", classes="remove-button")
            remove_btn.download_id = download_id
            action_widgets.append(remove_btn)
            
            if action_widgets:
                actions = Horizontal(*action_widgets, classes="download-actions")
                widgets.append(actions)
        
        # Create container with all widgets
        container = Vertical(*widgets, classes="download-item")
        
        item = ListItem(container)
        item.download_id = download_id
        return item
    
    def _get_download_info(self, download: DownloadItem) -> str:
        """Get info text for a download."""
        size_str = self._format_bytes(download.size)
        
        if download.status == "pending":
            return f"Size: {size_str} • Waiting to start..."
        elif download.status == "downloading":
            downloaded_str = self._format_bytes(download.downloaded)
            percent = download.progress
            speed = download.speed
            if speed:
                speed_str = self._format_bytes(speed) + "/s"
                eta = (download.size - download.downloaded) / speed
                eta_str = self._format_time(eta)
                return f"{downloaded_str} / {size_str} ({percent:.1f}%) • {speed_str} • ETA: {eta_str}"
            else:
                return f"{downloaded_str} / {size_str} ({percent:.1f}%)"
        elif download.status == "completed":
            return f"Size: {size_str} • Completed"
        elif download.status == "error":
            return f"Error: {download.error_message or 'Unknown error'}"
        else:
            return ""
    
    def _update_summary(self) -> None:
        """Update the summary text."""
        total = len(self.downloads)
        active = self.active_downloads
        completed = sum(1 for d in self.downloads.values() if d.status == "completed")
        errors = sum(1 for d in self.downloads.values() if d.status == "error")
        
        summary = self.query_one("#summary-text", Static)
        parts = []
        
        if total > 0:
            parts.append(f"Total: {total}")
        if active > 0:
            parts.append(f"Active: {active}")
        if completed > 0:
            parts.append(f"Completed: {completed}")
        if errors > 0:
            parts.append(f"Errors: {errors}")
        
        summary.update(" • ".join(parts) if parts else "No downloads")
    
    def add_download(self, repo_id: str, file_info: Dict[str, Any]) -> str:
        """
        Add a file to download queue.
        
        Args:
            repo_id: Repository ID
            file_info: File information from API
            
        Returns:
            Download ID
        """
        logger.info(f"Adding download for {repo_id}/{file_info.get('path', 'unknown')}")
        
        filename = Path(file_info["path"]).name
        size = file_info.get("size", 0)
        
        # Create destination path
        model_dir = self.download_dir / repo_id.replace("/", "_")
        model_dir.mkdir(parents=True, exist_ok=True)
        destination = model_dir / filename
        
        # Create download item
        download = DownloadItem(repo_id, filename, size, destination)
        download_id = f"{repo_id}_{filename}"
        
        # Add to downloads
        self.downloads = {**self.downloads, download_id: download}
        logger.info(f"Added download item to downloads dict: {download_id}")
        
        # Queue for processing
        if self.download_queue is not None:
            try:
                self.download_queue.put((download_id, file_info))
                queue_size = self.download_queue.qsize()
                logger.info(f"Successfully queued download task: {download_id}, queue size after put: {queue_size}")
            except Exception as e:
                logger.error(f"Failed to queue download: {e}")
        else:
            logger.error("Download queue is None!")
        
        return download_id
    
    def _download_worker_thread(self, worker_id: int) -> None:
        """Worker thread for processing downloads."""
        logger.info(f"Download worker {worker_id} started in thread")
        
        # Run the async worker in a new event loop
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self._download_worker(worker_id))
        except Exception as e:
            logger.error(f"Worker {worker_id} crashed: {e}")
        finally:
            loop.close()
    
    async def _download_worker(self, worker_id: int) -> None:
        """Worker coroutine for processing downloads."""
        while not self._shutdown.is_set():
            try:
                # Check for shutdown signal
                if self._shutdown.is_set():
                    logger.info(f"Worker {worker_id} shutting down")
                    break
                    
                # Get next download from thread-safe queue
                try:
                    download_id, file_info = self.download_queue.get(timeout=1)
                    logger.info(f"Worker {worker_id} successfully got download task: {download_id}")
                except queue.Empty:
                    # This is normal when queue is empty
                    await asyncio.sleep(0.1)
                    continue
                except Exception as e:
                    logger.error(f"Worker {worker_id} error getting from queue: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    await asyncio.sleep(0.1)
                    continue
                
                if download_id not in self.downloads:
                    continue
                
                download = self.downloads[download_id]
                
                # Update status
                download.status = "downloading"
                download.start_time = datetime.now()
                self.active_downloads += 1
                logger.info(f"Download {download_id} status changed to 'downloading', active downloads: {self.active_downloads}")
                
                # Update UI from thread
                def update_ui():
                    self.downloads = {**self.downloads}
                    self.call_later(self._refresh_downloads_list)
                
                self.app.call_from_thread(update_ui)
                
                # Perform download
                try:
                    from ...LLM_Calls.huggingface_api import HuggingFaceAPI
                    api = HuggingFaceAPI()
                    
                    logger.info(f"Starting download: repo_id={download.repo_id}, filename={download.filename}, destination={download.destination}")
                    
                    def progress_callback(downloaded: int, total: int):
                        download.downloaded = downloaded
                        # Update progress from thread
                        def update_progress():
                            self._update_download_progress(download_id)
                        self.app.call_from_thread(update_progress)
                    
                    success = await api.download_file(
                        download.repo_id,
                        download.filename,  # Use filename from download item
                        download.destination,
                        progress_callback
                    )
                    
                    if success:
                        download.status = "completed"
                        download.end_time = datetime.now()
                        logger.info(f"Download {download_id} completed successfully")
                    else:
                        download.status = "error"
                        download.error_message = "Download failed"
                        logger.error(f"Download {download_id} failed")
                
                except Exception as e:
                    logger.error(f"Download error: {e}")
                    download.status = "error"
                    download.error_message = str(e)
                
                finally:
                    self.active_downloads -= 1
                    logger.info(f"Download {download_id} finished, active downloads: {self.active_downloads}")
                    # Update UI from thread
                    def final_update():
                        self.downloads = {**self.downloads}
                        self.call_later(self._refresh_downloads_list)
                    
                    self.app.call_from_thread(final_update)
                    
            except asyncio.CancelledError:
                logger.info(f"Worker {worker_id} cancelled")
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                if not self._shutdown.is_set():
                    await asyncio.sleep(1)
        
        logger.info(f"Worker {worker_id} finished")
    
    def _update_download_progress(self, download_id: str) -> None:
        """Update progress bar for a download."""
        try:
            downloads_list = self.query_one("#downloads-list", ListView)
            for item in downloads_list.children:
                if hasattr(item, "download_id") and item.download_id == download_id:
                    # Find progress bar in item
                    progress_bars = item.query("ProgressBar")
                    if progress_bars:
                        download = self.downloads[download_id]
                        progress_bars[0].update(progress=download.progress)
                    break
        except Exception as e:
            logger.error(f"Error updating progress: {e}")
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button = event.button
        
        if hasattr(button, "download_id"):
            download_id = button.download_id
            
            if "retry" in button.classes:
                # Retry download
                if download_id in self.downloads:
                    download = self.downloads[download_id]
                    download.status = "pending"
                    download.downloaded = 0
                    download.error_message = None
                    
                    # Re-queue
                    file_info = {
                        "path": download.filename,
                        "size": download.size
                    }
                    self.download_queue.put((download_id, file_info))
                    self.call_later(self._refresh_downloads_list)
            
            elif "remove" in button.classes:
                # Remove from list
                if download_id in self.downloads:
                    del self.downloads[download_id]
                    self.call_later(self._refresh_downloads_list)
    
    @staticmethod
    def _format_bytes(bytes: int) -> str:
        """Format bytes to human readable string."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes < 1024.0:
                return f"{bytes:.1f} {unit}"
            bytes /= 1024.0
        return f"{bytes:.1f} PB"
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds to human readable time."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            return f"{minutes}m"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}h {minutes}m"
    
    def on_unmount(self) -> None:
        """Clean up when widget is unmounted."""
        logger.info("DownloadManager unmounting, signaling workers to shut down")
        # Signal workers to stop
        self._shutdown.set()
        
        # Clear the queue to unblock any workers waiting
        try:
            while not self.download_queue.empty():
                self.download_queue.get_nowait()
        except queue.Empty:
            pass
        
        logger.info("DownloadManager unmounted")