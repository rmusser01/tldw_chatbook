# serve.py
"""
Web server module for running tldw_chatbook in a browser using textual-serve.

This module provides functions to launch the Textual application as a web server,
allowing users to access the TUI through their web browser.
"""

import sys
from typing import Optional
from pathlib import Path
from loguru import logger

from ..Utils.optional_deps import DEPENDENCIES_AVAILABLE, require_dependency
from ..config import get_cli_setting


def check_web_server_available() -> bool:
    """Check if web server dependencies are available."""
    return DEPENDENCIES_AVAILABLE.get('web', False)


def create_server(
    host: str = "localhost",
    port: int = 8000,
    title: Optional[str] = None,
    debug: bool = False
):
    """
    Create and configure a textual-serve Server instance.
    
    Args:
        host: The host address to bind to (default: localhost)
        port: The port to bind to (default: 8000) 
        title: Title for the web page (default: "tldw chatbook")
        debug: Enable debug mode (default: False)
        
    Returns:
        Configured Server instance
        
    Raises:
        ImportError: If textual-serve is not installed
    """
    # Require the dependency
    textual_serve = require_dependency('textual_serve', 'web')
    
    # Import the Server class
    from textual_serve.server import Server
    
    # Create the command to run the app
    # textual-serve expects a command string, not a list
    command = f"{sys.executable} -m tldw_chatbook.app"
    
    # Configure title
    if title is None:
        title = get_cli_setting("web_server", "title", default="tldw chatbook")
    
    # Create the server
    logger.info(f"Creating web server on {host}:{port}")
    server = Server(
        command=command,
        host=host,
        port=port,
        title=title
    )
    
    return server


def run_web_server(
    host: Optional[str] = None,
    port: Optional[int] = None,
    title: Optional[str] = None,
    debug: Optional[bool] = None
):
    """
    Run the tldw_chatbook application as a web server.
    
    This function starts a web server that serves the Textual application,
    allowing users to access it through their web browser.
    
    Args:
        host: Host address (default: from config or "localhost")
        port: Port number (default: from config or 8000)
        title: Page title (default: from config or "tldw chatbook")
        debug: Enable debug mode (default: from config or False)
        
    Raises:
        ImportError: If textual-serve is not installed
    """
    if not check_web_server_available():
        logger.error("Web server dependencies not available.")
        logger.error("Install with: pip install tldw_chatbook[web]")
        sys.exit(1)
    
    # Load settings from config with defaults
    web_config = get_cli_setting("web_server", default={})
    
    # Use provided values or fall back to config/defaults
    host = host if host is not None else web_config.get("host", "localhost")
    port = port if port is not None else web_config.get("port", 8000)
    title = title if title is not None else web_config.get("title", "tldw chatbook")
    debug = debug if debug is not None else web_config.get("debug", False)
    
    # Create and run the server
    server = create_server(host=host, port=port, title=title, debug=debug)
    
    logger.info(f"Starting web server at http://{host}:{port}")
    logger.info("Press Ctrl+C to stop the server")
    
    try:
        server.serve()
    except KeyboardInterrupt:
        logger.info("Web server stopped by user")
    except Exception as e:
        logger.error(f"Error running web server: {e}")
        raise


def main():
    """Entry point for the tldw-serve command."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run tldw_chatbook in a web browser",
        prog="tldw-serve"
    )
    parser.add_argument(
        "--host",
        type=str,
        help="Host address to bind to (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Port to bind to (default: 8000)"
    )
    parser.add_argument(
        "--title",
        type=str,
        help="Title for the web page"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    # Run the web server with provided arguments
    run_web_server(
        host=args.host,
        port=args.port,
        title=args.title,
        debug=args.debug
    )


if __name__ == "__main__":
    main()