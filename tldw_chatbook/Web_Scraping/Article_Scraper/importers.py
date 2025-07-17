"""
article_scraper/importers.py
============================

Import URLs from bookmark files and CSV sources.

This module provides unified functionality to load URLs from
various bookmark formats and CSV files, making it easy to
bulk-import URLs for scraping.

Supported Formats:
-----------------
- Chrome/Edge bookmarks (JSON format)
- Firefox bookmarks (HTML format)
- CSV files with 'url' column

Functions:
----------
- collect_urls_from_file(): Main entry point for URL import
- _load_from_chromium_json(): Parse Chrome/Edge bookmarks
- _load_from_firefox_html(): Parse Firefox bookmarks
- _load_from_csv(): Parse CSV files

Example:
--------
    # Import Chrome bookmarks
    bookmarks = collect_urls_from_file("/path/to/Bookmarks")
    
    # Import from CSV
    urls = collect_urls_from_file("/path/to/urls.csv")
    
    # Use imported URLs for scraping
    for name, url in bookmarks.items():
        print(f"Processing: {name} - {url}")
"""
#
# Imports
import json
import logging
import os
import time
from typing import Dict, List, Union
#
# Third-Party Libraries
# Handle optional dependencies
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    BeautifulSoup = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None
#
# Local Imports
from ...Metrics.metrics_logger import log_counter, log_histogram
#
#######################################################################################################################
#
# Functions:

def _parse_chromium_bookmarks(nodes: List[Dict]) -> Dict[str, str]:
    """Recursively parses bookmark nodes from Chromium-based browsers."""
    bookmarks = {}
    for node in nodes:
        if node.get('type') == 'url' and 'url' in node and 'name' in node:
            bookmarks[node['name']] = node['url']
        elif node.get('type') == 'folder' and 'children' in node:
            bookmarks.update(_parse_chromium_bookmarks(node['children']))
    return bookmarks


def _load_from_chromium_json(file_path: str) -> Dict[str, str]:
    """Loads and parses a Chromium bookmarks JSON file."""
    start_time = time.time()
    log_counter("importer_chromium_json_attempt")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        bookmarks = {}
        # The actual bookmarks are nested under 'roots'
        if 'roots' in data:
            for root_name, root_content in data['roots'].items():
                if isinstance(root_content, dict) and 'children' in root_content:
                    bookmarks.update(_parse_chromium_bookmarks(root_content['children']))
        
        # Log success
        duration = time.time() - start_time
        log_histogram("importer_chromium_json_duration", duration, labels={"status": "success"})
        log_histogram("importer_chromium_json_bookmarks_count", len(bookmarks))
        log_counter("importer_chromium_json_success")
        
        return bookmarks
    except json.JSONDecodeError as e:
        # Log JSON error
        duration = time.time() - start_time
        log_histogram("importer_chromium_json_duration", duration, labels={"status": "error"})
        log_counter("importer_chromium_json_error", labels={"error_type": "json_decode"})
        logging.error(f"Invalid JSON in bookmarks file {file_path}: {e}")
    except Exception as e:
        # Log other errors
        duration = time.time() - start_time
        log_histogram("importer_chromium_json_duration", duration, labels={"status": "error"})
        log_counter("importer_chromium_json_error", labels={"error_type": type(e).__name__})
        logging.error(f"Failed to read or parse Chromium bookmarks {file_path}: {e}")
    return {}


def _load_from_firefox_html(file_path: str) -> Dict[str, str]:
    """Loads and parses a Firefox bookmarks HTML file."""
    start_time = time.time()
    log_counter("importer_firefox_html_attempt")
    
    if not BS4_AVAILABLE:
        logging.error("BeautifulSoup not available for parsing Firefox bookmarks. Install with: pip install tldw_chatbook[websearch]")
        return {}
    
    bookmarks = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')

        # Bookmarks are <a> tags with an href attribute
        total_links = 0
        skipped_links = 0
        
        for link in soup.find_all('a', href=True):
            total_links += 1
            name = link.get_text(strip=True)
            url = link.get('href')
            if name and url and url.startswith(('http://', 'https://')):
                bookmarks[name] = url
            else:
                skipped_links += 1
        
        # Log success
        duration = time.time() - start_time
        log_histogram("importer_firefox_html_duration", duration, labels={"status": "success"})
        log_histogram("importer_firefox_html_bookmarks_count", len(bookmarks))
        log_counter("importer_firefox_html_success", labels={
            "total_links": str(total_links),
            "valid_bookmarks": str(len(bookmarks)),
            "skipped": str(skipped_links)
        })
        
        return bookmarks
    except Exception as e:
        # Log error
        duration = time.time() - start_time
        log_histogram("importer_firefox_html_duration", duration, labels={"status": "error"})
        log_counter("importer_firefox_html_error", labels={"error_type": type(e).__name__})
        logging.error(f"Failed to read or parse Firefox bookmarks {file_path}: {e}")
    return {}


def _load_from_csv(file_path: str) -> Dict[str, str]:
    """Loads URLs from a CSV file. Expects 'url' and optionally 'title' columns."""
    start_time = time.time()
    log_counter("importer_csv_attempt")
    
    if not PANDAS_AVAILABLE:
        logging.error("Pandas not available for parsing CSV files. Install with: pip install tldw_chatbook[websearch]")
        return {}
    
    bookmarks = {}
    try:
        df = pd.read_csv(file_path)
        row_count = len(df)
        log_histogram("importer_csv_row_count", row_count)

        if 'url' not in df.columns:
            log_counter("importer_csv_error", labels={"error_type": "missing_url_column"})
            logging.error(f"CSV file {file_path} must contain a 'url' column.")
            return {}

        # Prefer 'title', then 'name', otherwise generate a key
        title_col = 'title' if 'title' in df.columns else ('name' if 'name' in df.columns else None)
        has_title_col = title_col is not None
        
        skipped_rows = 0
        for index, row in df.iterrows():
            url = row['url']
            if pd.notna(url):
                name = row[title_col] if title_col and pd.notna(row[title_col]) else f"URL from CSV row {index + 1}"
                bookmarks[name] = url
            else:
                skipped_rows += 1
        
        # Log success
        duration = time.time() - start_time
        log_histogram("importer_csv_duration", duration, labels={"status": "success"})
        log_histogram("importer_csv_urls_count", len(bookmarks))
        log_counter("importer_csv_success", labels={
            "total_rows": str(row_count),
            "valid_urls": str(len(bookmarks)),
            "skipped": str(skipped_rows),
            "has_title_column": str(has_title_col)
        })
        
        return bookmarks
    except FileNotFoundError:
        # Log file not found
        duration = time.time() - start_time
        log_histogram("importer_csv_duration", duration, labels={"status": "error"})
        log_counter("importer_csv_error", labels={"error_type": "file_not_found"})
        logging.error(f"CSV file not found at {file_path}")
    except Exception as e:
        # Log other errors
        duration = time.time() - start_time
        log_histogram("importer_csv_duration", duration, labels={"status": "error"})
        log_counter("importer_csv_error", labels={"error_type": type(e).__name__})
        logging.error(f"Failed to read or parse CSV file {file_path}: {e}")
    return {}


def collect_urls_from_file(file_path: str) -> Dict[str, str]:
    """
    Unified function to collect URLs from a file.
    Detects file type (JSON, HTML, CSV) and uses the appropriate parser.

    Args:
        file_path: Path to the bookmarks or CSV file.

    Returns:
        A dictionary mapping bookmark/entry names to their URLs.
    """
    start_time = time.time()
    
    if not os.path.exists(file_path):
        log_counter("importer_file_not_found")
        logging.error(f"File not found: {file_path}")
        return {}

    _, ext = os.path.splitext(file_path.lower())
    file_size = os.path.getsize(file_path)
    
    # Log import attempt
    log_counter("importer_collect_urls_attempt", labels={"file_type": ext or "no_extension"})
    log_histogram("importer_file_size_bytes", file_size)

    logging.info(f"Importing URLs from {file_path}...")

    if ext == '.json':
        urls = _load_from_chromium_json(file_path)
    elif ext in ['.html', '.htm']:
        urls = _load_from_firefox_html(file_path)
    elif ext == '.csv':
        urls = _load_from_csv(file_path)
    else:
        # As a fallback, try JSON parsing for files with no extension (like default Chrome Bookmarks file)
        if ext == '':
            logging.warning("File has no extension, attempting to parse as Chromium JSON bookmarks.")
            log_counter("importer_no_extension_fallback")
            urls = _load_from_chromium_json(file_path)
        else:
            log_counter("importer_unsupported_file_type", labels={"extension": ext})
            logging.error(f"Unsupported file type: '{ext}'. Please use .json, .html, or .csv.")
            return {}

    # Log final results
    duration = time.time() - start_time
    log_histogram("importer_collect_urls_duration", duration, labels={
        "file_type": ext or "no_extension",
        "urls_count": str(len(urls))
    })
    log_counter("importer_collect_urls_complete", labels={
        "file_type": ext or "no_extension",
        "urls_imported": str(len(urls)),
        "success": str(len(urls) > 0)
    })
    
    logging.info(f"Successfully imported {len(urls)} URLs from file.")
    return urls

#
# End of article_scraper/importers.py
#######################################################################################################################
