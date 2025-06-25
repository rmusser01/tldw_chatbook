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
from typing import Dict, List, Union
#
# Third-Party Libraries
from bs4 import BeautifulSoup
import pandas as pd
#
# Local Imports
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
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        bookmarks = {}
        # The actual bookmarks are nested under 'roots'
        if 'roots' in data:
            for root_name, root_content in data['roots'].items():
                if isinstance(root_content, dict) and 'children' in root_content:
                    bookmarks.update(_parse_chromium_bookmarks(root_content['children']))
        return bookmarks
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in bookmarks file {file_path}: {e}")
    except Exception as e:
        logging.error(f"Failed to read or parse Chromium bookmarks {file_path}: {e}")
    return {}


def _load_from_firefox_html(file_path: str) -> Dict[str, str]:
    """Loads and parses a Firefox bookmarks HTML file."""
    bookmarks = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')

        # Bookmarks are <a> tags with an href attribute
        for link in soup.find_all('a', href=True):
            name = link.get_text(strip=True)
            url = link.get('href')
            if name and url and url.startswith(('http://', 'https://')):
                bookmarks[name] = url
        return bookmarks
    except Exception as e:
        logging.error(f"Failed to read or parse Firefox bookmarks {file_path}: {e}")
    return {}


def _load_from_csv(file_path: str) -> Dict[str, str]:
    """Loads URLs from a CSV file. Expects 'url' and optionally 'title' columns."""
    bookmarks = {}
    try:
        df = pd.read_csv(file_path)

        if 'url' not in df.columns:
            logging.error(f"CSV file {file_path} must contain a 'url' column.")
            return {}

        # Prefer 'title', then 'name', otherwise generate a key
        title_col = 'title' if 'title' in df.columns else ('name' if 'name' in df.columns else None)

        for index, row in df.iterrows():
            url = row['url']
            if pd.notna(url):
                name = row[title_col] if title_col and pd.notna(row[title_col]) else f"URL from CSV row {index + 1}"
                bookmarks[name] = url
        return bookmarks
    except FileNotFoundError:
        logging.error(f"CSV file not found at {file_path}")
    except Exception as e:
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
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return {}

    _, ext = os.path.splitext(file_path.lower())

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
            urls = _load_from_chromium_json(file_path)
        else:
            logging.error(f"Unsupported file type: '{ext}'. Please use .json, .html, or .csv.")
            return {}

    logging.info(f"Successfully imported {len(urls)} URLs from file.")
    return urls

#
# End of article_scraper/importers.py
#######################################################################################################################
