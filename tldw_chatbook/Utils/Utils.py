# Utils.py
#########################################
# General Utilities Library
# This library is used to hold random utilities used by various other libraries.
#
####
####################
# Function Categories
#
#     Config loading
#     Misc-Functions
#     File-saving Function Definitions
#     UUID-Functions
#     Sanitization/Verification Functions
#     DB Config Loading
#     File Handling Functions
#
####################
# Function List
#
# 1. extract_text_from_segments(segments: List[Dict]) -> str
# 2. download_file(url, dest_path, expected_checksum=None, max_retries=3, delay=5)
# 3. verify_checksum(file_path, expected_checksum)
# 4. create_download_directory(title)
# 5. sanitize_filename(filename)
# 6. normalize_title(title)
# 7.
#
####################
#
# Import necessary libraries
from pathlib import Path
import configparser
import hashlib
import json
import logging
import os
import re
import tempfile
from .secure_temp_files import get_temp_manager, secure_delete_file
import time
import uuid
from datetime import timedelta, datetime
from typing import Union, AnyStr, Tuple, List, Optional, Protocol, cast
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
#
# 3rd-Party Imports
import chardet
import unicodedata
from loguru import logger

#
#######################################################################################################################
#
# Function Definitions


#######################################################################################################################
# Config loading
#

# --- Project Structure Constants ---

# Determine the Project Root Directory dynamically relative to *this file* (Utils.py)
# Assuming Utils.py is in ~/tldw_cli/Libs/
UTILS_FILE_PATH = Path(__file__).resolve() # Absolute path to Utils.py
LIBS_DIR = UTILS_FILE_PATH.parent          # Absolute path to ~/tldw_cli/Libs/
PROJECT_ROOT_DIR = LIBS_DIR.parent         # Absolute path to ~/tldw_cli/

# --- Configuration File Path ---
# Config file is directly within the project root
CONFIG_FILENAME = 'config.txt'
CONFIG_FILE_PATH = PROJECT_ROOT_DIR / CONFIG_FILENAME

# --- User Database Path (Fixed Location) ---
USER_DB_DIR = Path.home() / ".config" / "tldw_cli"
USER_DB_FILENAME = "tldw_cli_Media.db"
USER_DB_PATH = USER_DB_DIR / USER_DB_FILENAME

# --- Project-Internal Database Directory (if needed) ---
PROJECT_DB_DIR_NAME = "Databases" # Name of the subdir within project root
PROJECT_DATABASES_DIR = PROJECT_ROOT_DIR / PROJECT_DB_DIR_NAME





# --- Functions for Project-Internal Databases (if needed) ---

def extract_text_from_segments(segments, include_timestamps=True):
    logger.trace(f"Segments received: {segments}")
    logger.trace(f"Type of segments: {type(segments)}")

    def extract_text_recursive(data, include_timestamps):
        if isinstance(data, dict):
            text = data.get('Text', '')
            if include_timestamps and 'Time_Start' in data and 'Time_End' in data:
                return f"{data['Time_Start']}s - {data['Time_End']}s | {text}"
            for key, value in data.items():
                if key == 'Text':
                    return value
                elif isinstance(value, (dict, list)):
                    result = extract_text_recursive(value, include_timestamps)
                    if result:
                        return result
        elif isinstance(data, list):
            return '\n'.join(filter(None, [extract_text_recursive(item, include_timestamps) for item in data]))
        return None

    text = extract_text_recursive(segments, include_timestamps)

    if text:
        return text.strip()
    else:
        logging.error(f"Unable to extract text from segments: {segments}")
        return "Error: Unable to extract transcription"

def ensure_directory_exists(path):
    """Ensure that a directory exists, creating it if necessary."""
    os.makedirs(path, exist_ok=True)


global_api_endpoints = ["anthropic", "cohere", "google", "groq", "openai", "huggingface", "openrouter", "deepseek", "mistral", "custom_openai_api", "custom_openai_api_2", "llama", "ollama", "ooba", "kobold", "tabby", "vllm", "aphrodite"]

global_search_engines = ["baidu", "bing", "brave", "duckduckgo", "google", "kagi", "searx", "tavily", "yandex"]

openai_tts_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]


def format_api_name(api):
    name_mapping = {
        "openai": "OpenAI",
        "anthropic": "Anthropic",
        "cohere": "Cohere",
        "google": "Google",
        "groq": "Groq",
        "huggingface": "HuggingFace",
        "openrouter": "OpenRouter",
        "deepseek": "DeepSeek",
        "mistral": "Mistral",
        "custom_openai_api": "Custom-OpenAI-API",
        "custom_openai_api_2": "Custom-OpenAI-API-2",
        "llama": "Llama.cpp",
        "ooba": "Ooba",
        "kobold": "Kobold",
        "tabby": "Tabbyapi",
        "vllm": "VLLM",
        "ollama": "Ollama",
        "aphrodite": "Aphrodite"
    }
    return name_mapping.get(api, api.title())

#
# End of Config loading
#######################################################################################################################


#######################################################################################################################
#
# Misc-Functions

# Log file
# logging.basicConfig(filename='debug-runtime.log', encoding='utf-8', level=logging.DEBUG)


# # Example usage:
# example_metadata = {
#     'title': 'Sample Video Title',
#     'uploader': 'Channel Name',
#     'upload_date': '20230615',
#     'view_count': 1000000,
#     'like_count': 50000,
#     'duration': 3725,  # 1 hour, 2 minutes, 5 seconds
#     'tags': ['tag1', 'tag2', 'tag3'],
#     'description': 'This is a sample video description.'
# }
#
# print(format_metadata_as_text(example_metadata))


def convert_to_seconds(time_str):
    if not time_str:
        return 0

    # If it's already a number, assume it's in seconds
    if time_str.isdigit():
        return int(time_str)

    # Parse time string in format HH:MM:SS, MM:SS, or SS
    time_parts = time_str.split(':')
    if len(time_parts) == 3:
        return int(timedelta(hours=int(time_parts[0]),
                             minutes=int(time_parts[1]),
                             seconds=int(time_parts[2])).total_seconds())
    elif len(time_parts) == 2:
        return int(timedelta(minutes=int(time_parts[0]),
                             seconds=int(time_parts[1])).total_seconds())
    elif len(time_parts) == 1:
        return int(time_parts[0])
    else:
        raise ValueError(f"Invalid time format: {time_str}")


def truncate_content(content: Optional[str], max_length: int = 200) -> Optional[str]:
    """Truncate content to the specified maximum length with ellipsis."""
    if not content:
        return content

    if len(content) <= max_length:
        return content

    return content[:max_length - 3] + "..."

#
# End of Misc-Functions
#######################################################################################################################


#######################################################################################################################
#
# File-saving Function Definitions
def save_to_file(video_urls, filename):
    with open(filename, 'w') as file:
        file.write('\n'.join(video_urls))
    logging.info(f"Video URLs saved to {filename}")


def save_segments_to_json(segments, file_name="transcription_segments.json"):
    """
    Save transcription segments to a JSON file.

    Parameters:
    segments (list): List of transcription segments
    file_name (str): Name of the JSON file to save (default: "transcription_segments.json")

    Returns:
    str: Path to the saved JSON file
    """
    # Ensure the Results directory exists
    os.makedirs("Results", exist_ok=True)

    # Full path for the JSON file
    json_file_path = os.path.join("Results", file_name)

    # Save segments to JSON file
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(segments, json_file, ensure_ascii=False, indent=4)

    return json_file_path


def safe_read_file(file_path):
    encodings = ['utf-8', 'utf-16', 'ascii', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-8-sig']

    logging.info(f"Attempting to read file: {file_path}")

    try:
        with open(file_path, 'rb') as file:
            logging.debug(f"Reading file in binary mode: {file_path}")
            raw_data = file.read()
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return f"File not found: {file_path}"
    except Exception as e:
        logging.error(f"An error occurred while reading the file: {e}")
        return f"An error occurred while reading the file: {e}"

    if not raw_data:
        logging.warning(f"File is empty: {file_path}")
        return ""

    # Use chardet to detect the encoding
    detected = chardet.detect(raw_data)
    if detected['encoding'] is not None:
        encodings.insert(0, detected['encoding'])
        logging.info(f"Detected encoding: {detected['encoding']}")

    for encoding in encodings:
        logging.info(f"Trying encoding: {encoding}")
        try:
            decoded_content = raw_data.decode(encoding)
            # Check if the content is mostly printable
            if sum(c.isprintable() for c in decoded_content) / len(decoded_content) > 0.90:
                logging.info(f"Successfully decoded file with encoding: {encoding}")
                return decoded_content
        except UnicodeDecodeError:
            logging.debug(f"Failed to decode with {encoding}")
            continue

    # If all decoding attempts fail, return the error message
    logging.error(f"Unable to decode the file {file_path}")
    return f"Unable to decode the file {file_path}"

#
# End of Files-saving Function Definitions
#######################################################################################################################


#######################################################################################################################
#
# UUID-Functions

def generate_unique_filename(base_path, base_filename):
    """Generate a unique filename by appending a counter if necessary."""
    filename = base_filename
    counter = 1
    while os.path.exists(os.path.join(base_path, filename)):
        name, ext = os.path.splitext(base_filename)
        filename = f"{name}_{counter}{ext}"
        counter += 1
    return filename


def generate_unique_identifier(file_path):
    filename = os.path.basename(file_path)
    timestamp = int(time.time())

    # Generate a hash of the file content
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    content_hash = hasher.hexdigest()[:8]  # Use first 8 characters of the hash

    return f"local:{timestamp}:{content_hash}:{filename}"

#
# End of UUID-Functions
#######################################################################################################################


#######################################################################################################################
#
# Sanitization/Verification Functions

# Helper function to validate URL format
def is_valid_url(url: str) -> bool:
    regex = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|'  # ...or ipv4
        r'\[?[A-F0-9]*:[A-F0-9:]+\]?)'  # ...or ipv6
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return re.match(regex, url) is not None


def verify_checksum(file_path, expected_checksum):
    sha256_hash = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for byte_block in iter(lambda: f.read(4096), b''):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest() == expected_checksum


def normalize_title(title, preserve_spaces=False):
    # Normalize the string to 'NFKD' form and encode to 'ascii' ignoring non-ascii characters
    title = unicodedata.normalize('NFKD', title).encode('ascii', 'ignore').decode('ascii')

    if preserve_spaces:
        # Replace special characters with underscores, but keep spaces
        title = re.sub(r'[^\w\s\-.]', '_', title)
    else:
        # Replace special characters and spaces with underscores
        title = re.sub(r'[^\w\-.]', '_', title)

    # Replace multiple consecutive underscores with a single underscore
    title = re.sub(r'_+', '_', title)

    # Replace specific characters with underscores
    title = title.replace('/', '_').replace('\\', '_').replace(':', '_').replace('"', '_').replace('*', '_').replace(
        '?', '_').replace(
        '<', '_').replace('>', '_').replace('|', '_')

    return title.strip('_')


def clean_youtube_url(url):
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    if 'list' in query_params:
        query_params.pop('list')
    cleaned_query = urlencode(query_params, doseq=True)
    cleaned_url = urlunparse(parsed_url._replace(query=cleaned_query))
    return cleaned_url


def sanitize_user_input(message):
    """
    Removes or escapes '{{' and '}}' to prevent placeholder injection.

    Args:
        message (str): The user's message.

    Returns:
        str: Sanitized message.
    """
    # Replace '{{' and '}}' with their escaped versions
    message = re.sub(r'\{\{', '{ {', message)
    message = re.sub(r'\}\}', '} }', message)
    return message

def format_file_path(file_path, fallback_path=None):
    if file_path and os.path.exists(file_path):
        logging.debug(f"File exists: {file_path}")
        return file_path
    elif fallback_path and os.path.exists(fallback_path):
        logging.debug(f"File does not exist: {file_path}. Returning fallback path: {fallback_path}")
        return fallback_path
    else:
        logging.debug(f"File does not exist: {file_path}. No fallback path available.")
        return None


def safe_float(value: str, default: float, name: str) -> float:
    """Safely converts a string to a float, returning a default on failure."""
    if not value: # Handles empty string case
        return default
    try:
        return float(value)
    except ValueError:
        logging.warning(f"Invalid float value for {name}: '{value}'. Using default: {default}.")
        return default

def safe_int(value: str, default: Optional[int], name: str) -> Optional[int]:
    """
    Safely converts a string to an int, returning a default on failure.
    Allows default to be None, in which case None is returned on failure if default is None.
    """
    if not value: # Handles empty string case
        return default
    try:
        return int(value)
    except ValueError:
        logging.warning(f"Invalid integer value for {name}: '{value}'. Using default: {default}.")
        return default

#
# End of Sanitization/Verification Functions
#######################################################################################################################


#######################################################################################################################
#
# File Handling Functions

# Secure temporary file management using the new secure utilities
def save_temp_file(file):
    """Save uploaded file to a secure temporary location."""
    temp_manager = get_temp_manager()
    
    # Read file content
    file_content = file.read()
    
    # Create secure temporary file
    temp_path = temp_manager.create_temp_file(
        file_content, 
        suffix=f"_{file.name}",
        prefix="upload_"
    )
    return temp_path

def cleanup_temp_files():
    """Clean up all temporary files using secure deletion."""
    temp_manager = get_temp_manager()
    temp_manager.cleanup_all()


def generate_unique_id():
    return f"uploaded_file_{uuid.uuid4()}"


class FileProcessor:
    """Handles file reading and name processing"""

    VALID_EXTENSIONS = {'.md', '.txt', '.zip'}
    ENCODINGS_TO_TRY = [
        'utf-8',
        'utf-16',
        'windows-1252',
        'iso-8859-1',
        'ascii'
    ]

    @staticmethod
    def detect_encoding(file_path: str) -> str:
        """Detect the file encoding using chardet"""
        with open(file_path, 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            return result['encoding'] or 'utf-8'

    @staticmethod
    def read_file_content(file_path: str) -> str:
        """Read file content with automatic encoding detection"""
        detected_encoding = FileProcessor.detect_encoding(file_path)

        # Try detected encoding first
        try:
            with open(file_path, 'r', encoding=detected_encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            # If detected encoding fails, try others
            for encoding in FileProcessor.ENCODINGS_TO_TRY:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue

            # If all encodings fail, use utf-8 with error handling
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                return f.read()

    @staticmethod
    def process_filename_to_title(filename: str) -> str:
        """Convert filename to a readable title"""
        # Remove extension
        name = os.path.splitext(filename)[0]

        # Look for date patterns
        date_pattern = r'(\d{4}[-_]?\d{2}[-_]?\d{2})'
        date_match = re.search(date_pattern, name)
        date_str = ""
        if date_match:
            try:
                date = datetime.strptime(date_match.group(1).replace('_', '-'), '%Y-%m-%d')
                date_str = date.strftime("%b %d, %Y")
                name = name.replace(date_match.group(1), '').strip('-_')
            except ValueError:
                pass

        # Replace separators with spaces
        name = re.sub(r'[-_]+', ' ', name)

        # Remove redundant spaces
        name = re.sub(r'\s+', ' ', name).strip()

        # Capitalize words, excluding certain words
        exclude_words = {'a', 'an', 'the', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
        words = name.split()
        capitalized = []
        for i, word in enumerate(words):
            if i == 0 or word not in exclude_words:
                capitalized.append(word.capitalize())
            else:
                capitalized.append(word.lower())
        name = ' '.join(capitalized)

        # Add date if found
        if date_str:
            name = f"{name} - {date_str}"

        return name

def get_formatted_file_size(file_path: Path) -> Optional[str]:
    """Returns the file size in a human-readable format (KB, MB, GB) or None if file doesn't exist."""
    try:
        if not file_path.exists() or not file_path.is_file():
            return None
        size_bytes = os.path.getsize(file_path)
        if size_bytes < 1024:
            return f"{size_bytes} B"
        size_kb = size_bytes / 1024
        if size_kb < 1024:
            return f"{size_kb:.1f} KB"
        size_mb = size_kb / 1024
        if size_mb < 1024:
            return f"{size_mb:.1f} MB"
        size_gb = size_mb / 1024
        return f"{size_gb:.1f} GB"
    except OSError: # Catch potential errors like permission denied
        return None

#
# End of File Handling Functions
#######################################################################################################################

def extract_media_id_from_result_string(result_msg: Optional[str]) -> Optional[str]:
    """
    Extracts the Media ID from a string expected to contain 'Media ID: <id>'.

    This function searches for the pattern "Media ID:" followed by optional
    whitespace and captures the subsequent sequence of non-whitespace characters
    as the ID.

    Args:
        result_msg: The input string potentially containing the Media ID message,
                    typically returned by processing functions like import_epub.

    Returns:
        The extracted Media ID as a string if the pattern is found.
        Returns None if the input string is None, empty, or the pattern
        "Media ID: <id>" is not found.

    Examples:
        >>> extract_media_id_from_result_string("Ebook imported successfully. Media ID: ebook_789")
        'ebook_789'
        >>> extract_media_id_from_result_string("Success. Media ID: db_mock_id")
        'db_mock_id'
        >>> extract_media_id_from_result_string("Error during processing.")
        None
        >>> extract_media_id_from_result_string(None)
        None
        >>> extract_media_id_from_result_string("Media ID: id-with-hyphens123") # Test hyphens/numbers
        'id-with-hyphens123'
        >>> extract_media_id_from_result_string("Media ID:id_no_space") # Test no space
        'id_no_space'
    """
    # Handle None or empty input string gracefully
    if not result_msg:
        return None

    # Regular expression pattern:
    # - Looks for the literal string "Media ID:" (case-sensitive).
    # - Allows for zero or more whitespace characters (\s*) after the colon.
    # - Captures (\(...\)) one or more non-whitespace characters (\S+).
    #   Using \S+ is generally safer than \w+ as IDs might contain hyphens or other symbols.
    #   If IDs are strictly alphanumeric + underscore, you could use (\w+) instead.
    # - We use re.search to find the pattern anywhere in the string.
    pattern = r"Media ID:\s*(\S+)"

    match = re.search(pattern, result_msg)

    # If a match is found, match.group(1) will contain the captured ID part
    if match:
        return match.group(1)
    else:
        # The pattern "Media ID: <id>" was not found in the string
        return None

def get_api_name(app_instance, provider: str, endpoints: dict) -> Optional[str]:
    if not app_instance._ui_ready:
        return None
    provider_key_map = { "llama_cpp": "llama_cpp", "Ollama": "Ollama", "Oobabooga": "Oobabooga", "koboldcpp": "koboldcpp", "vllm": "vllm", "Custom": "Custom", "Custom-2": "Custom_2", }
    endpoint_key = provider_key_map.get(provider)
    if endpoint_key:
        url = endpoints.get(endpoint_key)
        if url: return url
        else: logging.warning(f"URL key '{endpoint_key}' for provider '{provider}' missing in config [api_endpoints].")
    return None


#
# End of Utils.py
########################################################################################################################