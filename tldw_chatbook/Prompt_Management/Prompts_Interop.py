# prompts_interop.py
#
"""
Prompts Interop Library
-----------------------

This library serves as an intermediary layer between an API endpoint (or other
application logic) and the Prompts_DB_v2 library. It manages a single instance
of the PromptsDatabase and exposes its functionality, promoting decoupling and
centralized database configuration.

Usage:
1. Initialize at application startup:
   `initialize_interop(db_path="your_prompts.db", client_id="your_api_client")`

2. Call interop functions in your application code:
   `prompts = list_prompts(page=1, per_page=10)`
   `import_results = import_prompts_from_files(["my_prompts.json", "more_prompts.md"])`

3. (Optional) Clean up at application shutdown:
   `shutdown_interop()`
"""
#
# Imports
import logging # Retained for example, but loguru is preferred
import json
import re
from typing import List, Tuple, Dict, Any, Optional, Union, Callable
from pathlib import Path
import tempfile # For example usage
import os       # For example usage

#
# 3rd-party Libraries
from loguru import logger

# Conditional imports for new functionality
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger.warning("PyYAML not installed. YAML import will not be available.")

try:
    import frontmatter # For Markdown
    FRONTMATTER_AVAILABLE = True
except ImportError:
    FRONTMATTER_AVAILABLE = False
    logger.warning("python-frontmatter not installed. Markdown import will not be available.")
#
# Local Imports
from tldw_chatbook.DB.Prompts_DB import (
    PromptsDatabase,
    DatabaseError,
    SchemaError,
    InputError,
    ConflictError,
    add_or_update_prompt as db_add_or_update_prompt,
    load_prompt_details_for_ui as db_load_prompt_details_for_ui,
    export_prompt_keywords_to_csv as db_export_prompt_keywords_to_csv,
    view_prompt_keywords_markdown as db_view_prompt_keywords_markdown,
    export_prompts_formatted as db_export_prompts_formatted
)
from tldw_chatbook.Utils.path_validation import validate_path
#
#######################################################################################################################
#
# Functions:

_db_instance: Optional[PromptsDatabase] = None
_db_path_global: Optional[Union[str, Path]] = None
_client_id_global: Optional[str] = None

# --- Initialization and Management ---

def initialize_interop(db_path: Union[str, Path], client_id: str) -> None:
    """
    Initializes the interop library with a single PromptsDatabase instance.
    This should be called once, e.g., at application startup.

    Args:
        db_path: Path to the SQLite database file or ':memory:'.
        client_id: A unique identifier for the client using this database instance.

    Raises:
        ValueError: If db_path or client_id are invalid.
        DatabaseError, SchemaError: If PromptsDatabase initialization fails.
    """
    global _db_instance, _db_path_global, _client_id_global
    if _db_instance is not None:
        logger.warning(
            f"Prompts interop library already initialized with DB: '{_db_path_global}'. "
            f"Re-initializing with DB: '{db_path}', Client ID: '{client_id}'."
        )
        # PromptsDatabase manages thread-local connections; explicit close of old global instance
        # might be complex if other threads are still using it.
        # The PromptsDatabase instance itself doesn't hold a global connection to close here.

    if not db_path: # Pathlib('') is False-y, so this covers empty strings too
        raise ValueError("db_path is required for initialization.")
    if not client_id:
        raise ValueError("client_id is required for initialization.")

    logger.info(f"Initializing Prompts Interop Library. DB Path: {db_path}, Client ID: {client_id}")
    try:
        _db_instance = PromptsDatabase(db_path=db_path, client_id=client_id)
        _db_path_global = db_path
        _client_id_global = client_id
        logger.info("Prompts Interop Library initialized successfully.")
    except (DatabaseError, SchemaError, ValueError) as e:
        logger.critical(f"Failed to initialize PromptsDatabase for interop: {e}", exc_info=True)
        _db_instance = None # Ensure it's None if init fails
        raise

def get_db_instance() -> PromptsDatabase:
    """
    Returns the initialized PromptsDatabase instance.

    Returns:
        The active PromptsDatabase instance.

    Raises:
        RuntimeError: If the library has not been initialized.
    """
    if _db_instance is None:
        msg = "Prompts Interop Library not initialized. Call initialize_interop() first."
        logger.error(msg)
        raise RuntimeError(msg)
    return _db_instance

def is_initialized() -> bool:
    """Checks if the interop library (and thus the DB instance) is initialized."""
    return _db_instance is not None

def shutdown_interop() -> None:
    """
    Cleans up resources. For PromptsDatabase, this primarily means attempting
    to close the database connection for the current thread if one is active.
    Other thread-local connections are managed by their respective threads.
    Resets the global _db_instance to None.
    """
    global _db_instance, _db_path_global, _client_id_global
    if _db_instance:
        logger.info(f"Shutting down Prompts Interop Library for DB: {_db_path_global}.")
        try:
            # This will close the connection for the current thread
            _db_instance.close_connection()
        except Exception as e:
            logger.error(f"Error during Prompts Interop Library shutdown (closing current thread's connection): {e}", exc_info=True)
        _db_instance = None
        _db_path_global = None
        _client_id_global = None
        logger.info("Prompts Interop Library shut down.")
    else:
        logger.info("Prompts Interop Library was not initialized or already shut down.")

# --- Wrapper Functions for PromptsDatabase methods ---

# --- Mutating Methods ---
def add_keyword(keyword_text: str) -> Tuple[Optional[int], Optional[str]]:
    """Adds a keyword. See PromptsDatabase.add_keyword for details."""
    db = get_db_instance()
    return db.add_keyword(keyword_text)

def add_prompt(name: str, author: Optional[str], details: Optional[str],
               system_prompt: Optional[str] = None, user_prompt: Optional[str] = None,
               keywords: Optional[List[str]] = None, overwrite: bool = False
               ) -> Tuple[Optional[int], Optional[str], str]:
    """Adds or updates a prompt. See PromptsDatabase.add_prompt for details."""
    db = get_db_instance()
    return db.add_prompt(name, author, details, system_prompt, user_prompt, keywords, overwrite)

def update_keywords_for_prompt(prompt_id: int, keywords_list: List[str]) -> None:
    """Updates keywords for a specific prompt. See PromptsDatabase.update_keywords_for_prompt for details."""
    db = get_db_instance()
    db.update_keywords_for_prompt(prompt_id, keywords_list)

def soft_delete_prompt(prompt_id_or_name_or_uuid: Union[int, str]) -> bool:
    """Soft deletes a prompt. See PromptsDatabase.soft_delete_prompt for details."""
    db = get_db_instance()
    return db.soft_delete_prompt(prompt_id_or_name_or_uuid)

def soft_delete_keyword(keyword_text: str) -> bool:
    """Soft deletes a keyword. See PromptsDatabase.soft_delete_keyword for details."""
    db = get_db_instance()
    return db.soft_delete_keyword(keyword_text)

# --- Read Methods ---
def get_prompt_by_id(prompt_id: int, include_deleted: bool = False) -> Optional[Dict]:
    """Fetches a prompt by its ID. See PromptsDatabase.get_prompt_by_id for details."""
    db = get_db_instance()
    return db.get_prompt_by_id(prompt_id, include_deleted)

def get_prompt_by_uuid(prompt_uuid: str, include_deleted: bool = False) -> Optional[Dict]:
    """Fetches a prompt by its UUID. See PromptsDatabase.get_prompt_by_uuid for details."""
    db = get_db_instance()
    return db.get_prompt_by_uuid(prompt_uuid, include_deleted)

def get_prompt_by_name(name: str, include_deleted: bool = False) -> Optional[Dict]:
    """Fetches a prompt by its name. See PromptsDatabase.get_prompt_by_name for details."""
    db = get_db_instance()
    return db.get_prompt_by_name(name, include_deleted)

def list_prompts(page: int = 1, per_page: int = 10, include_deleted: bool = False
                 ) -> Tuple[List[Dict], int, int, int]:
    """Lists prompts with pagination. See PromptsDatabase.list_prompts for details."""
    db = get_db_instance()
    return db.list_prompts(page, per_page, include_deleted)

def fetch_prompt_details(prompt_id_or_name_or_uuid: Union[int, str], include_deleted: bool = False
                         ) -> Optional[Dict]:
    """Fetches detailed information for a prompt. See PromptsDatabase.fetch_prompt_details for details."""
    db = get_db_instance()
    return db.fetch_prompt_details(prompt_id_or_name_or_uuid, include_deleted)

def fetch_all_keywords(include_deleted: bool = False) -> List[str]:
    """Fetches all keywords. See PromptsDatabase.fetch_all_keywords for details."""
    db = get_db_instance()
    return db.fetch_all_keywords(include_deleted)

def fetch_keywords_for_prompt(prompt_id: int, include_deleted: bool = False) -> List[str]:
    """Fetches keywords associated with a specific prompt. See PromptsDatabase.fetch_keywords_for_prompt for details."""
    db = get_db_instance()
    return db.fetch_keywords_for_prompt(prompt_id, include_deleted)

def search_prompts(search_query: Optional[str],
                   search_fields: Optional[List[str]] = None,
                   page: int = 1,
                   results_per_page: int = 20,
                   include_deleted: bool = False
                   ) -> Tuple[List[Dict[str, Any]], int]:
    """Searches prompts using FTS. See PromptsDatabase.search_prompts for details."""
    db = get_db_instance()
    return db.search_prompts(search_query, search_fields, page, results_per_page, include_deleted)

# --- Sync Log Access Methods ---
def get_sync_log_entries(since_change_id: int = 0, limit: Optional[int] = None) -> List[Dict]:
    """Retrieves entries from the sync log. See PromptsDatabase.get_sync_log_entries for details."""
    db = get_db_instance()
    return db.get_sync_log_entries(since_change_id, limit)

def delete_sync_log_entries(change_ids: List[int]) -> int:
    """Deletes entries from the sync log. See PromptsDatabase.delete_sync_log_entries for details."""
    db = get_db_instance()
    return db.delete_sync_log_entries(change_ids)


# --- Wrappers for Standalone Functions from Prompts_DB_v2 ---
# These functions from Prompts_DB_v2.py originally took a db_instance.
# Here, they use the globally managed _db_instance.

def add_or_update_prompt_interop(name: str, author: Optional[str], details: Optional[str],
                                 system_prompt: Optional[str] = None, user_prompt: Optional[str] = None,
                                 keywords: Optional[List[str]] = None
                                 ) -> Tuple[Optional[int], Optional[str], str]:
    """
    Adds a new prompt or updates an existing one (identified by name).
    If the prompt exists (even if soft-deleted), it will be updated/undeleted.
    This wraps the standalone add_or_update_prompt function from Prompts_DB_v2.
    """
    db = get_db_instance()
    return db_add_or_update_prompt(db, name, author, details, system_prompt, user_prompt, keywords)

def load_prompt_details_for_ui_interop(prompt_name: str) -> Tuple[str, str, str, str, str, str]:
    """
    Loads prompt details formatted for UI display.
    This wraps the standalone load_prompt_details_for_ui function from Prompts_DB_v2.
    """
    db = get_db_instance()
    return db_load_prompt_details_for_ui(db, prompt_name)

def export_prompt_keywords_to_csv_interop() -> Tuple[str, str]:
    """
    Exports prompt keywords to a CSV file.
    This wraps the standalone export_prompt_keywords_to_csv function from Prompts_DB_v2.
    """
    db = get_db_instance()
    return db_export_prompt_keywords_to_csv(db)

def view_prompt_keywords_markdown_interop() -> str:
    """
    Generates a Markdown representation of prompt keywords.
    This wraps the standalone view_prompt_keywords_markdown function from Prompts_DB_v2.
    """
    db = get_db_instance()
    return db_view_prompt_keywords_markdown(db)

def export_prompts_formatted_interop(export_format: str = 'csv',
                                     filter_keywords: Optional[List[str]] = None,
                                     include_system: bool = True,
                                     include_user: bool = True,
                                     include_details: bool = True,
                                     include_author: bool = True,
                                     include_associated_keywords: bool = True,
                                     markdown_template_name: Optional[str] = "Basic Template"
                                     ) -> Tuple[str, str]:
    """
    Exports prompts to a specified format (CSV or Markdown).
    This wraps the standalone export_prompts_formatted function from Prompts_DB_v2.
    """
    db = get_db_instance()
    return db_export_prompts_formatted(db, export_format, filter_keywords,
                                       include_system, include_user, include_details,
                                       include_author, include_associated_keywords,
                                       markdown_template_name)

# --- Mass Import Functionality ---

PROMPT_FIELDS = ["name", "author", "details", "system_prompt", "user_prompt", "keywords"]

def _normalize_prompt_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Ensures prompt data has all expected fields, defaulting to None or empty list."""
    normalized = {field: data.get(field) for field in PROMPT_FIELDS}
    if normalized["keywords"] is None:
        normalized["keywords"] = []
    if not isinstance(normalized["keywords"], list):
        logger.warning(f"Keywords for prompt '{normalized.get('name', 'Unknown')}' was not a list, converting. Value: {normalized['keywords']}")
        if isinstance(normalized["keywords"], str):
             normalized["keywords"] = [kw.strip() for kw in normalized["keywords"].split(',') if kw.strip()]
        else: # Attempt to cast to list, or empty if fails
            try:
                normalized["keywords"] = list(normalized["keywords"])
            except TypeError:
                normalized["keywords"] = []


    # Ensure specific string fields are strings or None
    for field in ["name", "author", "details", "system_prompt", "user_prompt"]:
        if normalized[field] is not None and not isinstance(normalized[field], str):
            logger.warning(f"Field '{field}' for prompt '{normalized.get('name', 'Unknown')}' was not a string, converting. Value: {normalized[field]}")
            try:
                normalized[field] = str(normalized[field])
            except Exception:
                logger.error(f"Could not convert field '{field}' to string for prompt '{normalized.get('name', 'Unknown')}'")
                normalized[field] = None # Or raise error
    return normalized

def parse_json_prompts_from_content(content: str) -> List[Dict[str, Any]]:
    """Parses JSON content into a list of prompt dictionaries."""
    try:
        data = json.loads(content)
        if isinstance(data, dict): # Single prompt object
            return [_normalize_prompt_data(data)]
        elif isinstance(data, list): # Array of prompt objects
            return [_normalize_prompt_data(item) for item in data if isinstance(item, dict)]
        else:
            raise ValueError("JSON content must be an object or an array of objects.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")
    except Exception as e:
        logger.error(f"Error processing JSON data: {e}")
        raise ValueError(f"Could not process JSON data: {e}")


def parse_yaml_prompts_from_content(content: str) -> List[Dict[str, Any]]:
    """Parses YAML content into a list of prompt dictionaries."""
    if not YAML_AVAILABLE:
        raise RuntimeError("YAML parsing is not available. Please install PyYAML.")
    try:
        data = list(yaml.safe_load_all(content)) # Handles multiple YAML documents in one file
        prompts = []
        if not data: # Handle empty yaml file
            return []
        if len(data) == 1 and isinstance(data[0], list): # A single document that is a list of prompts
             for item in data[0]:
                if isinstance(item, dict):
                    prompts.append(_normalize_prompt_data(item))
        else: # Multiple documents, or a single document that is a prompt object
            for item in data:
                if isinstance(item, dict):
                    prompts.append(_normalize_prompt_data(item))
        return prompts
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML format: {e}")
    except Exception as e:
        logger.error(f"Error processing YAML data: {e}")
        raise ValueError(f"Could not process YAML data: {e}")


def parse_markdown_prompts_from_content(content: str) -> List[Dict[str, Any]]:
    """
    Parses Markdown content with custom '### SECTION_NAME ###' headers
    into a list of prompt dictionaries.
    """
    prompts_data = []
    parsed_data: Dict[str, Any] = {
        "name": None, "author": None, "details": None,
        "system_prompt": None, "user_prompt": None, "keywords": [],
    }
    section_map = {
        "TITLE": "name", "AUTHOR": "author", "SYSTEM": "system_prompt",
        "USER": "user_prompt", "KEYWORDS": "keywords_str",
    }

    # Attempt to parse TITLE and DETAILS block first
    # This assumes details are between TITLE and AUTHOR sections, or after TITLE if AUTHOR isn't next.
    # Regex: ^\s*###\s*TITLE\s*###\s*\n(.*?)(\n\s*###\s*AUTHOR\s*###|\Z)
    # Group 1: Content after ### TITLE ### (name + details)
    # Group 2: The ### AUTHOR ### section or end of string (used as a delimiter)
    title_block_match = re.search(
        r"^\s*###\s*TITLE\s*###\s*\n(.*?)(?=(\n\s*###\s*AUTHOR\s*###|\Z))",
        content, re.MULTILINE | re.DOTALL | re.IGNORECASE
    )

    if title_block_match:
        title_and_details_content = title_block_match.group(1).strip()
        lines = title_and_details_content.split('\n', 1)
        parsed_data["name"] = lines[0].strip()
        if len(lines) > 1:
            parsed_data["details"] = lines[1].strip()
        else:
            parsed_data["details"] = ""  # No separate details lines
    else:  # Fallback if the specific TITLE...AUTHOR structure isn't found
        # Try to get just the TITLE content if available
        simple_title_match = re.search(
            r"^\s*###\s*TITLE\s*###\s*\n(.*?)(?=(\n\s*###|$))",
            content, re.MULTILINE | re.DOTALL | re.IGNORECASE
        )
        if simple_title_match:
            parsed_data["name"] = simple_title_match.group(1).strip()
            # In this fallback, details might be harder to reliably extract without more rules
            # For now, if we only find a TITLE section, details will be None or empty.

    # If no name was found, we can't proceed for this prompt format
    if not parsed_data.get("name"):
        logger.warning(
            f"Custom MD/TXT Parser: No 'TITLE' section found or it was empty. Content snippet: {content[:200]}")
        return []  # Return empty list if no title, as a valid prompt needs a name

    # Extract other sections
    for section_header, dict_key in section_map.items():
        if section_header == "TITLE":  # Already handled
            continue

        pattern = rf"^\s*###\s*{section_header}\s*###\s*\n(.*?)(?=(?:\n\s*###|$))"
        match = re.search(pattern, content, re.MULTILINE | re.DOTALL | re.IGNORECASE)
        if match:
            section_text = match.group(1).strip()
            if section_text:  # Only assign if there's actual content
                parsed_data[dict_key] = section_text

    # Post-process keywords
    if "keywords_str" in parsed_data and parsed_data["keywords_str"]:
        parsed_data["keywords"] = [kw.strip() for kw in parsed_data["keywords_str"].split(',') if kw.strip()]
    if "keywords_str" in parsed_data:  # Clean up temporary key
        del parsed_data["keywords_str"]

    prompts_data.append(_normalize_prompt_data(parsed_data))
    logger.debug(f"Parsed custom MD/TXT prompt: {parsed_data.get('name')}")

    return prompts_data


def parse_txt_prompts_from_content(content: str) -> List[Dict[str, Any]]:
    """
    Parses TXT content into a list of prompt dictionaries,
    assuming the same '### SECTION_NAME ###' format as Markdown.
    """
    logger.debug("Parsing TXT file using custom Markdown section logic.")
    # Simply call the markdown parser which now handles the ### SECTION ### format
    return parse_markdown_prompts_from_content(content)

def _get_file_type(file_path: Path) -> Optional[str]:
    """Determines file type from extension."""
    suffix = file_path.suffix.lower()
    if suffix == ".json":
        return "json"
    elif suffix in [".yaml", ".yml"]:
        return "yaml"
    elif suffix == ".md":
        return "markdown"
    elif suffix == ".txt":
        return "txt"
    return None

def import_prompts_from_files(
    file_paths: Union[str, Path, List[Union[str, Path]]],
    base_directory: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Imports prompts from one or more files (JSON, YAML, Markdown, TXT).

    Each file can contain one or multiple prompts according to its format's conventions.
    Prompts are added or updated in the database based on their 'name'.

    Args:
        file_paths: A single file path (str or Path) or a list of file paths.
        base_directory: Optional base directory to restrict file access to. If None,
                       uses a safe default directory.

    Returns:
        A list of dictionaries, where each dictionary represents the result of
        an attempted import operation for a single prompt. Each dict contains:
        - "file_path": str, the path of the source file.
        - "prompt_name": Optional[str], name of the prompt if parsable.
        - "status": str, "success" or "failure".
        - "message": str, details about the operation (e.g., error message or success info).
        - "prompt_id": Optional[int], ID of the prompt if successfully added/updated.
        - "prompt_uuid": Optional[str], UUID of the prompt if successfully added/updated.
    """
    if not is_initialized():
        msg = "Prompts Interop Library not initialized. Call initialize_interop() first."
        logger.error(msg)
        # Or raise RuntimeError(msg) if preferred to enforce initialization strictly before this call
        return [{"file_path": str(fp), "status": "failure", "message": msg} for fp in ([file_paths] if isinstance(file_paths, (str, Path)) else file_paths)]


    if isinstance(file_paths, (str, Path)):
        file_paths = [Path(file_paths)]
    else:
        file_paths = [Path(fp) for fp in file_paths]

    # Set default base directory for prompt files
    if base_directory is None:
        base_directory = os.path.expanduser("~/.config/tldw_cli/prompts/")

    results: List[Dict[str, Any]] = []
    parser_map: Dict[str, Callable[[str], List[Dict[str, Any]]]] = {
        "json": parse_json_prompts_from_content,
        "yaml": parse_yaml_prompts_from_content,
        "markdown": parse_markdown_prompts_from_content,
        "txt": parse_txt_prompts_from_content,
    }

    for file_path in file_paths:
        file_path_str = str(file_path)
        logger.info(f"Processing import file: {file_path_str}")
        
        # Validate the file path to prevent directory traversal
        try:
            validated_path = validate_path(file_path_str, base_directory)
            logger.debug(f"Validated prompt file path: {validated_path}")
        except ValueError as e:
            msg = f"Invalid file path '{file_path_str}': {e}"
            logger.error(msg)
            results.append({"file_path": file_path_str, "status": "failure", "message": msg})
            continue
        
        if not validated_path.exists() or not validated_path.is_file():
            msg = f"File not found or is not a regular file: {validated_path}"
            logger.error(msg)
            results.append({"file_path": file_path_str, "status": "failure", "message": msg})
            continue

        file_type = _get_file_type(validated_path)
        if not file_type:
            msg = f"Unsupported file type or unknown extension: {validated_path}"
            logger.warning(msg)
            results.append({"file_path": file_path_str, "status": "failure", "message": msg})
            continue

        parser = parser_map.get(file_type)
        if not parser: # Should not happen if _get_file_type returns a valid key
            msg = f"No parser available for file type '{file_type}': {validated_path}"
            logger.error(msg) # This would be an internal logic error
            results.append({"file_path": file_path_str, "status": "failure", "message": msg})
            continue

        try:
            with open(validated_path, "r", encoding="utf-8") as f:
                content = f.read()
            parsed_prompts = parser(content)
        except RuntimeError as e: # For unavailable parsers (PyYAML/python-frontmatter not installed)
             logger.error(f"Parser runtime error for {file_path_str}: {e}")
             results.append({"file_path": file_path_str, "status": "failure", "message": str(e)})
             continue
        except ValueError as e: # For parsing errors (invalid format)
            logger.error(f"Failed to parse {file_path_str}: {e}")
            results.append({"file_path": file_path_str, "status": "failure", "message": f"Parsing error: {e}"})
            continue
        except Exception as e:
            logger.error(f"Unexpected error reading or parsing {file_path_str}: {e}", exc_info=True)
            results.append({"file_path": file_path_str, "status": "failure", "message": f"Read/Parse error: {e}"})
            continue

        if not parsed_prompts:
            logger.info(f"No prompts found or parsed in file: {file_path_str}")
            # Optionally add a result indicating no prompts found
            # results.append({"file_path": file_path_str, "status": "skipped", "message": "No prompts found in file."})

        for prompt_data in parsed_prompts:
            prompt_name = prompt_data.get("name")
            if not prompt_name or not isinstance(prompt_name, str) or not prompt_name.strip():
                msg = "Prompt data is missing a valid 'name'."
                logger.warning(f"{msg} File: {file_path_str}, Data: {str(prompt_data)[:100]}")
                results.append({
                    "file_path": file_path_str,
                    "prompt_name": prompt_name,
                    "status": "failure",
                    "message": msg
                })
                continue

            try:
                # Ensure all expected fields are present, defaulting to None or []
                # This is now handled by _normalize_prompt_data within each parser.
                p_id, p_uuid, db_msg = add_or_update_prompt_interop(
                    name=prompt_name,
                    author=prompt_data.get("author"),
                    details=prompt_data.get("details"),
                    system_prompt=prompt_data.get("system_prompt"),
                    user_prompt=prompt_data.get("user_prompt"),
                    keywords=prompt_data.get("keywords")
                )
                logger.info(f"Imported prompt '{prompt_name}' from {file_path_str}: {db_msg}")
                results.append({
                    "file_path": file_path_str,
                    "prompt_name": prompt_name,
                    "status": "success",
                    "message": db_msg,
                    "prompt_id": p_id,
                    "prompt_uuid": p_uuid
                })
            except (InputError, ConflictError, DatabaseError, SchemaError) as e:
                logger.error(f"Failed to add/update prompt '{prompt_name}' from {file_path_str}: {e}")
                results.append({
                    "file_path": file_path_str,
                    "prompt_name": prompt_name,
                    "status": "failure",
                    "message": f"Database error: {type(e).__name__} - {e}"
                })
            except Exception as e:
                logger.error(f"Unexpected error importing prompt '{prompt_name}' from {file_path_str}: {e}", exc_info=True)
                results.append({
                    "file_path": file_path_str,
                    "prompt_name": prompt_name,
                    "status": "failure",
                    "message": f"Unexpected error: {type(e).__name__} - {e}"
                })
    return results


# --- Expose Exceptions for API layer ---
# These are already imported at the top: DatabaseError, SchemaError, InputError, ConflictError
# They can be caught by the calling code (e.g., your API endpoint) like:
#   try:
#       prompts_interop.add_prompt(...)
#   except prompts_interop.InputError as e:
#       # handle bad input
#   except prompts_interop.ConflictError as e:
#       # handle conflict
#   except prompts_interop.DatabaseError as e:
#       # handle general DB error


if __name__ == '__main__':
    # Example Usage (primarily for testing the interop layer)
    # Ensure Prompts_DB_v2.py is in the same directory or Python path

    # Setup basic logging for the example
    #logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Running prompts_interop.py example usage...")

    # --- Configuration ---
    # Use an in-memory database for this example for easy cleanup
    # For a real application, this would be a file path.
    EXAMPLE_DB_PATH = ":memory:"
    EXAMPLE_CLIENT_ID = "interop_example_client_v2"
    temp_dir = tempfile.mkdtemp()
    logger.info(f"Using temporary directory for import files: {temp_dir}")

    # Sample prompt data
    p_id1, p_uuid1, msg1 = add_prompt(
        name="My First Prompt",
        author="Interop Test",
        details="This is a test prompt added via interop.",
        system_prompt="You are a helpful assistant.",
        user_prompt="Tell me a joke.",
        keywords=["test", "funny"]  # "funny" will be newly created
    )
    sample_prompts_content = {
        "json_single": """
        {
            "name": "JSON Single Prompt",
            "author": "JSON Importer",
            "details": "A single prompt from a JSON file.",
            "system_prompt": "You are a JSON expert.",
            "user_prompt": "Explain JSON structure.",
            "keywords": ["json", "single"]
        }
        """,
        "json_multi": """
        [
            {
                "name": "JSON Multi Prompt 1",
                "author": "JSON Importer",
                "details": "First prompt in a JSON array.",
                "system_prompt": "System 1",
                "user_prompt": "User 1",
                "keywords": ["json", "multi", "one"]
            },
            {
                "name": "JSON Multi Prompt 2",
                "author": "JSON Importer",
                "details": "Second prompt in a JSON array.",
                "system_prompt": "System 2",
                "user_prompt": "User 2",
                "keywords": ["json", "multi", "two"]
            }
        ]
        """,
        "yaml_single": """
name: YAML Single Prompt
author: YAML Importer
details: A single prompt from a YAML file.
system_prompt: You are a YAML guru.
user_prompt: Explain YAML syntax.
keywords: [yaml, single]
        """,
        "yaml_multi_doc": """
name: YAML Multi-Doc Prompt 1
author: YAML Importer
details: First prompt in a multi-document YAML file.
system_prompt: System YAML 1
user_prompt: User YAML 1
keywords: [yaml, multidoc, one]
---
name: YAML Multi-Doc Prompt 2
author: YAML Importer
details: Second prompt in a multi-document YAML file.
system_prompt: System YAML 2
user_prompt: User YAML 2
keywords: [yaml, multidoc, two]
        """,
        "yaml_list_doc": """
- name: YAML List-Doc Prompt 1
  author: YAML Importer
  details: First prompt in a list YAML document.
  system_prompt: System YAML List 1
  user_prompt: User YAML List 1
  keywords: [yaml, listdoc, one]
- name: YAML List-Doc Prompt 2
  author: YAML Importer
  details: Second prompt in a list YAML document.
  system_prompt: System YAML List 2
  user_prompt: User YAML List 2
  keywords: [yaml, listdoc, two]
        """,
        "md_single": """
---
name: Markdown Single Prompt
author: MD Importer
details: A single prompt from a Markdown file.
keywords: [markdown, single, frontmatter]
---
## System Prompt
You are a Markdown professional.

## User Prompt
How to make a table in Markdown?
        """,
        "md_multi": """
---
name: Markdown Multi Prompt 1
author: MD Importer
details: First prompt in a multi-prompt MD file.
keywords: [markdown, multi, one]
---
## System Prompt
System MD 1

## User Prompt
User MD 1
---
---
name: Markdown Multi Prompt 2
author: MD Importer
details: Second prompt in a multi-prompt MD file.
keywords: [markdown, multi, two]
---
## System Prompt
System MD 2. Case test for System prompt.
## USER PROMPT
User MD 2. Case test for User prompt.
        """,
        "txt_single": """
Name: TXT Single Prompt
Author: TXT Importer
Keywords: txt, single, simple
Details: A single prompt from a TXT file.
This is more details.
---SYSTEM---
You are a TXT file reader.
This is the system prompt for TXT.
---USER---
How to parse TXT files?
This is the user prompt for TXT.
        """,
        "txt_multi": """
Name: TXT Multi Prompt 1
Author: TXT Importer
Keywords: txt, multi, one
Details: First prompt in a multi-prompt TXT.
---SYSTEM---
System TXT 1
---USER---
User TXT 1
---
Name: TXT Multi Prompt 2
Author: TXT Importer
Keywords: txt, multi, two
Details: Second prompt in a multi-prompt TXT.
---SYSTEM---
System TXT 2
---USER---
User TXT 2
        """,
        "txt_no_name": """
Author: TXT Importer
Keywords: txt, error, no_name
Details: This prompt is missing a name.
---SYSTEM---
System for no name.
---USER---
User for no name.
        """
    }

    # Create temporary files
    file_paths_to_import = []
    for name_key, content in sample_prompts_content.items():
        ext_map = {"json": ".json", "yaml": ".yaml", "md": ".md", "txt": ".txt"}
        file_ext = ""
        for k, v in ext_map.items():
            if name_key.startswith(k):
                file_ext = v
                break
        if not file_ext:
            logger.warning(f"Could not determine extension for {name_key}, skipping temp file creation.")
            continue

        temp_file_path = Path(temp_dir) / f"{name_key}{file_ext}"
        with open(temp_file_path, "w", encoding="utf-8") as f:
            f.write(content.strip())
        file_paths_to_import.append(temp_file_path)
        logger.info(f"Created temp file: {temp_file_path}")

    # Add a non-existent file and an unsupported file type for error testing
    file_paths_to_import.append(Path(temp_dir) / "non_existent_file.json")
    unsupported_file_path = Path(temp_dir) / "unsupported.doc"
    with open(unsupported_file_path, "w") as f: f.write("This is a doc file.")
    file_paths_to_import.append(unsupported_file_path)


    try:
        # 1. Initialize
        logging.info("\n--- Initializing Interop Library ---")
        initialize_interop(db_path=EXAMPLE_DB_PATH, client_id=EXAMPLE_CLIENT_ID)
        logging.info(f"Interop initialized: {is_initialized()}")
        logging.info(f"DB instance client ID: {get_db_instance().client_id}")
        # 2. Add some data using interop functions
        logging.info("\n--- Adding Data ---")
        kw_id1, kw_uuid1 = add_keyword("test")
        logging.info(f"Added keyword 'test': ID {kw_id1}, UUID {kw_uuid1}")
        kw_id2, kw_uuid2 = add_keyword("example")
        logging.info(f"Added keyword 'example': ID {kw_id2}, UUID {kw_uuid2}")

        p_id1, p_uuid1, msg1 = add_prompt(
            name="My First Prompt",
            author="Interop Test",
            details="This is a test prompt added via interop.",
            system_prompt="You are a helpful assistant.",
            user_prompt="Tell me a joke.",
            keywords=["test", "funny"]  # "funny" will be newly created
        )
        logging.info(f"Added prompt: {msg1} (ID: {p_id1}, UUID: {p_uuid1})")

        p_id2, p_uuid2, msg2 = add_or_update_prompt_interop(
            name="My Second Prompt",
            author="Interop Test",
            details="Another test prompt.",
            system_prompt="You are a creative writer.",
            user_prompt="Write a short story.",
            keywords=["example", "story"]
        )
        logging.info(f"Added/Updated prompt via interop wrapper: {msg2} (ID: {p_id2}, UUID: {p_uuid2})")

        # 3. Read data
        logging.info("\n--- Reading Data ---")
        prompt1_details = fetch_prompt_details(p_id1)
        if prompt1_details:
            logging.info(f"Details for Prompt ID {p_id1} ('{prompt1_details.get('name')}'):")
            logging.info(f"  Author: {prompt1_details.get('author')}")
            logging.info(f"  Keywords: {prompt1_details.get('keywords')}")
        else:
            logging.info(f"Could not fetch details for Prompt ID {p_id1}")

        # --- Test Mass Import ---
        logger.info("\n--- Testing Mass Import ---")
        if not YAML_AVAILABLE:
            logger.warning("YAML files will be skipped as PyYAML is not installed.")
        if not FRONTMATTER_AVAILABLE:
            logger.warning("Markdown files will be skipped as python-frontmatter is not installed.")

        import_results = import_prompts_from_files(file_paths_to_import)

        logger.info("\n--- Import Results ---")
        successful_imports = 0
        failed_imports = 0
        for res in import_results:
            logger.info(
                f"File: {res['file_path']}, Prompt: '{res.get('prompt_name', 'N/A')}', "
                f"Status: {res['status']}"
            )
            if res['status'] == 'success':
                logger.info(f"  Msg: {res['message']}, ID: {res.get('prompt_id')}, UUID: {res.get('prompt_uuid')}")
                successful_imports += 1
            else:
                logger.error(f"  Error: {res['message']}")
                failed_imports += 1

        logger.info(f"\nTotal Successful Imports: {successful_imports}")
        logger.info(f"Total Failed Imports/Attempts: {failed_imports}")

        # Verify some imported prompts
        logger.info("\n--- Verifying Imported Prompts ---")
        if is_initialized(): # Check again, in case init failed in a test setup
            test_prompt_name = "JSON Single Prompt" # Should have been imported
            details = fetch_prompt_details(test_prompt_name)
            if details:
                logger.info(f"Successfully fetched imported prompt '{test_prompt_name}': Author - {details.get('author')}, Keywords - {details.get('keywords')}")
            else:
                logger.error(f"Could not fetch imported prompt '{test_prompt_name}'.")

        all_prompts, total_pages, _, total_items = list_prompts()
        logging.info(f"List Prompts (Page 1): {len(all_prompts)} items. Total items: {total_items}, Total pages: {total_pages}")
        for p in all_prompts:
            logging.info(f"  - {p.get('name')} (Author: {p.get('author')})")

        all_kws = fetch_all_keywords()
        logging.info(f"All active keywords: {all_kws}")


        # 4. Search
        logging.info("\n--- Searching Data ---")
        search_results, total_matches = search_prompts(search_query="test", search_fields=["details", "keywords"])
        logging.info(f"Search results for 'test': {len(search_results)} matches (Total found: {total_matches})")
        for res in search_results:
            logging.info(f"  - Found: {res.get('name')} (Keywords: {res.get('keywords')})")


        # 5. Using other interop-wrapped standalone functions
        logging.info("\n--- Using Other Interop Functions ---")
        markdown_keywords = view_prompt_keywords_markdown_interop()
        logging.info("Keywords in Markdown:")
        logging.info(markdown_keywords)

        csv_export_status, csv_file_path = export_prompts_formatted_interop(export_format='csv')
        logging.info(f"CSV Export: {csv_export_status} -> {csv_file_path}")
        if csv_file_path != "None" and EXAMPLE_DB_PATH == ":memory:":
             logging.info(f" (Note: CSV file '{csv_file_path}' would exist if not using in-memory DB)")


        # 6. Soft delete
        logging.info("\n--- Soft Deleting ---")
        if p_id1:
            deleted = soft_delete_prompt(p_id1)
            logging.info(f"Soft deleted prompt ID {p_id1}: {deleted}")
            prompt1_after_delete = get_prompt_by_id(p_id1)
            logging.info(f"Prompt ID {p_id1} after delete (should be None): {prompt1_after_delete}")
            prompt1_deleted_rec = get_prompt_by_id(p_id1, include_deleted=True)
            logging.info(f"Prompt ID {p_id1} after delete (fetching deleted, should exist): {prompt1_deleted_rec is not None}")


        # 7. Sync Log (Example)
        logging.info("\n--- Sync Log ---")
        sync_entries = get_sync_log_entries(limit=5)
        logging.info(f"First 5 sync log entries:")
        for entry in sync_entries:
            logging.info(f"  ID: {entry['change_id']}, Entity: {entry['entity']}, Op: {entry['operation']}, UUID: {entry['entity_uuid']}")
        if sync_entries:
            deleted_count = delete_sync_log_entries([e['change_id'] for e in sync_entries])
            logging.info(f"Deleted {deleted_count} sync log entries.")


    except (DatabaseError, SchemaError, InputError, ConflictError, RuntimeError, ValueError) as e:
        logger.error(f"An error occurred during interop example: {type(e).__name__} - {e}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {type(e).__name__} - {e}", exc_info=True)
    finally:
        logger.info("\n--- Shutting Down Interop Library ---")
        shutdown_interop()
        logger.info(f"Interop initialized after shutdown: {is_initialized()}")

        # Clean up temporary files and directory
        logger.info(f"Cleaning up temporary directory: {temp_dir}")
        for fp_str in file_paths_to_import:
            fp = Path(fp_str)
            if fp.exists():
                try:
                    os.remove(fp)
                except OSError as e:
                    logger.error(f"Error removing temp file {fp}: {e}")
        try:
            os.rmdir(temp_dir)
            logger.info(f"Successfully removed temporary directory: {temp_dir}")
        except OSError as e:
            logger.error(f"Error removing temporary directory {temp_dir}: {e}")


    logger.info("Prompts_interop.py example usage finished.")

#
# End of prompts_interop.py
#######################################################################################################################
