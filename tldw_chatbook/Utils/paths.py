# paths.py
# Description: This file contains functions to manage file paths and directories for the tldw_cli application.
#
# Imports
import logging
import os
from pathlib import Path
from typing import Union, AnyStr
#
# 3rd-party Libraries
#
# Local Imports
# Remove non-existent imports
try:
    from ..Utils.Utils import PROJECT_DATABASES_DIR, log, PROJECT_ROOT_DIR, CONFIG_FILE_PATH, USER_DB_PATH, \
        USER_DB_DIR
except ImportError:
    # Set defaults if imports fail
    PROJECT_DATABASES_DIR = None
    log = logging
    PROJECT_ROOT_DIR = None
    CONFIG_FILE_PATH = None 
    USER_DB_PATH = None
    USER_DB_DIR = None
#
#######################################################################################################################
#
# Functions:

# Remove ALL PRINT statements from this file, replacing them with appropriate logging calls.


def get_project_root():
    """Get the absolute path to the project root directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    logging.info(f"Project root: {project_root}")
    return project_root


def get_project_databases_dir() -> Path:
    """
    Get the absolute path to the 'Databases' directory within the project structure.
    Ensures the directory exists.

    Returns:
        pathlib.Path: The absolute Path object for the project's Databases directory.
    """
    try:
        PROJECT_DATABASES_DIR.mkdir(parents=True, exist_ok=True)
        log.info(f"Ensured project-internal database directory exists: {PROJECT_DATABASES_DIR}")
        return PROJECT_DATABASES_DIR
    except OSError as e:
        log.error(f"Could not create or access project-internal database directory {PROJECT_DATABASES_DIR}: {e}", exc_info=True)
        raise OSError(f"Failed to ensure project-internal database directory exists at {PROJECT_DATABASES_DIR}") from e


def get_project_database_path(db_filename: str) -> Path:
    """
    Get the full absolute path for a database file stored within the
    project's 'Databases' directory (e.g., for templates, tests).

    Args:
        db_filename (str): The base name of the database file (e.g., 'test.db').
                           Directory components will be ignored for safety.

    Returns:
        pathlib.Path: The absolute Path object for the database file.
    """
    # Ensure we only use the filename part to prevent traversal
    safe_db_filename = Path(db_filename).name
    if not safe_db_filename:
        raise ValueError("db_filename cannot be empty or represent a directory.")

    # Get the project DB directory (ensuring it exists)
    project_db_dir = get_project_databases_dir()
    full_path = project_db_dir / safe_db_filename
    log.debug(f"Returning project-internal database path for '{safe_db_filename}': {full_path}")
    return full_path


# --- General Purpose Path Helper ---

def get_project_relative_path(relative_path_str: Union[str, os.PathLike[AnyStr]]) -> Path:
    """
    Resolves a path relative to the project root directory.

    Args:
        relative_path_str (Union[str, os.PathLike[AnyStr]]): The path string relative
            to the project root (e.g., "Assets/image.png", "Data/file.json").

    Returns:
        pathlib.Path: The absolute Path object.
    """
    # Note: Path() handles PathLike objects correctly
    # Using '/' operator joins paths appropriately
    absolute_path = (PROJECT_ROOT_DIR / relative_path_str).resolve()
    log.debug(f"Resolved project relative path for '{relative_path_str}': {absolute_path}")
    return absolute_path

def get_user_data_dir() -> Path:
    """
    Get the user data directory for the application.
    Creates the directory if it doesn't exist.
    
    Returns:
        Path to the user data directory
    """
    # Import here to avoid circular imports
    from ..config import get_user_data_dir as get_user_data_dir_from_config
    return get_user_data_dir_from_config()


# --- Example Usage within Utils.py (for testing) ---
if __name__ == '__main__':
    #logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s:%(name)s] %(message)s')
    print("\n--- Testing Utility Functions ---")

    print(f"\nProject Root: {get_project_root()}")
    print(f"Config File Path (Constant): {CONFIG_FILE_PATH}")
    print(f"User DB Path (Constant): {USER_DB_PATH}")

    try:
        # Create dummy config for loading test
        if not CONFIG_FILE_PATH.exists():
             print(f"Creating dummy config at {CONFIG_FILE_PATH} for test...")
             CONFIG_FILE_PATH.write_text("[Settings]\nvalue = test\n", encoding='utf-8')
        config_data = load_comprehensive_config()
        print("\nLoaded Config Sections:", config_data.sections())
        # Clean up dummy config if created just for test
        # if CONFIG_FILE_PATH.read_text() == "[Settings]\nvalue = test\n":
        #     CONFIG_FILE_PATH.unlink()
        #     print("Cleaned up dummy config.")
    except Exception as e:
        print(f"\nError loading config: {e}")

    try:
        user_db = get_user_database_path()
        print(f"\nUser Database Path (Ensured Dir): {user_db}")
    except Exception as e:
        print(f"\nError getting user database path: {e}")

    try:
        proj_db_dir = get_project_databases_dir()
        print(f"\nProject Databases Dir (Ensured): {proj_db_dir}")
        proj_db_file = get_project_database_path("template.db")
        print(f"Example Project DB Path: {proj_db_file}")
        # Test with unsafe path
        try:
             get_project_database_path("../outside.db")
        except ValueError:
             print("Correctly prevented path traversal for project DB.")
    except Exception as e:
        print(f"\nError with project database paths: {e}")


    try:
        asset_path = get_project_relative_path("Assets/logo.png")
        print(f"\nExample Relative Path: {asset_path}")
        data_path = get_project_relative_path(Path("Data") / "config.json")
        print(f"Example Relative Path (using Path): {data_path}")
    except Exception as e:
        print(f"\nError with relative path resolution: {e}")

    print("\n--- End Testing ---")


def get_project_root() -> Path:
    """
    Returns the absolute path to the project root directory (containing app.py).

    Returns:
        pathlib.Path: The absolute Path object for the project root.
    """
    # This is now determined as a constant at the top
    log.debug(f"Returning project root: {PROJECT_ROOT_DIR}")
    return PROJECT_ROOT_DIR


def get_user_database_path(username: str = None) -> Path:
    """
    Returns the absolute path to the user's primary database file.
    If username is provided, it creates a user-specific database file.
    Otherwise, it uses the default database path.
    
    Args:
        username: Optional username for user-specific database paths.
                 If None, uses the default database path.

    Returns:
        pathlib.Path: The absolute Path object for the user database file.
    """
    try:
        # Ensure the directory ~/.config/tldw_cli exists
        USER_DB_DIR.mkdir(parents=True, exist_ok=True)
        log.info(f"Ensured user database directory exists: {USER_DB_DIR}")
        
        if username:
            # Create a sanitized filename for the user-specific database
            safe_username = "".join(c for c in username if c.isalnum() or c in ('_', '-'))
            if not safe_username:
                safe_username = "default"
            user_db_path = USER_DB_DIR / f"tldw_cli_{safe_username}.db"
            log.debug(f"Returning user-specific database path for '{username}': {user_db_path}")
            return user_db_path
        else:
            # Use the default database path
            log.debug(f"Returning default user database path: {USER_DB_PATH}")
            return USER_DB_PATH
    except OSError as e:
        log.error(f"Could not create or access user database directory {USER_DB_DIR}: {e}", exc_info=True)
        # Depending on requirements, you might want to raise an exception here
        # or return None, or let the subsequent DB access fail.
        # For now, let's re-raise to make the problem explicit.
        raise OSError(f"Failed to ensure user database directory exists at {USER_DB_DIR}") from e

#
# End of paths.py
#######################################################################################################################
