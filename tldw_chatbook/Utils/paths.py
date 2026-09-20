# paths.py
# Description: This file contains functions to manage file paths and directories for the tldw_cli application.
#
# The project-* helpers (get_project_root/get_project_databases_dir/
# get_project_database_path/get_project_relative_path) and their PROJECT_*/
# CONFIG_FILE_PATH constants were removed in TASK-32807.6: they had zero
# importers, and the `from ..Utils.Utils import PROJECT_DATABASES_DIR, ...`
# they depended on always raised ImportError (those names never existed),
# so every one of them fell through to `None` and would have raised on use.
# `get_user_data_dir` (the one live export, 620+ callers) delegates to config.
#
from pathlib import Path


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


#
# End of paths.py
#######################################################################################################################
