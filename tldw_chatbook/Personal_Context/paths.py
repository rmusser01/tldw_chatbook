"""Default Personal Context storage paths."""

from pathlib import Path

from tldw_chatbook.config import get_user_data_dir


def get_personal_context_db_path() -> Path:
    """Return the dedicated database directly below the secured user data dir."""

    return get_user_data_dir() / "tldw_chatbook_personal_context.db"
