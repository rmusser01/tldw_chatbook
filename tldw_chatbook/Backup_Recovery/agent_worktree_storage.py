"""Independent ordinary admission for the fixed profile worktree directory."""

from pathlib import Path

from tldw_chatbook.Utils.private_paths import secure_private_directory

from . import storage_admission as storage


def prepare_agent_worktrees_base() -> Path:
    """Prepare the selected profile's private worktree root, retaining DB custody.

    Returns:
        The existing code-owned agent worktree directory.

    Raises:
        RecoveryRequired: If fresh storage admission or parent custody refuses.
        PrivatePathError: If the native private-directory checks refuse.
    """
    from tldw_chatbook.config import get_user_data_dir

    core = getattr(storage._operation_local, "operation", None)
    if core is not None:
        storage._check_operation(core, core.path)
    # Retain the counted DB operation; it grants no filesystem path authority.
    storage._operation_local.operation = None
    try:
        base = get_user_data_dir() / "agent_worktrees"
        secure_private_directory(base, create=True, application_owned=True)
        return base
    finally:
        storage._operation_local.operation = core
        if core is not None:
            storage._check_operation(core, core.path)
