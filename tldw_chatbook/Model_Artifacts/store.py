"""Process-wide managed model store location and service construction.

Only the synchronous service layer is imported at module scope so worker-side
imports do not pull in the acquisition or fetch runtimes.
"""

from __future__ import annotations

from pathlib import Path

from .service import ModelArtifactService


def managed_model_artifact_root() -> Path:
    """Return the shared managed-model store root.

    Returns:
        The absolute path to the shared managed-model store root.
    """
    from tldw_chatbook.Utils.paths import get_user_data_dir

    return get_user_data_dir() / "models" / "managed"


def managed_service(root: Path | None = None) -> ModelArtifactService:
    """Return a service bound to the managed-model store.

    Args:
        root: Optional store-root override, primarily for isolated tests.

    Returns:
        A service rooted at ``root`` or the process-wide managed store.
    """
    return ModelArtifactService(root or managed_model_artifact_root())
