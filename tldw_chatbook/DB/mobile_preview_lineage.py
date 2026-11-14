"""Canonical compatibility exports for profile-owned preview preflight.

The complete operation lives with profile selection. Native DB owner modules
still own admitted connections, source checks, finite bounds and physical close.
"""

from tldw_chatbook.Backup_Recovery.profile_paths import (
    _GUIDANCE as _GUIDANCE,  # noqa: PLC0414 - canonical compatibility export
)
from tldw_chatbook.Backup_Recovery.profile_paths import (
    IncompatiblePreviewError as IncompatiblePreviewError,  # noqa: PLC0414 - canonical compatibility export
)
from tldw_chatbook.Backup_Recovery.profile_paths import (
    _existing_preview_target as _existing_preview_target,  # noqa: PLC0414 - canonical compatibility export
)
from tldw_chatbook.Backup_Recovery.profile_paths import (
    _reject_preview_connection as _reject_preview_connection,  # noqa: PLC0414 - canonical compatibility export
)
from tldw_chatbook.Backup_Recovery.profile_paths import (
    reject_incompatible_console_stores as reject_incompatible_console_stores,  # noqa: PLC0414 - canonical compatibility export
)
