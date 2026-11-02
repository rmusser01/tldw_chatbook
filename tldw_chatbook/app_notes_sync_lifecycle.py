"""App-owned safe Notes runtime construction; invoked only by native readers."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .Notes.notes_sync_runtime import NotesSyncRuntimeOwner


def _read_notes_sync_runtime_owner_safe(self) -> "NotesSyncRuntimeOwner | None":
    """Read the runtime owner WITHOUT letting a build failure vanish.

    TASK-21249: the Library screen used to read this through
    ``getattr(app, "notes_sync_runtime_owner", None)`` -- which is right
    for harness doubles that lack the property, but once the name became
    a lazily-BUILDING property the getattr default also swallowed an
    exception raised INSIDE the build, painting a construction failure
    to the user as "awaiting cutover" (feature not available yet).
    This reader distinguishes the two: a real build failure is recorded
    on ``_notes_sync_runtime_build_failure`` (and logged once) so the UI
    can show a distinct, honest state; the owner slot stays None.

    Returns:
        The built (or substituted) owner, or None when the property is
        absent, unbuildable, or previously failed to build.
    """
    from tldw_chatbook import app_service_wiring as _owner

    # Screen preparation and Notes attention may arrive together. Keep the
    # failure check and publication serialized as well as construction;
    # otherwise a waiter can retry a failed build and leave a live owner
    # shadowed by the first caller's sticky failure.
    read_lock = self.__dict__.setdefault(
        "_notes_sync_runtime_safe_read_lock", _owner.threading.Lock()
    )
    with read_lock:
        if self.__dict__.get("_notes_sync_runtime_build_failure") is not None:
            return None
        descriptor = getattr(type(self), "notes_sync_runtime_owner", None)
        if not isinstance(descriptor, property):
            return getattr(self, "notes_sync_runtime_owner", None)
        try:
            return self.notes_sync_runtime_owner
        except Exception as error:  # noqa: BLE001 - UI read must not crash
            self._notes_sync_runtime_build_failure = error
            _owner.logger.opt(exception=True).error(
                "notes-sync runtime construction failed; the Library screen "
                "will report a runtime error rather than feature-unavailable."
            )
            return None
