"""Pure session and responsive-layout state for the Library Media reader."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from .library_adaptive_reader_state import (
    ITEMS_MAX_WIDTH as ITEMS_MAX_WIDTH,
    ITEMS_MIN_WIDTH as ITEMS_MIN_WIDTH,
    ITEMS_TARGET_WIDTH as ITEMS_TARGET_WIDTH,
    LAYOUT_HYSTERESIS_WIDTH as LAYOUT_HYSTERESIS_WIDTH,
    LIBRARY_MAX_WIDTH as LIBRARY_MAX_WIDTH,
    LIBRARY_MIN_WIDTH as LIBRARY_MIN_WIDTH,
    LIBRARY_TARGET_WIDTH as LIBRARY_TARGET_WIDTH,
    PANE_GRIP_WIDTH as PANE_GRIP_WIDTH,
    READER_COMFORT_WIDTH as READER_COMFORT_WIDTH,
    AdaptiveReaderEffectiveLayout,
    AdaptiveReaderLayoutPreferences,
    AdaptiveReaderLayoutProfile,
    PaneName,
    normalize_adaptive_reader_preferences,
    resolve_adaptive_reader_layout,
)

SELECTION_SETTLE_SECONDS = 0.12

ReaderMode = Literal["read", "analysis", "highlights", "info"]
BackingMediaId = int | str

MediaReaderLayoutPreferences = AdaptiveReaderLayoutPreferences
MediaReaderEffectiveLayout = AdaptiveReaderEffectiveLayout
MEDIA_READER_LAYOUT_PROFILE = AdaptiveReaderLayoutProfile()
normalize_media_reader_preferences = normalize_adaptive_reader_preferences


def _validate_media_identity(canonical_id: str, backing_id: BackingMediaId) -> None:
    if not isinstance(canonical_id, str):
        raise TypeError("canonical media id must be a string.")
    if type(backing_id) is int:
        if backing_id < 1:
            raise ValueError("backing media id must be positive.")
    elif not isinstance(backing_id, str) or not backing_id.strip():
        raise ValueError("backing media id must be a positive integer or text.")
    expected_suffix = f":media:{backing_id}"
    if canonical_id not in {
        f"local{expected_suffix}",
        f"server{expected_suffix}",
    }:
        raise ValueError("canonical media id must match its backing id.")


@dataclass(frozen=True)
class MediaReaderDetailRequest:
    """One generation-fenced detail request without its database row."""

    generation: int
    requested_id: str
    backing_id: BackingMediaId
    delay_seconds: float

    def __post_init__(self) -> None:
        if type(self.generation) is not int or self.generation < 1:
            raise ValueError("request generation must be a positive integer.")
        _validate_media_identity(self.requested_id, self.backing_id)
        if self.delay_seconds not in {0, SELECTION_SETTLE_SECONDS}:
            raise ValueError("request delay must be immediate or the settle delay.")


@dataclass(frozen=True)
class LibraryMediaReaderSessionState:
    """Transient Reader identities, request fence, mode, and error state."""

    selected_id: str | None = None
    selected_backing_id: BackingMediaId | None = None
    selected_title: str = ""
    loaded_id: str | None = None
    loaded_backing_id: BackingMediaId | None = None
    loaded_title: str = ""
    pending_request: MediaReaderDetailRequest | None = None
    request_generation: int = 0
    mode: ReaderMode = "read"
    more_open: bool = False
    error: str | None = None
    external_detail: bool = False

    def __post_init__(self) -> None:
        if type(self.request_generation) is not int or self.request_generation < 0:
            raise ValueError("request_generation must be a non-negative integer.")
        if self.mode not in {"read", "analysis", "highlights", "info"}:
            raise ValueError("mode is not a supported Reader mode.")
        self._validate_slot(
            "selected", self.selected_id, self.selected_backing_id, self.selected_title
        )
        self._validate_slot(
            "loaded", self.loaded_id, self.loaded_backing_id, self.loaded_title
        )
        if self.pending_request is not None:
            if self.pending_request.generation != self.request_generation:
                raise ValueError("pending request generation must be current.")
            if (
                self.pending_request.requested_id != self.selected_id
                or self.pending_request.backing_id != self.selected_backing_id
            ):
                raise ValueError("pending request must match the selected item.")
        elif (
            self.selected_id,
            self.selected_backing_id,
            self.selected_title,
        ) != (self.loaded_id, self.loaded_backing_id, self.loaded_title):
            raise ValueError(
                "selected and loaded identity may differ only while pending."
            )
        if self.error is not None:
            if not isinstance(self.error, str) or not self.error.strip():
                raise ValueError("error must be non-blank text.")
            if self.pending_request is None:
                raise ValueError("an error requires its current pending request.")
        if self.external_detail:
            if (
                self.selected_id is None
                or not self.selected_id.startswith("server:media:")
                or self.selected_id != self.loaded_id
                or self.pending_request is not None
            ):
                raise ValueError("external detail must be one settled server item.")
        elif any(
            identity is not None and identity.startswith("server:media:")
            for identity in (self.selected_id, self.loaded_id)
        ):
            raise ValueError("server identities require an external detail session.")

    @staticmethod
    def _validate_slot(
        name: str,
        canonical_id: str | None,
        backing_id: BackingMediaId | None,
        title: str,
    ) -> None:
        if not isinstance(title, str):
            raise TypeError(f"{name} title must be a string.")
        if canonical_id is None:
            if backing_id is not None or title:
                raise ValueError(f"empty {name} identity cannot carry metadata.")
            return
        if backing_id is None:
            raise ValueError(f"{name} identity requires a backing id.")
        _validate_media_identity(canonical_id, backing_id)

    @property
    def pending_banner(self) -> str | None:
        """Return truthful selected-versus-loaded loading copy.

        Returns:
            Loading copy for the pending request, or None when no request is
            pending or the current request is in an error state.
        """
        if self.pending_request is None or self.error is not None:
            return None
        selected = self.selected_title or self.selected_id or "selected item"
        if self.loaded_id is None or self.loaded_id == self.selected_id:
            return f"Loading preview for “{selected}”…"
        loaded = self.loaded_title or self.loaded_id
        return f"Loading preview for “{selected}”… showing “{loaded}” until ready."


def resolve_media_reader_layout(
    width: int,
    preferences: MediaReaderLayoutPreferences,
    *,
    previous: MediaReaderEffectiveLayout | None = None,
    priority: PaneName | None = None,
) -> MediaReaderEffectiveLayout:
    """Resolve Media preferences through the shared adaptive layout policy.

    Args:
        width: Available shell width in terminal cells.
        preferences: Persisted manual pane preferences.
        previous: Previously resolved layout used for hysteresis.
        priority: Pane explicitly requested by the user, if any.

    Returns:
        Media-compatible effective pane geometry.
    """
    return resolve_adaptive_reader_layout(
        width,
        preferences,
        MEDIA_READER_LAYOUT_PROFILE,
        previous=previous,
        priority=priority,
    )


def begin_selection(
    state: LibraryMediaReaderSessionState,
    canonical_id: str,
    backing_id: BackingMediaId,
    title: str,
    *,
    immediate: bool = False,
) -> LibraryMediaReaderSessionState:
    """Select a local item and create its next fenced request.

    Args:
        state: Current Reader session.
        canonical_id: Local backend-qualified media id.
        backing_id: Service-facing id matching ``canonical_id``.
        title: Display title for loading and recovery copy.
        immediate: Whether to bypass the selection-settle delay.

    Returns:
        Session with the new selection and pending request.

    Raises:
        TypeError: If an identity or title has the wrong type.
        ValueError: If the identity is invalid or is not local.
    """
    _validate_media_identity(canonical_id, backing_id)
    if not canonical_id.startswith("local:media:"):
        raise ValueError("Items selection must use a local canonical id.")
    if not isinstance(title, str):
        raise TypeError("title must be a string.")
    generation = state.request_generation + 1
    request = MediaReaderDetailRequest(
        generation=generation,
        requested_id=canonical_id,
        backing_id=backing_id,
        delay_seconds=0 if immediate else SELECTION_SETTLE_SECONDS,
    )
    clear_external = state.external_detail
    return replace(
        state,
        selected_id=canonical_id,
        selected_backing_id=backing_id,
        selected_title=title,
        loaded_id=None if clear_external else state.loaded_id,
        loaded_backing_id=None if clear_external else state.loaded_backing_id,
        loaded_title="" if clear_external else state.loaded_title,
        pending_request=request,
        request_generation=generation,
        error=None,
        external_detail=False,
    )


def _matches_pending(
    state: LibraryMediaReaderSessionState,
    generation: int,
    requested_id: str,
) -> bool:
    request = state.pending_request
    return (
        request is not None
        and request.generation == generation
        and request.requested_id == requested_id
    )


def settle_success(
    state: LibraryMediaReaderSessionState,
    generation: int,
    requested_id: str,
) -> LibraryMediaReaderSessionState:
    """Apply a detail success only when both request fence fields match.

    Args:
        state: Current Reader session.
        generation: Completing request generation.
        requested_id: Completing request's canonical id.

    Returns:
        Settled session, or ``state`` unchanged for a stale completion.
    """
    if not _matches_pending(state, generation, requested_id):
        return state
    return replace(
        state,
        loaded_id=state.selected_id,
        loaded_backing_id=state.selected_backing_id,
        loaded_title=state.selected_title,
        pending_request=None,
        error=None,
    )


def settle_failure(
    state: LibraryMediaReaderSessionState,
    generation: int,
    requested_id: str,
    error: str,
) -> LibraryMediaReaderSessionState:
    """Record a current detail failure while rejecting stale completions.

    Args:
        state: Current Reader session.
        generation: Completing request generation.
        requested_id: Completing request's canonical id.
        error: Non-blank user-facing failure copy.

    Returns:
        Failed pending session, or ``state`` unchanged for a stale completion.

    Raises:
        ValueError: If a current request receives blank error text.
    """
    if not _matches_pending(state, generation, requested_id):
        return state
    if not isinstance(error, str) or not error.strip():
        raise ValueError("error must be non-blank text.")
    return replace(state, error=error.strip())


def set_mode(
    state: LibraryMediaReaderSessionState, mode: ReaderMode
) -> LibraryMediaReaderSessionState:
    """Change the Reader mode without touching item identities.

    Args:
        state: Current Reader session.
        mode: Destination Reader mode.

    Returns:
        Session with ``mode`` selected.

    Raises:
        ValueError: If ``mode`` is unsupported.
    """
    if mode not in {"read", "analysis", "highlights", "info"}:
        raise ValueError("mode is not a supported Reader mode.")
    return replace(state, mode=mode)


def set_more_open(
    state: LibraryMediaReaderSessionState, more_open: bool
) -> LibraryMediaReaderSessionState:
    """Change the transient inline More region without touching identity.

    Args:
        state: Current Reader session.
        more_open: Whether the secondary-action region is open.

    Returns:
        Session with the requested More-region state.

    Raises:
        TypeError: If ``more_open`` is not a bool.
    """
    if type(more_open) is not bool:
        raise TypeError("more_open must be a bool.")
    return replace(state, more_open=more_open)


def enter_external_detail(
    state: LibraryMediaReaderSessionState,
    backing_id: BackingMediaId,
    title: str,
) -> LibraryMediaReaderSessionState:
    """Enter one settled server detail outside the local Items list.

    Args:
        state: Current Reader session.
        backing_id: Server-facing media id.
        title: Server item title.

    Returns:
        Settled read-only external-detail session.

    Raises:
        TypeError: If the title or identity has the wrong type.
        ValueError: If the backing identity is invalid.
    """
    canonical_id = f"server:media:{backing_id}"
    _validate_media_identity(canonical_id, backing_id)
    if not isinstance(title, str):
        raise TypeError("title must be a string.")
    return replace(
        state,
        selected_id=canonical_id,
        selected_backing_id=backing_id,
        selected_title=title,
        loaded_id=canonical_id,
        loaded_backing_id=backing_id,
        loaded_title=title,
        pending_request=None,
        request_generation=state.request_generation + 1,
        error=None,
        external_detail=True,
    )


def leave_external_detail(
    state: LibraryMediaReaderSessionState,
) -> LibraryMediaReaderSessionState:
    """Leave an external detail and invalidate server selection anchors.

    Args:
        state: Current Reader session.

    Returns:
        Local empty session, or ``state`` unchanged when already local.
    """
    if not state.external_detail:
        return state
    return replace(
        state,
        selected_id=None,
        selected_backing_id=None,
        selected_title="",
        loaded_id=None,
        loaded_backing_id=None,
        loaded_title="",
        pending_request=None,
        request_generation=state.request_generation + 1,
        error=None,
        external_detail=False,
    )
