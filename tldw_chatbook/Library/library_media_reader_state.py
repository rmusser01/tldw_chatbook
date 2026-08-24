"""Pure session and responsive-layout state for the Library Media reader."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal, Mapping

LIBRARY_TARGET_WIDTH = 28
LIBRARY_MIN_WIDTH = 24
LIBRARY_MAX_WIDTH = 48
ITEMS_TARGET_WIDTH = 40
ITEMS_MIN_WIDTH = 32
ITEMS_MAX_WIDTH = 72
READER_COMFORT_WIDTH = 44
PANE_GRIP_WIDTH = 5
LAYOUT_HYSTERESIS_WIDTH = 4
SELECTION_SETTLE_SECONDS = 0.12

ReaderMode = Literal["read", "analysis", "highlights", "info"]
PaneName = Literal["library", "items"]
BackingMediaId = int | str


@dataclass(frozen=True)
class MediaReaderLayoutPreferences:
    """Persisted manual pane choices and normalized target widths."""

    library_open: bool = True
    items_open: bool = True
    custom_widths_enabled: bool = False
    library_width: int = LIBRARY_TARGET_WIDTH
    items_width: int = ITEMS_TARGET_WIDTH


@dataclass(frozen=True)
class MediaReaderEffectiveLayout:
    """One rendered layout derived from preferences and available width."""

    library_open: bool
    items_open: bool
    library_width: int
    items_width: int
    reader_width: int
    priority_pane: PaneName | None


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
        """Return truthful selected-versus-loaded loading copy when applicable."""
        if self.pending_request is None or self.error is not None:
            return None
        selected = self.selected_title or self.selected_id or "selected item"
        if self.loaded_id is None or self.loaded_id == self.selected_id:
            return f"Loading preview for “{selected}”…"
        loaded = self.loaded_title or self.loaded_id
        return f"Loading preview for “{selected}”… showing “{loaded}” until ready."


def _coerce_bool(value: Any, default: bool) -> bool:
    if type(value) is bool:
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return default


def _coerce_width(value: Any, default: int, minimum: int, maximum: int) -> int:
    if type(value) is int:
        width = value
    elif isinstance(value, str):
        try:
            width = int(value.strip())
        except ValueError:
            return default
    else:
        return default
    return min(max(width, minimum), maximum)


def normalize_media_reader_preferences(
    raw: Mapping[str, Any],
) -> MediaReaderLayoutPreferences:
    """Normalize persisted values without importing application configuration."""
    library_open = _coerce_bool(raw.get("library_open"), True)
    items_open = _coerce_bool(raw.get("items_open"), True)
    custom_widths_enabled = _coerce_bool(raw.get("custom_widths_enabled"), False)
    if not custom_widths_enabled:
        return MediaReaderLayoutPreferences(
            library_open=library_open,
            items_open=items_open,
        )
    return MediaReaderLayoutPreferences(
        library_open=library_open,
        items_open=items_open,
        custom_widths_enabled=True,
        library_width=_coerce_width(
            raw.get("library_width"),
            LIBRARY_TARGET_WIDTH,
            LIBRARY_MIN_WIDTH,
            LIBRARY_MAX_WIDTH,
        ),
        items_width=_coerce_width(
            raw.get("items_width"),
            ITEMS_TARGET_WIDTH,
            ITEMS_MIN_WIDTH,
            ITEMS_MAX_WIDTH,
        ),
    )


def resolve_media_reader_layout(
    width: int,
    preferences: MediaReaderLayoutPreferences,
    *,
    previous: MediaReaderEffectiveLayout | None = None,
    priority: PaneName | None = None,
) -> MediaReaderEffectiveLayout:
    """Resolve preferred panes into one overflow-free effective layout.

    The two fixed grips always consume ten columns. Normal responsive
    resolution keeps target widths and collapses Library before Items. An
    explicit open instead protects the requested pane, collapses its sibling,
    and may use the requested pane's declared minimum.
    """
    if type(width) is not int or width < 0:
        raise ValueError("width must be a non-negative integer.")
    if not isinstance(preferences, MediaReaderLayoutPreferences):
        raise TypeError("preferences must be MediaReaderLayoutPreferences.")
    if priority not in {None, "library", "items"}:
        raise ValueError("priority must be library, items, or None.")
    if priority is None and previous is not None:
        inherited = previous.priority_pane
        if (
            inherited == "library"
            and preferences.library_open
            or inherited == "items"
            and preferences.items_open
        ):
            priority = inherited

    grip_width = 2 * PANE_GRIP_WIDTH
    library_open = preferences.library_open
    items_open = preferences.items_open
    if priority is not None:
        if priority == "library":
            library_open = True
        else:
            items_open = True

        full_width = (
            grip_width
            + (preferences.library_width if library_open else 0)
            + (preferences.items_width if items_open else 0)
            + READER_COMFORT_WIDTH
        )
        if width < full_width:
            if priority == "library":
                items_open = False
                library_width = (
                    preferences.library_width
                    if width
                    >= grip_width + preferences.library_width + READER_COMFORT_WIDTH
                    else LIBRARY_MIN_WIDTH
                )
                items_width = 0
            else:
                library_open = False
                library_width = 0
                items_width = (
                    preferences.items_width
                    if width
                    >= grip_width + preferences.items_width + READER_COMFORT_WIDTH
                    else ITEMS_MIN_WIDTH
                )
            return MediaReaderEffectiveLayout(
                library_open=library_open,
                items_open=items_open,
                library_width=library_width,
                items_width=items_width,
                reader_width=max(width - grip_width - library_width - items_width, 0),
                priority_pane=priority,
            )
        priority = None

    def required_width(open_library: bool, open_items: bool) -> int:
        return (
            grip_width
            + (preferences.library_width if open_library else 0)
            + (preferences.items_width if open_items else 0)
            + READER_COMFORT_WIDTH
        )

    if width < required_width(library_open, items_open):
        library_open = False
    if width < required_width(library_open, items_open):
        items_open = False

    if previous is not None:
        nominal_width = required_width(library_open, items_open)
        if (
            library_open
            and not previous.library_open
            and width < nominal_width + LAYOUT_HYSTERESIS_WIDTH
        ):
            library_open = False
        if (
            items_open
            and not previous.items_open
            and width
            < required_width(library_open, items_open) + LAYOUT_HYSTERESIS_WIDTH
        ):
            items_open = False

    library_width = preferences.library_width if library_open else 0
    items_width = preferences.items_width if items_open else 0
    return MediaReaderEffectiveLayout(
        library_open=library_open,
        items_open=items_open,
        library_width=library_width,
        items_width=items_width,
        reader_width=max(width - grip_width - library_width - items_width, 0),
        priority_pane=priority,
    )


def begin_selection(
    state: LibraryMediaReaderSessionState,
    canonical_id: str,
    backing_id: BackingMediaId,
    title: str,
    *,
    immediate: bool = False,
) -> LibraryMediaReaderSessionState:
    """Select a local item immediately and create its next fenced request."""
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
    """Apply a detail success only when both request fence fields match."""
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
    """Record a current detail failure while rejecting stale completions."""
    if not _matches_pending(state, generation, requested_id):
        return state
    if not isinstance(error, str) or not error.strip():
        raise ValueError("error must be non-blank text.")
    return replace(state, error=error.strip())


def set_mode(
    state: LibraryMediaReaderSessionState, mode: ReaderMode
) -> LibraryMediaReaderSessionState:
    """Change the session Reader mode without touching item identities."""
    if mode not in {"read", "analysis", "highlights", "info"}:
        raise ValueError("mode is not a supported Reader mode.")
    return replace(state, mode=mode)


def enter_external_detail(
    state: LibraryMediaReaderSessionState,
    backing_id: BackingMediaId,
    title: str,
) -> LibraryMediaReaderSessionState:
    """Enter one settled, read-only server detail outside the local Items list."""
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
    """Leave an external detail and invalidate all server selection anchors."""
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
