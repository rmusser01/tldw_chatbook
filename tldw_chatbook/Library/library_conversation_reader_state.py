"""Immutable state transitions for the Library Conversations reader."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal, Mapping
from unicodedata import normalize


LIBRARY_CONVERSATION_PAGE_SIZE = 20
ConversationReaderMode = Literal["read", "info"]


@dataclass(frozen=True)
class ConversationReaderRequest:
    """One Conversations detail request fenced against stale settlement."""

    destination: Literal["conversations"]
    conversation_id: str
    version: int
    generation: int
    message_offset: int = 0
    message_limit: int = LIBRARY_CONVERSATION_PAGE_SIZE


@dataclass(frozen=True)
class ConversationMessageView:
    """One progressively assembled service message body."""

    message_id: str
    sender: str
    timestamp: str
    revision: str
    total_chars: int
    text: str

    @property
    def complete(self) -> bool:
        """Whether this view contains the message's entire saved body."""
        return len(self.text) == self.total_chars


@dataclass(frozen=True)
class ConversationFindMatch:
    """One stable match location in the normalized transcript."""

    message_id: str
    message_index: int
    message_offset: int
    transcript_offset: int
    match_length: int


@dataclass(frozen=True)
class ConversationReaderState:
    """One read-only Conversations reader projection."""

    selected_id: str | None = None
    selected_version: int | None = None
    loaded_id: str | None = None
    loaded_version: int | None = None
    loaded_generation: int | None = None
    generation: int = 0
    mode: ConversationReaderMode = "read"
    messages: tuple[ConversationMessageView, ...] = ()
    message_total: int = 0
    message_epoch: str | None = None
    complete: bool = False
    find_query: str = ""
    find_matches: tuple[ConversationFindMatch, ...] = ()
    find_complete: bool = False
    error: str | None = None
    loading: bool = False
    unavailable: bool = False
    bulk_active: bool = False
    bulk_selected_count: int = 0
    bulk_loaded_preview_selected: bool | None = None

    @property
    def loaded_actions_eligible(self) -> bool:
        """Whether actions may target the currently selected loaded item."""
        return (
            not self.bulk_active
            and self.bulk_selected_count == 0
            and not self.unavailable
            and self.complete
            and not self.loading
            and self.error is None
            and self.selected_id is not None
            and self.selected_id == self.loaded_id
            and self.selected_version == self.loaded_version
            and self.loaded_generation == self.generation
        )


def select_conversation(
    state: ConversationReaderState,
    conversation_id: str,
    *,
    version: int,
) -> tuple[ConversationReaderState, ConversationReaderRequest]:
    """Select an item and create its next destination-fenced request."""
    generation = state.generation + 1
    request = ConversationReaderRequest(
        destination="conversations",
        conversation_id=conversation_id,
        version=version,
        generation=generation,
    )
    return (
        replace(
            state,
            selected_id=conversation_id,
            selected_version=version,
            generation=generation,
            mode=state.mode if conversation_id == state.loaded_id else "read",
            error=None,
            loading=True,
            unavailable=False,
            bulk_active=False,
            bulk_selected_count=0,
            bulk_loaded_preview_selected=None,
        ),
        request,
    )


def set_conversation_reader_mode(
    state: ConversationReaderState,
    mode: ConversationReaderMode,
) -> ConversationReaderState:
    """Set the concrete Conversations Read/Info mode."""
    if mode not in {"read", "info"}:
        raise ValueError("mode must be 'read' or 'info'.")
    return replace(state, mode=mode)


def _matches_request(
    state: ConversationReaderState,
    request: ConversationReaderRequest,
    response: Mapping[str, Any] | None = None,
) -> bool:
    matches = (
        request.destination == "conversations"
        and request.conversation_id == state.selected_id
        and request.version == state.selected_version
        and request.generation == state.generation
    )
    if not matches or response is None:
        return matches
    return (
        response.get("id") == request.conversation_id
        and response.get("version") == request.version
    )


def _message_view(raw: Mapping[str, Any]) -> ConversationMessageView:
    if not isinstance(raw, Mapping):
        raise ValueError("message mapping is required.")
    text = raw.get("text")
    total_chars = raw.get("total_chars")
    if (
        not isinstance(text, str)
        or type(total_chars) is not int
        or total_chars < len(text)
        or raw.get("char_start") != 0
        or raw.get("returned_chars") != len(text)
    ):
        raise ValueError("message page contains an invalid bounded body window.")
    values = {
        name: raw.get(source)
        for name, source in (
            ("message_id", "id"),
            ("sender", "sender"),
            ("timestamp", "timestamp"),
            ("revision", "revision"),
        )
    }
    if any(not isinstance(value, str) or not value for value in values.values()):
        raise ValueError("message page contains invalid stable message metadata.")
    return ConversationMessageView(
        **values,  # type: ignore[arg-type]
        total_chars=total_chars,
        text=text,
    )


def _transcript_complete(
    messages: tuple[ConversationMessageView, ...], message_total: int
) -> bool:
    return len(messages) == message_total and all(
        message.complete for message in messages
    )


def _find_matches(
    messages: tuple[ConversationMessageView, ...], query: str
) -> tuple[ConversationFindMatch, ...]:
    normalized_query = normalize("NFKC", query).strip().casefold()
    if not normalized_query:
        return ()
    matches: list[ConversationFindMatch] = []
    transcript_start = 0
    for message_index, message in enumerate(messages):
        text = normalize("NFKC", message.text).casefold()
        start = 0
        while (offset := text.find(normalized_query, start)) >= 0:
            matches.append(
                ConversationFindMatch(
                    message.message_id,
                    message_index,
                    offset,
                    transcript_start + offset,
                    len(normalized_query),
                )
            )
            start = offset + 1
        transcript_start += len(text) + 1
    return tuple(matches)


def _refresh_find(state: ConversationReaderState) -> ConversationReaderState:
    return replace(
        state,
        find_matches=(
            _find_matches(state.messages, state.find_query) if state.complete else ()
        ),
        find_complete=state.complete,
    )


def set_conversation_find_query(
    state: ConversationReaderState, query: str
) -> ConversationReaderState:
    """Find normalized case-insensitive text once every body is hydrated."""
    if not isinstance(query, str):
        raise TypeError("find query must be text.")
    return _refresh_find(replace(state, find_query=query.strip()))


def settle_conversation_page(
    state: ConversationReaderState,
    request: ConversationReaderRequest,
    response: Mapping[str, Any],
) -> ConversationReaderState:
    """Apply one initial or subsequent bounded service message page."""
    if not _matches_request(state, request, response):
        return state
    raw_messages = response.get("messages")
    offset = response.get("message_offset")
    message_total = response.get("message_total")
    returned_count = response.get("returned_message_count")
    has_more = response.get("has_more")
    next_message_offset = response.get("next_message_offset")
    message_epoch = response.get("message_epoch")
    if (
        not isinstance(raw_messages, list)
        or type(offset) is not int
        or offset < 0
        or type(message_total) is not int
        or message_total < 0
        or type(returned_count) is not int
        or returned_count != len(raw_messages)
        or type(has_more) is not bool
        or type(request.message_offset) is not int
        or request.message_offset < 0
        or type(request.message_limit) is not int
        or request.message_limit <= 0
        or not isinstance(message_epoch, str)
        or not message_epoch.strip()
    ):
        raise ValueError("conversation page has invalid coordinates or epoch.")
    expected_next = offset + len(raw_messages)
    expected_has_more = expected_next < message_total
    if (
        offset != request.message_offset
        or offset > message_total
        or expected_next > message_total
        or len(raw_messages) != min(request.message_limit, message_total - offset)
        or has_more is not expected_has_more
        or next_message_offset != (expected_next if expected_has_more else None)
    ):
        raise ValueError("conversation page has invalid coordinates.")
    page = tuple(_message_view(raw) for raw in raw_messages)

    current_request_loaded = (
        state.loaded_id == request.conversation_id
        and state.loaded_version == request.version
        and state.loaded_generation == request.generation
    )
    if offset == 0 and not current_request_loaded:
        messages = page
    else:
        if (
            not current_request_loaded
            or state.message_epoch != message_epoch
            or offset > len(state.messages)
        ):
            return state
        overlap = min(len(page), len(state.messages) - offset)
        if state.messages[offset : offset + overlap] != page[:overlap]:
            return state
        messages = state.messages + page[overlap:]

    if len({message.message_id for message in messages}) != len(messages):
        return state
    if current_request_loaded and message_total != state.message_total:
        return state
    complete = _transcript_complete(messages, message_total)
    updated = _refresh_find(
        replace(
            state,
            loaded_id=request.conversation_id,
            loaded_version=request.version,
            loaded_generation=request.generation,
            messages=messages,
            message_total=message_total,
            message_epoch=message_epoch,
            complete=complete,
            error=None,
            loading=not complete,
            unavailable=False,
        )
    )
    return state if updated == state else updated


def settle_conversation_continuation(
    state: ConversationReaderState,
    request: ConversationReaderRequest,
    response: Mapping[str, Any],
) -> ConversationReaderState:
    """Extend one matching message with a bounded contiguous body window."""
    if not _matches_request(state, request, response):
        return state
    if (
        state.loaded_id != request.conversation_id
        or state.loaded_version != request.version
        or state.loaded_generation != request.generation
        or response.get("message_total") != state.message_total
        or not isinstance(response.get("message_epoch"), str)
        or response.get("message_epoch") != state.message_epoch
    ):
        return state
    raw_messages = response.get("messages")
    if not isinstance(raw_messages, list) or len(raw_messages) != 1:
        return state
    raw = raw_messages[0]
    if not isinstance(raw, Mapping):
        raise ValueError("conversation continuation must contain one message mapping.")
    message_id = raw.get("id")
    index = next(
        (
            position
            for position, message in enumerate(state.messages)
            if message.message_id == message_id
        ),
        None,
    )
    if index is None:
        return state
    current = state.messages[index]
    text = raw.get("text")
    char_start = raw.get("char_start")
    if (
        raw.get("revision") != current.revision
        or raw.get("total_chars") != current.total_chars
        or not isinstance(text, str)
        or type(char_start) is not int
        or char_start < 0
        or raw.get("returned_chars") != len(text)
        or char_start > len(current.text)
        or char_start + len(text) > current.total_chars
    ):
        return state
    overlap = len(current.text) - char_start
    if current.text[char_start:] != text[:overlap]:
        return state
    suffix = text[overlap:]
    if not suffix:
        return state
    messages = list(state.messages)
    messages[index] = replace(current, text=current.text + suffix)
    assembled = tuple(messages)
    complete = _transcript_complete(assembled, state.message_total)
    return _refresh_find(
        replace(
            state,
            messages=assembled,
            complete=complete,
            error=None,
            loading=not complete,
            unavailable=False,
        )
    )


def settle_conversation_error(
    state: ConversationReaderState,
    request: ConversationReaderRequest,
    error: str,
) -> ConversationReaderState:
    """Record a matching initial or refresh failure without relabelling content."""
    if not _matches_request(state, request):
        return state
    if not isinstance(error, str) or not error.strip():
        raise ValueError("error must be non-blank text.")
    return replace(state, error=error.strip(), loading=False)


def retry_conversation(
    state: ConversationReaderState,
) -> tuple[ConversationReaderState, ConversationReaderRequest]:
    """Retry the selected conversation with a new request generation."""
    if state.selected_id is None or state.selected_version is None:
        raise ValueError("retry requires a selected conversation.")
    return select_conversation(
        state,
        state.selected_id,
        version=state.selected_version,
    )


def settle_conversation_unavailable(
    state: ConversationReaderState,
    request: ConversationReaderRequest,
) -> ConversationReaderState:
    """Settle a matching None/deleted result without mislabelling retained content."""
    if not _matches_request(state, request):
        return state
    clear_loaded = state.loaded_id == request.conversation_id
    unavailable = replace(
        state,
        loaded_id=None if clear_loaded else state.loaded_id,
        loaded_version=None if clear_loaded else state.loaded_version,
        loaded_generation=None if clear_loaded else state.loaded_generation,
        messages=() if clear_loaded else state.messages,
        message_total=0 if clear_loaded else state.message_total,
        message_epoch=None if clear_loaded else state.message_epoch,
        complete=False if clear_loaded else state.complete,
        error="Conversation unavailable.",
        loading=False,
        unavailable=True,
    )
    return _refresh_find(unavailable)


def mark_conversation_deleted(
    state: ConversationReaderState,
    request: ConversationReaderRequest,
) -> ConversationReaderState:
    """Apply a matching deletion settlement and clear its loaded transcript."""
    if not _matches_request(state, request):
        return state
    return confirm_conversation_deleted(
        state,
        request.conversation_id,
        version=request.version,
        generation=request.generation,
    )


def confirm_conversation_deleted(
    state: ConversationReaderState,
    conversation_id: str,
    *,
    version: int | None,
    generation: int,
) -> ConversationReaderState:
    """Apply an exact-ID deletion fence, including a version bootstrap epoch."""
    if (
        state.selected_id != conversation_id
        or state.selected_version != version
        or state.generation != generation
    ):
        return state
    selected = state.selected_id == conversation_id
    loaded = state.loaded_id == conversation_id
    deleted = replace(
        state,
        loaded_id=None if loaded else state.loaded_id,
        loaded_version=None if loaded else state.loaded_version,
        loaded_generation=None if loaded else state.loaded_generation,
        generation=state.generation + 1 if selected else state.generation,
        messages=() if loaded else state.messages,
        message_total=0 if loaded else state.message_total,
        message_epoch=None if loaded else state.message_epoch,
        complete=False if loaded else state.complete,
        error="Conversation deleted." if selected else state.error,
        loading=False if selected else state.loading,
        unavailable=selected,
    )
    return _refresh_find(deleted)


def project_conversation_multiselect(
    state: ConversationReaderState,
    *,
    active: bool,
    selected_count: int,
    loaded_preview_selected: bool | None,
) -> ConversationReaderState:
    """Project read-only bulk selection without replacing the single preview."""
    if type(active) is not bool:
        raise TypeError("active must be a boolean.")
    if type(selected_count) is not int or selected_count < 0:
        raise ValueError("selected_count must be a non-negative integer.")
    if not active:
        selected_count = 0
        loaded_preview_selected = None
    elif state.loaded_id is None:
        if loaded_preview_selected is not None:
            raise ValueError("a missing loaded preview cannot be selected.")
    elif type(loaded_preview_selected) is not bool:
        raise TypeError("loaded_preview_selected must describe the loaded preview.")
    entering = active and not state.bulk_active
    if not entering:
        return replace(
            state,
            bulk_active=active,
            bulk_selected_count=selected_count,
            bulk_loaded_preview_selected=loaded_preview_selected,
        )
    if not state.loading:
        return replace(
            state,
            bulk_active=True,
            bulk_selected_count=selected_count,
            bulk_loaded_preview_selected=loaded_preview_selected,
        )
    generation = state.generation + 1
    return replace(
        state,
        generation=generation,
        loading=False,
        bulk_active=True,
        bulk_selected_count=selected_count,
        bulk_loaded_preview_selected=loaded_preview_selected,
    )
