"""Media destination message contracts.

The mounted ``MediaWindow`` is the sole handler for these messages. This
module intentionally contains no app-root dispatch, mutation, search, or
presentation implementation.
"""

from __future__ import annotations

from typing import Any, Dict

from textual.message import Message


class MediaMetadataUpdateEvent(Message):
    """Event for updating media metadata."""

    def __init__(
        self,
        media_id: Any,
        title: str,
        media_type: str,
        author: str,
        url: str,
        keywords: list,
        type_slug: str,
        record_id: Any = None,
        backing_media_id: Any = None,
    ) -> None:
        super().__init__()
        self.media_id = media_id
        self.record_id = record_id if record_id is not None else media_id
        self.backing_media_id = backing_media_id
        self.title = title
        self.media_type = media_type
        self.author = author
        self.url = url
        self.keywords = keywords
        self.type_slug = type_slug


class MediaDeleteConfirmationEvent(Message):
    """Event requesting media deletion confirmation."""

    def __init__(
        self,
        media_id: Any,
        media_title: str,
        type_slug: str,
        record_id: Any = None,
        backing_media_id: Any = None,
    ) -> None:
        super().__init__()
        self.media_id = media_id
        self.record_id = record_id if record_id is not None else media_id
        self.backing_media_id = backing_media_id
        self.media_title = media_title
        self.type_slug = type_slug


class MediaUndeleteEvent(Message):
    """Event requesting media restoration."""

    def __init__(self, media_id: Any, type_slug: str, record_id: Any = None) -> None:
        super().__init__()
        self.media_id = media_id
        self.record_id = record_id if record_id is not None else media_id
        self.type_slug = type_slug


class MediaListCollapseEvent(Message):
    """Event fired when the media list should collapse or expand."""


class SidebarCollapseEvent(Message):
    """Event fired when the Media sidebar should collapse or expand."""


class MediaAnalysisRequestEvent(Message):
    """Event requesting LLM analysis of media content."""

    def __init__(
        self,
        media_id: Any,
        provider: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        type_slug: str,
        temperature: float = 0.7,
        top_p: float = 0.95,
        min_p: float = 0.05,
        max_tokens: int = 4096,
        record_id: Any = None,
        backing_media_id: Any = None,
    ) -> None:
        super().__init__()
        self.media_id = media_id
        self.record_id = record_id if record_id is not None else media_id
        self.backing_media_id = backing_media_id
        self.provider = provider
        self.model = model
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.type_slug = type_slug
        self.temperature = temperature
        self.top_p = top_p
        self.min_p = min_p
        self.max_tokens = max_tokens


class MediaAnalysisSaveEvent(Message):
    """Event requesting a new saved analysis version."""

    def __init__(
        self,
        media_id: Any,
        analysis_content: str,
        type_slug: str,
        record_id: Any = None,
    ) -> None:
        super().__init__()
        self.media_id = media_id
        self.record_id = record_id if record_id is not None else media_id
        self.analysis_content = analysis_content
        self.type_slug = type_slug


class MediaAnalysisSaveAsNoteEvent(Message):
    """Event requesting an analysis copy in Notes."""

    def __init__(
        self,
        media_id: Any,
        media_title: str,
        analysis_content: str,
        record_id: Any = None,
    ) -> None:
        super().__init__()
        self.media_id = media_id
        self.record_id = record_id if record_id is not None else media_id
        self.media_title = media_title
        self.analysis_content = analysis_content


class MediaAnalysisOverwriteEvent(Message):
    """Event requesting overwrite of an analysis version."""

    def __init__(
        self,
        media_id: Any,
        analysis_content: str,
        type_slug: str,
        record_id: Any = None,
    ) -> None:
        super().__init__()
        self.media_id = media_id
        self.record_id = record_id if record_id is not None else media_id
        self.analysis_content = analysis_content
        self.type_slug = type_slug


class MediaAnalysisDeleteEvent(Message):
    """Event requesting deletion of an analysis version."""

    def __init__(
        self,
        media_id: Any,
        version_uuid: str,
        type_slug: str,
        record_id: Any = None,
        version_number: Any = None,
    ) -> None:
        super().__init__()
        self.media_id = media_id
        self.record_id = record_id if record_id is not None else media_id
        self.version_uuid = version_uuid
        self.version_number = version_number
        self.type_slug = type_slug


class MediaReadItLaterToggleEvent(Message):
    """Event requesting a read-it-later state change."""

    def __init__(
        self,
        media_id: Any,
        *,
        record_id: Any = None,
        save_for_later: bool = True,
    ) -> None:
        super().__init__()
        self.media_id = media_id
        self.record_id = record_id if record_id is not None else media_id
        self.save_for_later = save_for_later


class MediaReadingHighlightCreateEvent(Message):
    """Event requesting creation of a reading highlight."""

    def __init__(
        self,
        media_id: Any,
        *,
        record_id: Any = None,
        quote: str,
        start_offset: int | None = None,
        end_offset: int | None = None,
        color: str | None = None,
        note: str | None = None,
        anchor_strategy: str = "fuzzy_quote",
        media_data: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.media_id = media_id
        self.record_id = record_id if record_id is not None else media_id
        self.quote = quote
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.color = color
        self.note = note
        self.anchor_strategy = anchor_strategy
        self.media_data = media_data


class MediaReadingHighlightUpdateEvent(Message):
    """Event requesting update of a reading highlight."""

    def __init__(
        self,
        media_id: Any,
        *,
        highlight_id: Any,
        record_id: Any = None,
        quote: str | None = None,
        color: str | None = None,
        note: str | None = None,
        state: str | None = "active",
        media_data: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.media_id = media_id
        self.record_id = record_id if record_id is not None else media_id
        self.highlight_id = highlight_id
        self.quote = quote
        self.color = color
        self.note = note
        self.state = state
        self.media_data = media_data


class MediaReadingHighlightDeleteEvent(Message):
    """Event requesting deletion of a reading highlight."""

    def __init__(
        self,
        media_id: Any,
        *,
        highlight_id: Any,
        record_id: Any = None,
        media_data: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.media_id = media_id
        self.record_id = record_id if record_id is not None else media_id
        self.highlight_id = highlight_id
        self.media_data = media_data
