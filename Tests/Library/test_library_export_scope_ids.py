import pytest

from tldw_chatbook.Chatbooks.chatbook_models import ContentType
from tldw_chatbook.Library.library_export_scope import (
    ExportScope,
    resolve_export_selections,
    count_export_scope,
    export_scope_label,
)


class _RecordingMediaDB:
    def __init__(self):
        self.calls = 0

    def get_all_active_media_ids(self, media_type=None):
        self.calls += 1
        return [1, 2, 3]


class _RecordingChaChaDB:
    def __init__(self):
        self.calls = 0

    def get_all_conversation_ids(self):
        self.calls += 1
        return ["c1", "c2"]

    def get_all_note_ids(self):
        self.calls += 1
        return ["n1"]


class _RecordingPromptsDB:
    def __init__(self):
        self.calls = 0

    def get_all_active_prompt_ids(self):
        self.calls += 1
        return [11, 12]


def test_ids_only_valid_for_single_source():
    ExportScope(kind="media", ids=("1", "2"))  # ok
    ExportScope(kind="prompts", ids=("1", "2"))  # ok
    with pytest.raises(ValueError):
        ExportScope(kind="everything", ids=("1",))


def test_scope_with_ids_is_hashable_and_equal():
    a = ExportScope(kind="media", ids=("1", "2"))
    b = ExportScope(kind="media", ids=("1", "2"))
    assert a == b and hash(a) == hash(b)
    assert a in {b}


def test_resolve_with_ids_returns_them_without_querying():
    media, chacha, prompts = (
        _RecordingMediaDB(),
        _RecordingChaChaDB(),
        _RecordingPromptsDB(),
    )
    sel = resolve_export_selections(
        ExportScope(kind="media", ids=("7", "8")), media, chacha, prompts
    )
    assert sel == {ContentType.MEDIA: ["7", "8"]}
    assert media.calls == chacha.calls == prompts.calls == 0


def test_resolve_prompt_ids_maps_directly_without_querying():
    media, chacha, prompts = (
        _RecordingMediaDB(),
        _RecordingChaChaDB(),
        _RecordingPromptsDB(),
    )
    sel = resolve_export_selections(
        ExportScope(kind="prompts", ids=("7", "8")), media, chacha, prompts
    )
    assert sel == {ContentType.PROMPT: ["7", "8"]}
    assert media.calls == chacha.calls == prompts.calls == 0


def test_resolve_without_ids_unchanged():
    media, chacha, prompts = (
        _RecordingMediaDB(),
        _RecordingChaChaDB(),
        _RecordingPromptsDB(),
    )
    sel = resolve_export_selections(ExportScope(kind="media"), media, chacha, prompts)
    assert sel == {ContentType.MEDIA: ["1", "2", "3"]}
    assert media.calls == 1
    assert prompts.calls == 0


def test_count_with_ids():
    media, chacha, prompts = (
        _RecordingMediaDB(),
        _RecordingChaChaDB(),
        _RecordingPromptsDB(),
    )
    counts = count_export_scope(
        ExportScope(kind="notes", ids=("a", "b", "c")), media, chacha, prompts
    )
    assert counts == {"media": 0, "conversations": 0, "notes": 3, "prompts": 0}
    assert chacha.calls == 0


def test_label_with_ids():
    counts = count_export_scope(
        ExportScope(kind="conversations", ids=("x", "y")),
        _RecordingMediaDB(),
        _RecordingChaChaDB(),
        _RecordingPromptsDB(),
    )
    assert (
        export_scope_label(ExportScope(kind="conversations", ids=("x", "y")), counts)
        == "Selected conversations · 2 items"
    )
