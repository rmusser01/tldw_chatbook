"""Strict operational values never coerce choices or disclose content."""

import json
from dataclasses import FrozenInstanceError, replace
from uuid import uuid4

import pytest

from tldw_chatbook.Chat.console_prompt_mixins import (
    MixinAttemptSnapshot,
    MixinAttemptWrite,
    MixinPreferences,
    MixinSelection,
    MixinSnapshot,
    MixinSourceRef,
    decode_mixin_preferences,
    encode_mixin_preferences,
)
from tldw_chatbook.Prompt_Management.prompt_artifact_models import (
    LiteralMixinDefinition,
)
from tldw_chatbook.Prompt_Management.prompt_source_capabilities import (
    LOCAL_COMPILED_LANE_LIMIT,
)


def selection():
    return MixinSelection(
        MixinSourceRef("local:profile", "local", str(uuid4())),
        True,
        "PRIVATE NAME",
        "before",
        1,
    )


def test_preferences_roundtrip_and_immutable_values_hide_content():
    item = selection()
    value = MixinPreferences(entries=(item,))
    raw = encode_mixin_preferences(value)
    assert decode_mixin_preferences(raw).preferences == value
    assert decode_mixin_preferences(None).preferences == MixinPreferences()
    snapshot = MixinSnapshot(
        item.source, 1, item.name, LiteralMixinDefinition("PRIVATE BODY\n", "before")
    )
    batch = MixinAttemptWrite(
        "attempt", "user-native", "assistant-native", MixinAttemptSnapshot((snapshot,))
    )
    for obj in (value, item, snapshot, batch, decode_mixin_preferences(raw)):
        assert "PRIVATE" not in repr(obj)
    with pytest.raises(FrozenInstanceError):
        value.revision = 3


@pytest.mark.parametrize(
    "patch",
    [
        {"enabled": 1},
        {"enabled": "true"},
        {"last_used_version": True},
        {"placement": "middle"},
        {"name": None},
    ],
)
def test_selection_rejects_coercion(patch):
    with pytest.raises(ValueError):
        replace(selection(), **patch)


@pytest.mark.parametrize(
    "patch",
    [
        {"record_uuid": "not-uuid"},
        {"backend": "remote"},
        {"authority_key": "https://user:secret@example.test"},
    ],
)
def test_source_rejects_invalid_identity(patch):
    with pytest.raises(ValueError):
        replace(selection().source, **patch)


def test_duplicate_references_and_boolean_versions_are_refused():
    entry = selection()
    with pytest.raises(ValueError):
        MixinPreferences(entries=(entry, entry))
    for patch in ({"revision": True}, {"schema_version": True}, {"entries": [entry]}):
        with pytest.raises(ValueError):
            MixinPreferences(**patch)
    with pytest.raises(ValueError, match="mixin_"):
        MixinSnapshot(
            entry.source,
            1,
            entry.name,
            LiteralMixinDefinition("x" * (LOCAL_COMPILED_LANE_LIMIT + 1), "before"),
        )


def test_future_and_corrupt_preferences_keep_raw_without_repr_leaks():
    for raw, status in [
        ('{"schema_version":2,"body":"PRIVATE BODY"}', "unsupported"),
        ('{"schema_version":true}', "corrupt"),
        ("PRIVATE BAD JSON", "corrupt"),
    ]:
        result = decode_mixin_preferences(raw)
        assert result.status == status and result.preferences is None
        assert result.raw == raw and "PRIVATE" not in repr(result)
    raw = json.loads(encode_mixin_preferences(MixinPreferences()))
    raw["extra"] = "PRIVATE BODY"
    assert decode_mixin_preferences(raw).status == "corrupt"
    assert (
        decode_mixin_preferences('{"schema_version":1,"schema_version":2}').status
        == "corrupt"
    )


def test_public_mixin_values_are_the_shared_models_without_service_imports():
    import subprocess
    import sys

    script = """
import sys
from tldw_chatbook.Chat import console_prompt_mixin_models as models
from tldw_chatbook.Chat import console_prompt_mixins as public
for name in (
    'MixinSourceRef', 'MixinSelection', 'MixinPreferences', 'MixinSnapshot',
    'MixinAttemptSnapshot', 'MixinAttemptWrite', 'MixinPreferenceRead',
    'MixinReplayRead', 'MixinForkAttempt', 'MixinForkState',
    'MixinPreferenceConflictError', 'MixinSnapshotUnavailableError',
    'MixinSourceChangedError', 'decode_mixin_preferences', 'encode_mixin_preferences',
):
    assert getattr(public, name) is getattr(models, name), name
for name in (
    'prompt_source_capabilities', 'prompt_artifact_codec', 'prompt_block_compiler',
):
    assert 'tldw_chatbook.Prompt_Management.' + name not in sys.modules, name
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
