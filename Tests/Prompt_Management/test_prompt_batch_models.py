"""Strict contracts for atomic local Prompt batch mutations."""

from dataclasses import FrozenInstanceError

import pytest

from tldw_chatbook.Prompt_Management.prompt_batch_models import (
    PromptBatchDeleteResult,
    PromptBatchRestoreResult,
    PromptBatchTarget,
    PromptDeleteReceiptEntry,
    PromptRestoreResultEntry,
)


class _IntSubclass(int):
    """An integer lookalike that strict public boundaries must reject."""


class _StrSubclass(str):
    """A text lookalike that strict public boundaries must reject."""


class _TupleSubclass(tuple):
    """A tuple lookalike that strict public boundaries must reject."""


@pytest.mark.parametrize(
    ("model", "kwargs", "field"),
    [
        (PromptBatchTarget, {"local_id": 0, "expected_version": 1}, "local_id"),
        (PromptBatchTarget, {"local_id": True, "expected_version": 1}, "local_id"),
        (
            PromptBatchTarget,
            {"local_id": _IntSubclass(1), "expected_version": 1},
            "local_id",
        ),
        (
            PromptBatchTarget,
            {"local_id": 2**63, "expected_version": 1},
            "local_id",
        ),
        (
            PromptBatchTarget,
            {"local_id": 1, "expected_version": False},
            "expected_version",
        ),
        (
            PromptBatchTarget,
            {"local_id": 1, "expected_version": _IntSubclass(1)},
            "expected_version",
        ),
        (
            PromptBatchTarget,
            {"local_id": 1, "expected_version": 2**63},
            "expected_version",
        ),
        (
            PromptDeleteReceiptEntry,
            {
                "local_id": -1,
                "title": "Title",
                "artifact_type": "prompt",
                "tombstone_version": 2,
            },
            "local_id",
        ),
        (
            PromptDeleteReceiptEntry,
            {
                "local_id": 1,
                "title": "Title",
                "artifact_type": "prompt",
                "tombstone_version": 0,
            },
            "tombstone_version",
        ),
        (
            PromptRestoreResultEntry,
            {"local_id": 1.0, "restored_version": 2},
            "local_id",
        ),
        (
            PromptRestoreResultEntry,
            {"local_id": 1, "restored_version": "2"},
            "restored_version",
        ),
    ],
)
def test_batch_integer_contracts_require_exact_positive_sqlite_range(
    model, kwargs, field
):
    with pytest.raises((TypeError, ValueError), match=field):
        model(**kwargs)


@pytest.mark.parametrize("title", [None, "", " \n\t", _StrSubclass("Title")])
def test_batch_delete_receipt_requires_exact_nonempty_title(title):
    with pytest.raises((TypeError, ValueError), match="title"):
        PromptDeleteReceiptEntry(1, title, "prompt", 2)


@pytest.mark.parametrize(
    "artifact_type", ["Prompt", "block_recipe", "", None, _StrSubclass("prompt")]
)
def test_batch_delete_receipt_accepts_only_exact_supported_artifact_types(
    artifact_type,
):
    with pytest.raises((TypeError, ValueError), match="artifact_type"):
        PromptDeleteReceiptEntry(1, "Title", artifact_type, 2)


@pytest.mark.parametrize(
    ("result_type", "entry"),
    [
        (
            PromptBatchDeleteResult,
            PromptDeleteReceiptEntry(1, "Title", "prompt", 2),
        ),
        (PromptBatchRestoreResult, PromptRestoreResultEntry(1, 3)),
    ],
)
@pytest.mark.parametrize("entries", [[], _TupleSubclass(())])
def test_batch_results_require_exact_tuple(result_type, entry, entries):
    payload = entries if entries else type(entries)((entry,))

    with pytest.raises(TypeError, match="entries"):
        result_type(entries=payload)


@pytest.mark.parametrize(
    ("result_type", "entries"),
    [
        (PromptBatchDeleteResult, ()),
        (PromptBatchRestoreResult, ()),
    ],
)
def test_batch_results_require_nonempty_entries(result_type, entries):
    with pytest.raises(ValueError, match="entries"):
        result_type(entries=entries)


@pytest.mark.parametrize(
    ("result_type", "entries"),
    [
        (
            PromptBatchDeleteResult,
            (
                PromptDeleteReceiptEntry(2, "Two", "prompt", 3),
                PromptDeleteReceiptEntry(1, "One", "recipe", 4),
            ),
        ),
        (
            PromptBatchDeleteResult,
            (
                PromptDeleteReceiptEntry(1, "One", "prompt", 3),
                PromptDeleteReceiptEntry(1, "Again", "recipe", 4),
            ),
        ),
        (
            PromptBatchRestoreResult,
            (PromptRestoreResultEntry(2, 3), PromptRestoreResultEntry(1, 4)),
        ),
        (
            PromptBatchRestoreResult,
            (PromptRestoreResultEntry(1, 3), PromptRestoreResultEntry(1, 4)),
        ),
    ],
)
def test_batch_results_require_unique_canonical_ascending_ids(result_type, entries):
    with pytest.raises(ValueError, match="canonical"):
        result_type(entries=entries)


def test_batch_models_expose_deterministic_targets_and_hide_identities_from_repr():
    target = PromptBatchTarget(local_id=7, expected_version=3)
    entry = PromptDeleteReceiptEntry(
        local_id=7,
        title="Literal [name]",
        artifact_type="recipe",
        tombstone_version=4,
    )
    deleted = PromptBatchDeleteResult(entries=(entry,))
    restored = PromptBatchRestoreResult(
        entries=(PromptRestoreResultEntry(local_id=7, restored_version=5),)
    )

    assert deleted.targets == (PromptBatchTarget(7, 4),)
    assert restored.entries[0].restored_version == 5
    for value in (target, entry, deleted, restored.entries[0], restored):
        representation = repr(value)
        assert "7" not in representation
        assert "Literal [name]" not in representation
    with pytest.raises(FrozenInstanceError):
        target.expected_version = 8  # type: ignore[misc]
    assert not hasattr(target, "__dict__")
