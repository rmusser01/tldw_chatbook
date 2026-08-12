"""Strict contracts for atomic local Prompt batch mutations."""

import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from tldw_chatbook.Prompt_Management.prompt_batch_models import (
    PromptBatchDeleteResult,
    PromptBatchRestoreResult,
    PromptBatchTarget,
    PromptDeleteReceiptEntry,
    PromptRestoreResultEntry,
    validate_prompt_batch_targets,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_isolated_python(tmp_path: Path, code: str) -> subprocess.CompletedProcess[str]:
    """Run an import probe without inherited profile or module state."""
    data_home = tmp_path / "data"
    config_home = tmp_path / "config"
    home = tmp_path / "home"
    for path in (data_home, config_home, home):
        path.mkdir(parents=True, exist_ok=True)
    environment = {
        **os.environ,
        "TLDW_TEST_MODE": "1",
        "XDG_DATA_HOME": str(data_home),
        "XDG_CONFIG_HOME": str(config_home),
        "HOME": str(home),
        "PYTHONPATH": str(REPO_ROOT),
    }
    environment.pop("PYTEST_CURRENT_TEST", None)
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )


def test_config_first_then_batch_model_imports_in_fresh_process(tmp_path: Path):
    result = _run_isolated_python(
        tmp_path,
        """
import tldw_chatbook.config
from tldw_chatbook.Prompt_Management.prompt_batch_models import PromptBatchTarget
PromptBatchTarget(1, 1)
""",
    )

    assert result.returncode == 0, result.stderr


def test_app_imports_in_fresh_process_without_prompt_config_cycle(tmp_path: Path):
    result = _run_isolated_python(tmp_path, "import tldw_chatbook.app")

    assert result.returncode == 0, result.stderr


def test_batch_model_import_does_not_eagerly_import_prompt_services(
    tmp_path: Path,
):
    result = _run_isolated_python(
        tmp_path,
        """
import sys
from tldw_chatbook.Prompt_Management.prompt_batch_models import PromptBatchTarget
forbidden = {
    'tldw_chatbook.config',
    'tldw_chatbook.Prompt_Management.Prompts_Interop',
    'tldw_chatbook.Prompt_Management.local_prompt_service',
    'tldw_chatbook.Prompt_Management.prompt_chatbook_scope_service',
    'tldw_chatbook.Prompt_Management.server_prompt_service',
}
assert forbidden.isdisjoint(sys.modules), forbidden.intersection(sys.modules)
PromptBatchTarget(1, 1)
""",
    )

    assert result.returncode == 0, result.stderr


def test_prompt_management_lazy_public_exports_preserve_class_identities():
    import tldw_chatbook.Prompt_Management as prompt_management

    from tldw_chatbook.Prompt_Management import (
        LocalPromptService,
        PromptChatbookBackend,
        PromptChatbookScopeService,
        ServerPromptService,
    )
    from tldw_chatbook.Prompt_Management.local_prompt_service import (
        LocalPromptService as DirectLocalPromptService,
    )
    from tldw_chatbook.Prompt_Management.prompt_chatbook_scope_service import (
        PromptChatbookBackend as DirectPromptChatbookBackend,
        PromptChatbookScopeService as DirectPromptChatbookScopeService,
    )
    from tldw_chatbook.Prompt_Management.server_prompt_service import (
        ServerPromptService as DirectServerPromptService,
    )

    assert LocalPromptService is DirectLocalPromptService
    assert PromptChatbookBackend is DirectPromptChatbookBackend
    assert PromptChatbookScopeService is DirectPromptChatbookScopeService
    assert ServerPromptService is DirectServerPromptService
    assert prompt_management.__all__ == [
        "LocalPromptService",
        "PromptChatbookBackend",
        "PromptChatbookScopeService",
        "ServerPromptService",
    ]


def test_prompt_management_submodule_imports_remain_available(tmp_path: Path):
    result = _run_isolated_python(
        tmp_path,
        """
from tldw_chatbook.Prompt_Management import Prompts_Interop
from tldw_chatbook.Prompt_Management import prompt_scope_service
assert Prompts_Interop.__name__.endswith('.Prompts_Interop')
assert prompt_scope_service.__name__.endswith('.prompt_scope_service')
""",
    )

    assert result.returncode == 0, result.stderr


class _IntSubclass(int):
    """An integer lookalike that strict public boundaries must reject."""


class _StrSubclass(str):
    """A text lookalike that strict public boundaries must reject."""


class _TupleSubclass(tuple):
    """A tuple lookalike that strict public boundaries must reject."""


class _PromptBatchTargetSubclass(PromptBatchTarget):
    """A target lookalike that strict public boundaries must reject."""


def _forged_batch_target(local_id, expected_version) -> PromptBatchTarget:
    target = object.__new__(PromptBatchTarget)
    object.__setattr__(target, "local_id", local_id)
    object.__setattr__(target, "expected_version", expected_version)
    return target


@pytest.mark.parametrize(
    "targets",
    [
        [],
        _TupleSubclass((PromptBatchTarget(1, 1),)),
        (object(),),
        (_PromptBatchTargetSubclass(1, 1),),
    ],
)
def test_prompt_batch_target_validator_requires_exact_tuple_and_target_types(targets):
    with pytest.raises(TypeError, match="targets"):
        validate_prompt_batch_targets(targets)


def test_prompt_batch_target_validator_requires_nonempty_unique_ids_without_leaking_ids():
    with pytest.raises(ValueError, match="non-empty"):
        validate_prompt_batch_targets(())

    targets = (
        PromptBatchTarget(71_234_567, 81_234_567),
        PromptBatchTarget(71_234_567, 91_234_567),
    )
    with pytest.raises(ValueError) as raised:
        validate_prompt_batch_targets(targets)

    assert "unique local IDs" in str(raised.value)
    assert "71234567" not in str(raised.value)
    assert "81234567" not in str(raised.value)
    assert "91234567" not in str(raised.value)


def test_prompt_batch_target_validator_canonicalizes_and_preserves_identity_when_sorted():
    first = PromptBatchTarget(7, 3)
    second = PromptBatchTarget(9, 2)
    canonical = (first, second)

    assert validate_prompt_batch_targets(canonical) is canonical
    assert validate_prompt_batch_targets((second, first)) == canonical


@pytest.mark.parametrize(
    ("local_id", "expected_version", "field"),
    [
        (True, 1, "local_id"),
        (0, 1, "local_id"),
        (-1, 1, "local_id"),
        (2**63, 1, "local_id"),
        (1, False, "expected_version"),
        (1, 0, "expected_version"),
        (1, -1, "expected_version"),
        (1, 2**63, "expected_version"),
    ],
)
def test_prompt_batch_target_validator_revalidates_exact_target_fields(
    local_id, expected_version, field
):
    target = _forged_batch_target(local_id, expected_version)

    with pytest.raises(ValueError, match=field):
        validate_prompt_batch_targets((target,))


@pytest.mark.parametrize(
    ("missing_field", "private_local_id", "private_version"),
    [
        ("local_id", 71_234_567, 81_234_567),
        ("expected_version", 91_234_567, 101_234_567),
    ],
)
def test_prompt_batch_target_validator_bounds_missing_slots_without_private_values(
    missing_field, private_local_id, private_version
):
    target = PromptBatchTarget(private_local_id, private_version)
    object.__delattr__(target, missing_field)

    assert str(private_local_id) not in str(target)
    assert str(private_version) not in repr(target)
    with pytest.raises(ValueError, match=missing_field) as raised:
        validate_prompt_batch_targets((target,))

    error_text = f"{raised.value!s} {raised.value!r}"
    assert str(private_local_id) not in error_text
    assert str(private_version) not in error_text


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
    target = PromptBatchTarget(local_id=71_234_567, expected_version=81_234_567)
    entry = PromptDeleteReceiptEntry(
        local_id=91_234_567,
        title="Literal [private-991827]",
        artifact_type="recipe",
        tombstone_version=101_234_567,
    )
    deleted = PromptBatchDeleteResult(entries=(entry,))
    restored = PromptBatchRestoreResult(
        entries=(
            PromptRestoreResultEntry(
                local_id=111_234_567, restored_version=121_234_567
            ),
        )
    )

    assert deleted.targets == (PromptBatchTarget(91_234_567, 101_234_567),)
    assert restored.entries[0].restored_version == 121_234_567

    target_repr = repr(target)
    assert "71234567" not in target_repr
    assert "81234567" not in target_repr

    entry_repr = repr(entry)
    assert "91234567" not in entry_repr
    assert "Literal [private-991827]" not in entry_repr
    assert "101234567" not in entry_repr

    deleted_repr = repr(deleted)
    assert "91234567" not in deleted_repr
    assert "Literal [private-991827]" not in deleted_repr
    assert "101234567" not in deleted_repr

    restore_target_repr = repr(deleted.targets[0])
    assert "91234567" not in restore_target_repr
    assert "101234567" not in restore_target_repr

    restored_entry_repr = repr(restored.entries[0])
    assert "111234567" not in restored_entry_repr
    assert "121234567" not in restored_entry_repr

    restored_repr = repr(restored)
    assert "111234567" not in restored_repr
    assert "121234567" not in restored_repr
    with pytest.raises(FrozenInstanceError):
        target.expected_version = 131_234_567  # type: ignore[misc]
    assert not hasattr(target, "__dict__")
