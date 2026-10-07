"""Lossless hook inventory and exact execution identity."""

import pytest

from tldw_chatbook.Agents import run_hooks


def test_disabled_invalid_row_is_visible_without_execution_authority():
    inventory = run_hooks.inspect_hooks_config(
        {
            "hooks": {
                "hook": [
                    {
                        "id": "one",
                        "enabled": False,
                        "event": "unknown",
                        "command": "bad",
                    }
                ]
            }
        }
    )
    assert len(inventory.rows) == 1
    assert inventory.rows[0].error
    assert not inventory.requires_authority


def test_fingerprint_normalizes_timeout_and_preserves_argument_boundaries():
    first = run_hooks.HookSpec("PreToolUse", ("python3", "guard.py", "a b"), "fs_*", 5)
    equivalent = run_hooks.HookSpec("PreToolUse", first.command, "fs_*", 5.0)
    changed = run_hooks.HookSpec(
        "PreToolUse", ("python3", "guard.py", "a", "b"), "fs_*", 5
    )
    assert run_hooks.fingerprint_hook(first) == run_hooks.fingerprint_hook(equivalent)
    assert run_hooks.fingerprint_hook(first) != run_hooks.fingerprint_hook(changed)


def test_legacy_reorder_preserves_definition_keys():
    first = {"event": "Stop", "command": ["python3", "first.py"]}
    second = {"event": "Stop", "command": ["python3", "second.py"]}
    original = run_hooks.inspect_hooks_config({"hooks": {"hook": [first, second]}})
    reordered = run_hooks.inspect_hooks_config({"hooks": {"hook": [second, first]}})
    assert original.rows[0].key == reordered.rows[1].key
    assert original.rows[1].key == reordered.rows[0].key


def test_duplicate_explicit_ids_invalidate_every_collision():
    row = {"id": "one", "event": "Stop", "command": ["python3"]}
    inventory = run_hooks.inspect_hooks_config({"hooks": {"hook": [row, row]}})
    assert all(item.error and item.spec is None for item in inventory.rows)
    assert len({item.key for item in inventory.rows}) == 2
    assert not run_hooks.load_hooks_config({"hooks": {"hook": [row, row]}}).hooks


@pytest.mark.parametrize("section", [False, [], {"enabled": "yes"}, {"hook": "bad"}])
def test_malformed_containers_require_authority(section):
    assert run_hooks.inspect_hooks_config({"hooks": section}).requires_authority


@pytest.mark.parametrize(
    "section", [{}, {"hook": []}, {"enabled": False, "hook": "bad"}]
)
def test_verified_empty_or_master_disabled_needs_no_authority(section):
    assert not run_hooks.inspect_hooks_config({"hooks": section}).requires_authority


@pytest.mark.parametrize(
    "change",
    [
        {"id": ""},
        {"id": 7},
        {"enabled": 1},
        {"event": "unsupported"},
        {"command": "python3"},
        {"command": []},
        {"command": [""]},
        {"command": ["python3", 7]},
        {"command": ["python3", "bad\x00arg"]},
        {"timeout_s": True},
        {"timeout_s": 0},
        {"timeout_s": float("inf")},
        {"timeout_s": float("nan")},
        {"timeout_s": 10**400},
        {"matcher": "fs_*"},
    ],
)
def test_invalid_definition_retained_but_not_executable(change):
    raw = {"event": "Stop", "command": ["python3"]} | change
    inventory = run_hooks.inspect_hooks_config({"hooks": {"hook": [raw]}})
    assert inventory.rows[0].error
    assert inventory.requires_authority
    assert not run_hooks.load_hooks_config({"hooks": {"hook": [raw]}}).hooks


def test_valid_disabled_definition_not_in_execution_projection():
    raw = {"event": "Stop", "command": ["python3"], "enabled": False}
    inventory = run_hooks.inspect_hooks_config({"hooks": {"hook": [raw]}})
    assert inventory.rows[0].spec is not None
    assert not run_hooks.load_hooks_config({"hooks": {"hook": [raw]}}).hooks
