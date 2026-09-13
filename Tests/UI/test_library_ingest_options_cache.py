"""Invalidation guards for the app-scoped Library ingest-option cache.

`_load_library_ingest_options_from_config` performs 43 `get_cli_setting` reads
and runs from `on_mount`, and the app builds a NEW `LibraryScreen` per visit --
so the result is memoised on the running app (task-24456).

A memoised config view is only as correct as its invalidation key, and the
first version of this cache got that wrong: it keyed on the config GENERATION
alone. Retargeting `TLDW_CONFIG_PATH` selects a different config file and the
loader serves the new file's values *without* advancing the generation, so a
generation-only key kept serving the old file's options. Caught in review by
Qodo on PR #2217 and reproduced before fixing.

These tests pin both halves of the key.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _write_config(root: Path, name: str, chunk_size: int) -> Path:
    path = root / name / "tldw_cli" / "config.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "[first_run]\nsetup_completed = true\n\n"
        "[library.ingest_options.generic]\n"
        f"chunk_size = {chunk_size}\n"
    )
    return path


class _FakeApp:
    """Stand-in for the running App -- the cache's storage scope."""


def test_config_identity_changes_when_the_config_path_is_retargeted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The memo key must move when a DIFFERENT config file becomes effective.

    This is the regression: `_CONFIG_GENERATION` does not advance on a
    retarget, so a key built from it alone is stable across two files with
    different contents.
    """
    from tldw_chatbook.config import current_config_identity

    first = _write_config(tmp_path, "A", 1111)
    second = _write_config(tmp_path, "B", 2222)

    monkeypatch.setenv("TLDW_CONFIG_PATH", str(first))
    before = current_config_identity()
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(second))
    after = current_config_identity()

    assert before != after, (
        "current_config_identity() did not change when TLDW_CONFIG_PATH was "
        "retargeted to a different file. Any cache keyed on it will serve the "
        "previous config's values."
    )
    assert before[0] == after[0], (
        "precondition for this test: the generation is expected NOT to move on "
        "a retarget -- that is exactly why the path belongs in the key"
    )


def test_retargeting_the_config_path_invalidates_the_ingest_option_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A retarget must be served the NEW file's options, not the cached ones."""
    from tldw_chatbook.UI.Screens import library_screen

    first = _write_config(tmp_path, "A", 1111)
    second = _write_config(tmp_path, "B", 2222)
    app = _FakeApp()
    owner = type("Owner", (), {"app": app})()

    monkeypatch.setenv("TLDW_CONFIG_PATH", str(first))
    monkeypatch.setattr(
        library_screen,
        "_read_library_ingest_options_from_config",
        lambda: {
            "transcribe_cpp_configured": False,
            "generic_form_fields": {},
            "type_options": {"generic": {"chunk_size": _chunk_size()}},
        },
    )

    def _chunk_size() -> int:
        from tldw_chatbook.config import get_cli_setting

        return int(get_cli_setting("library.ingest_options.generic", "chunk_size", 0))

    payload_a = library_screen._library_ingest_options_for(owner)
    assert payload_a["type_options"]["generic"]["chunk_size"] == 1111

    cached_again = library_screen._library_ingest_options_for(owner)
    assert cached_again is payload_a, "unchanged config should hit the cache"

    monkeypatch.setenv("TLDW_CONFIG_PATH", str(second))
    payload_b = library_screen._library_ingest_options_for(owner)
    assert payload_b["type_options"]["generic"]["chunk_size"] == 2222, (
        "the cache served the previous config file's ingest options after "
        "TLDW_CONFIG_PATH was retargeted"
    )


def test_a_screen_without_a_running_app_always_reads_fresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No app means no navigation to amortise across -- never cache.

    This is what keeps a test that stubs `get_cli_setting` on an unmounted
    screen seeing its own values, and it is the production-correct behaviour
    for any caller that has swapped the settings source.
    """
    from tldw_chatbook.UI.Screens import library_screen

    calls: list[int] = []

    def _fake_read() -> dict:
        calls.append(1)
        return {
            "transcribe_cpp_configured": False,
            "generic_form_fields": {},
            "type_options": {},
        }

    monkeypatch.setattr(
        library_screen, "_read_library_ingest_options_from_config", _fake_read
    )

    class _NoApp:
        @property
        def app(self):  # noqa: ANN201 - mirrors Textual raising off-app
            raise RuntimeError("no running app")

    owner = _NoApp()
    library_screen._library_ingest_options_for(owner)
    library_screen._library_ingest_options_for(owner)
    assert len(calls) == 2, "an app-less owner must not be served a cached payload"
