"""Tests for optional_deps.ensure_svg_rendering (TASK-222)."""

import os

import pytest

from tldw_chatbook.Utils import optional_deps


@pytest.fixture(autouse=True)
def _reset_svg_cache(monkeypatch):
    """Each test starts from the unchecked state."""
    monkeypatch.setattr(optional_deps, "_svg_rendering_available", None)


def test_registered_in_dependencies_available():
    assert "svg_rendering" in optional_deps.DEPENDENCIES_AVAILABLE


def test_success_is_cached(monkeypatch):
    calls = []

    def fake_check(module, feature=None):
        calls.append(module)
        return True

    monkeypatch.setattr(optional_deps, "check_dependency", fake_check)
    assert optional_deps.ensure_svg_rendering() is True
    assert optional_deps.ensure_svg_rendering() is True
    assert calls == ["cairosvg"]  # second call served from cache


def test_failure_is_cached(monkeypatch):
    calls = []

    def fake_check(module, feature=None):
        calls.append(module)
        return False

    monkeypatch.setattr(optional_deps, "check_dependency", fake_check)
    assert optional_deps.ensure_svg_rendering() is False
    assert optional_deps.ensure_svg_rendering() is False
    assert calls == ["cairosvg"]


def test_darwin_sets_fallback_path_when_unset(monkeypatch):
    monkeypatch.setattr(optional_deps.sys, "platform", "darwin")
    monkeypatch.setattr(optional_deps, "check_dependency", lambda m, f=None: True)
    monkeypatch.setattr(
        optional_deps.os.path, "isdir", lambda p: p == "/opt/homebrew/lib"
    )
    monkeypatch.delenv("DYLD_FALLBACK_LIBRARY_PATH", raising=False)
    optional_deps.ensure_svg_rendering()
    value = os.environ["DYLD_FALLBACK_LIBRARY_PATH"]
    entries = value.split(":")
    assert entries[-1] == "/opt/homebrew/lib"
    # the dyld default fallback chain is preserved ahead of the append
    assert "/usr/local/lib" in entries and "/usr/lib" in entries


def test_darwin_appends_to_existing_value(monkeypatch):
    monkeypatch.setattr(optional_deps.sys, "platform", "darwin")
    monkeypatch.setattr(optional_deps, "check_dependency", lambda m, f=None: True)
    monkeypatch.setattr(
        optional_deps.os.path, "isdir", lambda p: p == "/opt/homebrew/lib"
    )
    monkeypatch.setenv("DYLD_FALLBACK_LIBRARY_PATH", "/custom/lib")
    optional_deps.ensure_svg_rendering()
    assert os.environ["DYLD_FALLBACK_LIBRARY_PATH"] == "/custom/lib:/opt/homebrew/lib"


def test_darwin_append_is_idempotent(monkeypatch):
    monkeypatch.setattr(optional_deps.sys, "platform", "darwin")
    monkeypatch.setattr(optional_deps, "check_dependency", lambda m, f=None: True)
    monkeypatch.setattr(
        optional_deps.os.path, "isdir", lambda p: p == "/opt/homebrew/lib"
    )
    monkeypatch.setenv("DYLD_FALLBACK_LIBRARY_PATH", "/custom/lib:/opt/homebrew/lib")
    optional_deps.ensure_svg_rendering()
    assert (
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] == "/custom/lib:/opt/homebrew/lib"
    )


def test_non_darwin_leaves_env_alone(monkeypatch):
    monkeypatch.setattr(optional_deps.sys, "platform", "linux")
    monkeypatch.setattr(optional_deps, "check_dependency", lambda m, f=None: True)
    monkeypatch.delenv("DYLD_FALLBACK_LIBRARY_PATH", raising=False)
    optional_deps.ensure_svg_rendering()
    assert "DYLD_FALLBACK_LIBRARY_PATH" not in os.environ


def test_oserror_from_check_dependency_is_gated_off_and_cached(monkeypatch):
    """A native dlopen failure (cairocffi -> OSError on a cairo-less Mac)
    must gate SVG support off instead of crashing callers, and must not be
    re-attempted once cached."""
    calls = []

    def fake_check(module, feature=None):
        calls.append(module)
        raise OSError('cannot load library \'libcairo.so.2\': dlopen failed')

    monkeypatch.setattr(optional_deps, "check_dependency", fake_check)
    assert optional_deps.ensure_svg_rendering() is False
    assert optional_deps.ensure_svg_rendering() is False
    assert calls == ["cairosvg"]  # second call served from cache, no re-raise
    assert optional_deps.DEPENDENCIES_AVAILABLE["svg_rendering"] is False
