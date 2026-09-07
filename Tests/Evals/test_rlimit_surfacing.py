import pytest

from tldw_chatbook.Evals import specialized_runners as sr


def test_memory_limit_not_enforced_on_darwin(monkeypatch):
    monkeypatch.setattr(sr.platform, "system", lambda: "Darwin")
    assert sr._memory_limit_enforced() is False


def test_memory_limit_enforced_on_linux(monkeypatch):
    monkeypatch.setattr(sr.platform, "system", lambda: "Linux")
    assert sr._memory_limit_enforced() is True


def test_warns_and_records_when_unenforced(monkeypatch):
    # A helper that builds the results dict + appends the sandbox warning when
    # memory isn't enforced. On Darwin it must surface a warning entry.
    monkeypatch.setattr(sr, "_memory_limit_enforced", lambda: False)
    warnings = sr._sandbox_warnings()
    assert any("memory" in w.lower() for w in warnings)


def test_no_warning_when_enforced(monkeypatch):
    monkeypatch.setattr(sr, "_memory_limit_enforced", lambda: True)
    assert sr._sandbox_warnings() == []
