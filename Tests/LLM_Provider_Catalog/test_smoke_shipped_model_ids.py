"""TASK-34100.2 review round 1 (C10, F9): the live model-id smoke, offline.

``Helper_Scripts/smoke_shipped_model_ids.py`` (AC#6) is run by hand against
real providers. Its own logic -- parsing the shipped template file,
mapping a 404 to 'retired' and failing the run, stopping a provider on a
refused key, telling a transient 5xx from a rejected id -- broke silently
until someone ran it live. ``urlopen`` is replaced here; nothing touches a
network or the app's profile.
"""

from __future__ import annotations

import importlib.util
import io
import sys
import urllib.error
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "Helper_Scripts" / "smoke_shipped_model_ids.py"


@pytest.fixture(scope="module")
def smoke():
    name = "smoke_shipped_model_ids"
    spec = importlib.util.spec_from_file_location(name, _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve their module by name
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)


def _http_error(code: int, message: str = "") -> urllib.error.HTTPError:
    body = f'{{"error": {{"message": "{message}"}}}}'.encode()
    return urllib.error.HTTPError("https://example.test", code, "x", {}, io.BytesIO(body))


class _Answer:
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def _route(monkeypatch, smoke, answers: dict[str, object]) -> list[str]:
    asked: list[str] = []

    def urlopen(request, timeout):
        url = request.full_url
        body = (request.data or b"").decode()
        model = next(
            (model for model in answers if model in url or f'"{model}"' in body), None
        )
        asked.append(model)
        answer = answers.get(model, "ok")
        if isinstance(answer, Exception):
            raise answer
        return _Answer()

    monkeypatch.setattr(smoke.urllib.request, "urlopen", urlopen)
    return asked


def test_the_template_parse_lists_the_google_default_and_catalog(smoke) -> None:
    from tldw_chatbook.config import DEFAULT_CONFIG_FROM_TOML

    ids = smoke.shipped_ids(smoke.shipped_template(), "google")

    assert ids[0] == DEFAULT_CONFIG_FROM_TOML["api_settings"]["google"]["model"]
    assert set(DEFAULT_CONFIG_FROM_TOML["providers"]["Google"]) <= set(ids)


@pytest.mark.parametrize(
    ("error", "verdict"),
    [
        (_http_error(404, "no longer available"), "retired"),
        (_http_error(429, "quota"), "quota"),
        (_http_error(400, "bad id"), "rejected"),
        # F9: Google's 503 'high demand' is transient, not a bad id.
        (_http_error(503, "This model is currently experiencing high demand"), "unavailable"),
        (_http_error(500, "internal"), "unavailable"),
        (_http_error(401, "bad key"), "key"),
        # Review round 2 (F11): no HTTP answer at all -- documented now.
        (TimeoutError("timed out"), "unreachable"),
        (urllib.error.URLError("no route"), "unreachable"),
    ],
)
def test_each_answer_maps_to_one_verdict(smoke, monkeypatch, error, verdict) -> None:
    _route(monkeypatch, smoke, {"gemini-x": error})

    assert smoke.probe("google", "gemini-x", "AIza-test").verdict == verdict


def test_a_retired_id_fails_the_run_and_a_refused_key_stops_its_provider(
    smoke, monkeypatch, capsys
) -> None:
    template = {
        "api_settings": {
            "google": {"model": "gemini-live"},
            "openai": {"model": "gpt-a"},
        },
        "providers": {"Google": ["gemini-gone"], "OpenAI": ["gpt-b"]},
    }
    monkeypatch.setattr(smoke, "shipped_template", lambda: template)
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test-smoke")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-smoke")
    asked = _route(
        monkeypatch,
        smoke,
        {
            "gemini-live": "ok",
            "gemini-gone": _http_error(404, "gone"),
            "gpt-a": _http_error(401, "bad key"),
            "gpt-b": "ok",
        },
    )

    status = smoke.main(["--providers", "google,openai"])

    out = capsys.readouterr().out
    assert status == 1
    assert "RETIRED shipped ids: google/gemini-gone" in out
    assert "gpt-b" not in asked  # the refused key stopped OpenAI's remaining ids
    assert "AIza-test-smoke" not in out and "sk-test-smoke" not in out


def test_a_catalog_ids_transient_outage_never_fails_the_run(
    smoke, monkeypatch, capsys
) -> None:
    """F9: a 503 'high demand' on a catalog id is transient. It is named in
    the summary (review v5r2 G2C-R3-1: never hidden) but does not fail the
    run; only the default is held to answering (below)."""
    template = {
        "api_settings": {"google": {"model": "gemini-live"}},
        "providers": {"Google": ["gemini-busy"]},
    }
    monkeypatch.setattr(smoke, "shipped_template", lambda: template)
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test-smoke")
    _route(monkeypatch, smoke, {"gemini-busy": _http_error(503, "high demand")})

    assert smoke.main(["--providers", "google"]) == 0
    assert "Did not answer (retry later): google/gemini-busy" in capsys.readouterr().out


def test_an_unreachable_catalog_id_never_fails_the_run_and_is_documented(
    smoke, monkeypatch
) -> None:
    template = {
        "api_settings": {"google": {"model": "gemini-live"}},
        "providers": {"Google": ["gemini-slow"]},
    }
    monkeypatch.setattr(smoke, "shipped_template", lambda: template)
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test-smoke")
    _route(monkeypatch, smoke, {"gemini-slow": TimeoutError("timed out")})

    assert smoke.main(["--providers", "google"]) == 0
    assert "``unreachable``" in smoke.__doc__


def _flaky(monkeypatch, smoke, model: str, answers: list[object]) -> list[str]:
    """``model`` answers each of ``answers`` in turn (the last repeats)."""
    asked: list[str] = []

    def urlopen(request, timeout):
        asked.append(request.full_url)
        answer = answers[min(len(asked), len(answers)) - 1]
        if isinstance(answer, Exception):
            raise answer
        return _Answer()

    monkeypatch.setattr(smoke.urllib.request, "urlopen", urlopen)
    return asked


@pytest.mark.parametrize(
    "silence",
    [_http_error(503, "high demand"), TimeoutError("timed out")],
    ids=("503", "timeout"),
)
def test_a_shipped_default_that_never_answers_fails_the_run(
    smoke, monkeypatch, capsys, silence
) -> None:
    """Review v5r2 G2C-R3-1: the shipped Google default (gemini-flash-latest)
    timed out or answered 503 on every smoke run for five weeks, and the
    script exited 0 each time ('unreachable'/'unavailable' never fail a
    run). The default is what a first chat sends to, so one that does not
    answer in three attempts fails the run and is named."""
    template = {"api_settings": {"google": {"model": "gemini-default"}}, "providers": {}}
    monkeypatch.setattr(smoke, "shipped_template", lambda: template)
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test-smoke")
    asked = _flaky(monkeypatch, smoke, "gemini-default", [silence])

    status = smoke.main(["--providers", "google"])

    out = capsys.readouterr().out
    assert status == 1
    assert len(asked) == smoke.DEFAULT_ATTEMPTS == 3
    assert "Shipped default did not answer (3 tries): google/gemini-default" in out
    assert "AIza-test-smoke" not in out


def test_a_shipped_default_that_answers_on_a_retry_passes(smoke, monkeypatch) -> None:
    template = {"api_settings": {"google": {"model": "gemini-default"}}, "providers": {}}
    monkeypatch.setattr(smoke, "shipped_template", lambda: template)
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test-smoke")
    asked = _flaky(
        monkeypatch, smoke, "gemini-default", [_http_error(503, "high demand"), "ok"]
    )

    assert smoke.main(["--providers", "google"]) == 0
    assert len(asked) == 2
