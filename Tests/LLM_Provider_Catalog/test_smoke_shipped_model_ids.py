"""TASK-34100.2 review round 1 (C10, F9): the live model-id smoke, offline.

``Helper_Scripts/smoke_shipped_model_ids.py`` (AC#6) is run by hand against
real providers. Its own logic -- parsing the template out of config.py,
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


def test_a_transient_outage_never_fails_the_run(smoke, monkeypatch) -> None:
    template = {"api_settings": {"google": {"model": "gemini-live"}}, "providers": {}}
    monkeypatch.setattr(smoke, "shipped_template", lambda: template)
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test-smoke")
    _route(monkeypatch, smoke, {"gemini-live": _http_error(503, "high demand")})

    assert smoke.main(["--providers", "google"]) == 0


def test_an_unreachable_id_never_fails_the_run_and_is_documented(
    smoke, monkeypatch
) -> None:
    template = {"api_settings": {"google": {"model": "gemini-live"}}, "providers": {}}
    monkeypatch.setattr(smoke, "shipped_template", lambda: template)
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test-smoke")
    _route(monkeypatch, smoke, {"gemini-live": TimeoutError("timed out")})

    assert smoke.main(["--providers", "google"]) == 0
    assert "``unreachable``" in smoke.__doc__
