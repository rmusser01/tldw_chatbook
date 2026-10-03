"""What every engine preset's real API says without a key (TASK-33640).

``Tests/fixtures/cloud_live/noauth/<key>.json`` records, for each preset with
a shipped URL, how its real API answered with no key and with a deliberately
fake one, captured by ``capture.py --no-auth`` (no tokens spent). Probed
2026-09-30 with the app's own User-Agent. That matters: Cloudflare answers
urllib's default User-Agent with 403 ``error code: 1010`` at six of these
providers.

These tests pin that evidence against the app:

- every chat route exists at the shipped URL;
- a bad key reaches the user as an authentication failure, not a generic
  rejection;
- every public model listing parses through the real discovery parser;
- every seeded model is still listed by its provider.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from Tests.fixtures.cloud_live.capture import _config_seeds
from tldw_chatbook.Chat.Chat_Deps import ChatAuthenticationError
from tldw_chatbook.LLM_Calls.hosted_chat import _raise_http_error
from tldw_chatbook.LLM_Provider_Catalog.openai_compatible_model_discovery import (
    normalize_models_response,
)
from tldw_chatbook.provider_registry import RECORDS_BY_KEY

PROBE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "cloud_live" / "noauth"
PROBES = sorted(path.stem for path in PROBE_DIR.glob("*.json"))

# Providers that check the model before the key, so a bad key with a model it
# does not recognise answers 404 "model not found". Fireworks answers that for
# a well-known model id, and GMI's listing needs a key, so neither can be
# probed with a model it accepts. Whether a real but bad key reaches the user
# as an auth failure there needs a live key.
MODEL_BEFORE_KEY = frozenset({"fireworks", "gmi"})


@pytest.fixture(params=PROBES, ids=lambda name: name)
def probe(request: pytest.FixtureRequest) -> dict[str, Any]:
    """One provider's no-key probe, loaded from disk."""
    return json.loads((PROBE_DIR / f"{request.param}.json").read_text(encoding="utf-8"))


def _text(body: object) -> str:
    return (body if isinstance(body, str) else json.dumps(body)).lower()


def test_probe_covers_the_shipped_url(probe: dict[str, Any]) -> None:
    """Each probe is for a real engine preset, at the URL the preset ships."""
    record = RECORDS_BY_KEY[probe["server"]]
    assert record.engine_driven
    assert probe["base_url"] == record.default_base_url.rstrip("/")


def test_chat_route_exists_at_the_shipped_url(probe: dict[str, Any]) -> None:
    """The chat route answers: an auth or model error, never a missing route or a block."""
    for round_name in ("chat_no_key", "chat_bad_key"):
        answer = probe[round_name]
        assert answer["status"] > 0, f"{round_name}: no HTTP answer"
        assert "error code: 1010" not in _text(answer["body"]), f"{round_name}: Cloudflare block"
        if answer["status"] == 404:
            assert "model" in _text(answer["body"]), f"{round_name}: route not found"


def test_bad_key_reaches_the_user_as_an_authentication_failure(probe: dict[str, Any]) -> None:
    """The engine maps the provider's bad-key status to "authentication failed. Check the API key."."""
    answer = probe["chat_bad_key"]
    if probe["server"] in MODEL_BEFORE_KEY:
        assert answer["status"] == 404 and "model" in _text(answer["body"])
        pytest.skip("model checked before key; needs a live key to settle")
    with pytest.raises(ChatAuthenticationError):
        _raise_http_error(probe["server"], answer["status"])


def test_model_before_key_list_matches_the_evidence() -> None:
    """Only the listed providers answer a bad key with a model error."""
    observed = {
        name for name in PROBES
        if json.loads((PROBE_DIR / f"{name}.json").read_text(encoding="utf-8"))["chat_bad_key"]["status"] == 404
    }
    assert observed == MODEL_BEFORE_KEY & set(PROBES)


def test_public_listing_parses_through_discovery(probe: dict[str, Any]) -> None:
    """A public model listing yields every listed model through the discovery parser."""
    listing = probe["listing"]
    if listing["status"] != 200:
        pytest.skip(f"listing needs a key (HTTP {listing['status']})")
    entries = listing["entries"]
    models = normalize_models_response(
        {"data": entries},
        provider=probe["server"],
        provider_list_key=RECORDS_BY_KEY[probe["server"]].config_key,
        endpoint_fingerprint="probe",
        now_iso="2026-09-30T00:00:00Z",
    )
    assert len(models) == len({entry["id"] for entry in entries})


def test_seeded_models_are_still_listed(probe: dict[str, Any]) -> None:
    """A seeded preset with a public listing lists every seeded model id."""
    record = RECORDS_BY_KEY[probe["server"]]
    listing = probe["listing"]
    seeds = _config_seeds(record)
    if record.auto_refresh or not seeds or listing["status"] != 200:
        pytest.skip("not a seeded preset with a public listing")
    listed = {entry["id"] for entry in listing["entries"]}
    assert [seed for seed in seeds if seed not in listed] == []
