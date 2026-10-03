#!/usr/bin/env python3
"""Capture raw responses from every engine preset's real API (TASK-33640).

Registry-driven: every engine-driven record in ``tldw_chatbook.provider_registry``
(except the ``custom-hosted`` execution key) is a candidate, and only those
whose API key is available are captured -- the rest skip cleanly. Per
provider, up to four rounds, each built the way the engine builds it (base
URL + suffix, ``api-key`` vs Bearer auth, ``max_tokens`` spelling, extra body
fields, ``stream_options`` when the record asks, ``tool_choice`` only where
the record allows it):

1. ``GET {base}/models``      (only when the record has a discovery route),
2. plain chat                  (tiny prompt),
3. tool-call round             (one function tool; skipped when the record
                                ships native tools off),
4. streamed chat               (every SSE ``data`` payload, ``[DONE]`` included).

Each capture is written RAW to ``Tests/fixtures/cloud_live/<key>.json``; the
offline replay test (``Tests/LLM_Calls/test_live_capture_replay.py``) then
runs it through the real engine. After each capture this script also prints
the response KEY NAMES the record's allowances do not cover -- never content.

Keys: taken from the environment, or from a user-owned keys file (default
``~/.config/tldw-live/keys.env``; ``NAME=value`` lines, ``export`` and quotes
allowed; the environment wins). A key is only ever placed in the request
header at call time: it is never printed, logged, or written to a fixture.

Per-provider environment overrides (``<KEY>`` is the record key upper-cased):

- ``TLDW_LIVE_<KEY>_MODEL``     the model to use (required for Azure, where
                                 it is your deployment name),
- ``TLDW_LIVE_<KEY>_BASE_URL``  the base URL (required for Azure, Cloudflare
                                 and Databricks, whose URL is per account),
- ``TLDW_LIVE_<KEY>_<SETTING>`` a config-sourced header value (e.g.
                                 ``TLDW_LIVE_WANDB_PROJECT``,
                                 ``TLDW_LIVE_CLOUDFLARE_GATEWAY_ID``).

Usage (repo root)::

    .venv/bin/python Tests/fixtures/cloud_live/capture.py            # every available key
    .venv/bin/python Tests/fixtures/cloud_live/capture.py together nvidia
    .venv/bin/python Tests/fixtures/cloud_live/capture.py --list     # who would run, and why not
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator  # noqa: E402

from tldw_chatbook.Utils.path_validation import validate_path_simple  # noqa: E402
from tldw_chatbook.provider_registry import ALL_RECORDS, ProviderRecord  # noqa: E402

try:  # the app sends requests' default User-Agent; match it exactly
    from requests.utils import default_user_agent as _requests_user_agent

    USER_AGENT = _requests_user_agent()
except ImportError:  # pragma: no cover - requests ships with the app
    USER_AGENT = "python-requests"
# Cloudflare answers urllib's own "Python-urllib/3.x" with 403 "error code: 1010"
# at Together, Cerebras, GMI, W&B, OpenCode Zen and Command Code (probed
# 2026-09-30), so every request here carries the app's User-Agent instead.

FIXTURE_DIR = Path(__file__).resolve().parent
DEFAULT_KEYS_FILE = Path.home() / ".config" / "tldw-live" / "keys.env"
MAX_STREAM_EVENTS = 5000
MAX_LISTED_MODELS = 20
MAX_TOKENS = 200
PER_ACCOUNT = "<per-account>"
REDACTED = "<redacted-credential>"
_ENV_VAR_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# Closed shapes of the engine's strict parser (hosted_chat.py); anything else
# must be in the record's allowances.
KNOWN_TOP = frozenset({"id", "object", "created", "model", "system_fingerprint", "choices", "usage"})
KNOWN_CHOICE = frozenset({"index", "message", "finish_reason"})
KNOWN_STREAM_CHOICE = frozenset({"index", "delta", "finish_reason", "usage"})
KNOWN_MESSAGE = frozenset({"role", "content", "reasoning_content", "tool_calls"})

PLAIN_MESSAGES = [{"role": "user", "content": "Say ok."}]
TOOL_MESSAGES = [
    {"role": "user", "content": "What is the weather in Tokyo? Use the get_weather tool."}
]
TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}
# Model ids that are clearly not chat models, skipped when choosing from a listing.
_NON_CHAT = re.compile(
    r"embed|rerank|whisper|tts|speech|audio|image|flux|sdxl|diffusion|moderation|guard|ocr|clip",
    re.IGNORECASE,
)
_CHEAP = re.compile(r"mini|flash|small|lite|nano|haiku|turbo|[^0-9](1|3|7|8)b", re.IGNORECASE)


def engine_presets() -> list[ProviderRecord]:
    """Every engine preset a user can configure, in registry order."""
    return [r for r in ALL_RECORDS if r.engine_driven and r.key != "custom-hosted"]


def read_keys_file(path: Path) -> dict[str, str]:
    """Parse ``NAME=value`` lines; never echoes values.

    Args:
        path: The keys file; a missing file yields no keys.

    Returns:
        Names to values, excluding blanks and comments.
    """
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.removeprefix("export ").split("=", 1)
        value = value.strip().strip('"').strip("'")
        if value:
            values[name.strip()] = value
    return values


def _env_name(record: ProviderRecord, suffix: str) -> str:
    return f"TLDW_LIVE_{record.key.upper().replace('-', '_')}_{suffix}"


def _config_seeds(record: ProviderRecord) -> list[str]:
    """The ``[providers]`` seed list shipped in config.py, read as text.

    Importing ``tldw_chatbook.config`` would load the real app configuration,
    so the seed line is parsed from the source instead.
    """
    source = (REPO_ROOT / "tldw_chatbook" / "config.py").read_text(encoding="utf-8")
    match = re.search(rf"^{re.escape(record.config_key)} = (\[.*\])", source, re.MULTILINE)
    if not match:
        return []
    try:
        seeds = ast.literal_eval(match.group(1))
    except (SyntaxError, ValueError):
        return []
    return [seed for seed in seeds if isinstance(seed, str)]


def _single_line(value: str, limit: int) -> str:
    if len(value) > limit or not value.isprintable():
        raise ValueError(f"must be one printable line of at most {limit} characters")
    return value


class CaptureOverrides(BaseModel):
    """The per-provider ``TLDW_LIVE_<KEY>_*`` settings, validated before any request.

    Values come from the environment or the keys file; blanks mean "not set".
    """

    model_config = ConfigDict(extra="forbid")

    base_url: str | None = None
    model: str | None = None
    api_key_env_var: str | None = None
    headers: dict[str, str] = {}

    @field_validator("base_url")
    @classmethod
    def _check_base_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parts = urllib.parse.urlsplit(_single_line(value, 2048))
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError("must be an http(s) URL with a host")
        if parts.query or parts.fragment or parts.username or parts.password:
            raise ValueError("must not carry a query, fragment or credentials")
        return value

    @field_validator("model")
    @classmethod
    def _check_model(cls, value: str | None) -> str | None:
        return None if value is None else _single_line(value, 256)

    @field_validator("api_key_env_var")
    @classmethod
    def _check_env_var(cls, value: str | None) -> str | None:
        if value is not None and not _ENV_VAR_NAME.fullmatch(value):
            raise ValueError("must be an environment variable name")
        return value

    @field_validator("headers")
    @classmethod
    def _check_headers(cls, value: dict[str, str]) -> dict[str, str]:
        return {name: _single_line(item, 256) for name, item in value.items()}

    @classmethod
    def from_env(cls, record: ProviderRecord, env: dict[str, str]) -> "CaptureOverrides":
        """Read and validate one provider's overrides.

        Args:
            record: The preset whose ``TLDW_LIVE_<KEY>_*`` names are read.
            env: Environment merged with the keys file.

        Returns:
            The validated overrides.

        Raises:
            ValidationError: When a set value is unusable (its value is never echoed).
        """
        def value(suffix: str) -> str | None:
            return env.get(_env_name(record, suffix), "").strip() or None

        return cls(
            base_url=value("BASE_URL"),
            model=value("MODEL"),
            api_key_env_var=value("API_KEY_ENV_VAR"),
            headers={
                header: item
                for header, setting in record.config_headers.items()
                if (item := value(setting.upper())) is not None
            },
        )


class Target:
    """One provider's resolved capture settings (the key is never exposed)."""

    def __init__(self, record: ProviderRecord, env: dict[str, str]) -> None:
        self.record = record
        self.invalid_override: str | None = None
        try:
            overrides = CaptureOverrides.from_env(record, env)
        except ValidationError as error:
            fields = sorted({str(item["loc"][0]) for item in error.errors()})
            self.invalid_override = ", ".join(fields)
            overrides = CaptureOverrides()
        # The engine reads a configured api_key_env_var before the record's
        # candidates (StepFun's STEP_API_KEY, Meta's MODEL_API_KEY); so does this.
        names = ((overrides.api_key_env_var,) if overrides.api_key_env_var else ()) + tuple(
            record.api_key_env_candidates
        )
        self.key_names = names
        self._api_key = next((env[name].strip() for name in names if env.get(name, "").strip()), "")
        base = overrides.base_url or (record.default_base_url or "")
        if base and record.base_url_suffix and urllib.parse.urlsplit(base).path in ("", "/"):
            base = base.rstrip("/") + record.base_url_suffix
        self.base_url = base.rstrip("/")
        self.base_url_display = PER_ACCOUNT if overrides.base_url else self.base_url
        self.model_override = overrides.model or ""
        self.headers_extra = dict(overrides.headers)
        timeout = record.settings_defaults.get("timeout", 120)
        self.timeout = float(timeout) if isinstance(timeout, (int, float)) else 120.0

    def redact(self, text: str) -> tuple[str, bool]:
        """Remove the credential from ``text`` if the provider echoed it.

        Args:
            text: Serialized fixture content.

        Returns:
            The text with every occurrence replaced, and whether any was found.
        """
        if not self._api_key:
            return text, False
        forms = {self._api_key, json.dumps(self._api_key)[1:-1]}
        found = any(form in text for form in forms)
        for form in forms:
            text = text.replace(form, REDACTED)
        return text, found

    def skip_reason(self) -> str | None:
        """Why this provider cannot be captured, or None when it can."""
        if self.invalid_override:
            return f"invalid override ({self.invalid_override}); value not shown"
        if not self._api_key:
            return f"no key ({' / '.join(self.key_names)})"
        if not self.base_url:
            return f"per-account URL: set {_env_name(self.record, 'BASE_URL')}"
        if self.record.key == "azure" and not self.model_override:
            return f"deployment name: set {_env_name(self.record, 'MODEL')}"
        return None

    def headers(self) -> dict[str, str]:
        """Request headers, credential included, built at call time only."""
        auth = (
            {"api-key": self._api_key}
            if self.record.auth_scheme == "api_key_header"
            else {"Authorization": f"Bearer {self._api_key}"}
        )
        return {**auth, "Content-Type": "application/json", "User-Agent": USER_AGENT, **self.headers_extra}

    def payload(self, model: str, messages: list[dict[str, Any]], *, stream: bool, tools: bool) -> dict[str, Any]:
        """A request body shaped the way the engine shapes it for this record."""
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
            self.record.max_tokens_key or "max_tokens": MAX_TOKENS,
        }
        if stream and self.record.stream_include_usage:
            body["stream_options"] = {"include_usage": True}
        if tools:
            body["tools"] = [TOOL]
            if "tool_choice" in self.record.payload_flags:
                body["tool_choice"] = "auto"
        body.update(self.record.extra_body_fields)
        return body


def _request(target: Target, url: str, body: dict[str, Any] | None) -> tuple[int, Any]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(
        url, data=data, method="POST" if body is not None else "GET", headers=target.headers()
    )
    try:
        with urllib.request.urlopen(request, timeout=target.timeout) as response:
            raw, status = response.read().decode("utf-8", errors="replace"), response.status
    except urllib.error.HTTPError as error:
        with error:
            raw, status = error.read().decode("utf-8", errors="replace"), error.code
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return -1, f"{type(error).__name__}"
    try:
        return status, json.loads(raw)
    except json.JSONDecodeError:
        return status, raw[:2000]


def _stream(target: Target, url: str, body: dict[str, Any]) -> tuple[int, list[str]]:
    request = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), method="POST", headers=target.headers()
    )
    payloads: list[str] = []
    data_lines: list[str] = []
    try:
        with urllib.request.urlopen(request, timeout=target.timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                if line.startswith("data:"):
                    data_lines.append(line[len("data:"):].lstrip(" "))
                elif line == "" and data_lines:
                    payloads.append("\n".join(data_lines))
                    data_lines = []
                    if payloads[-1] == "[DONE]" or len(payloads) >= MAX_STREAM_EVENTS:
                        break
            status = response.status
    except urllib.error.HTTPError as error:
        with error:
            return error.code, [error.read().decode("utf-8", errors="replace")[:2000]]
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return -1, [type(error).__name__]
    if data_lines:
        payloads.append("\n".join(data_lines))
    return status, payloads


def choose_model(target: Target, listing: Any) -> str | None:
    """Pick the model a capture uses.

    Args:
        target: The provider; its ``TLDW_LIVE_<KEY>_MODEL`` override wins.
        listing: The provider's ``/models`` body, or None when not listed.

    Returns:
        The override, else a cheap-looking chat model from the listing, else
        the first ``[providers]`` seed; None when there is nothing to use.
    """
    if target.model_override:
        return target.model_override
    ids = [
        entry.get("id")
        for entry in (listing.get("data") if isinstance(listing, dict) else None) or []
        if isinstance(entry, dict) and isinstance(entry.get("id"), str)
    ]
    chat_ids = [model for model in ids if not _NON_CHAT.search(model)]
    for model in chat_ids:
        if _CHEAP.search(model):
            return model
    if chat_ids:
        return chat_ids[0]
    seeds = _config_seeds(target.record)
    return seeds[0] if seeds else None


def uncovered_keys(record: ProviderRecord, fixture: dict[str, Any]) -> dict[str, list[str]]:
    """Response key NAMES outside the strict shape AND the record's allowances.

    Args:
        record: The preset whose allowances are subtracted.
        fixture: A capture with ``chat_response``/``tool_call_response`` bodies
            and ``stream_events``.

    Returns:
        Uncovered key names per level (``top``, ``choice``, ``message``), sorted.
    """
    top: set[str] = set()
    choice: set[str] = set()
    message: set[str] = set()
    for field in ("chat_response", "tool_call_response"):
        body = fixture.get(field)
        if not isinstance(body, dict) or "choices" not in body:
            continue
        top |= set(body) - KNOWN_TOP
        for item in body.get("choices") or []:
            if isinstance(item, dict):
                choice |= set(item) - KNOWN_CHOICE
                if isinstance(item.get("message"), dict):
                    message |= set(item["message"]) - KNOWN_MESSAGE
    for payload in fixture.get("stream_events") or []:
        try:
            event = json.loads(payload)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(event, dict):
            continue
        top |= set(event) - KNOWN_TOP
        for item in event.get("choices") or []:
            if isinstance(item, dict):
                choice |= set(item) - KNOWN_STREAM_CHOICE
                if isinstance(item.get("delta"), dict):
                    message |= set(item["delta"]) - KNOWN_MESSAGE
    return {
        "top": sorted(top - record.response_allowances),
        "choice": sorted(choice - record.choice_allowances),
        "message": sorted(message - record.message_allowances),
    }


def capture(target: Target) -> Path | None:
    """Run the rounds for one provider and write its raw fixture.

    Args:
        target: The provider's resolved capture settings.

    Returns:
        The fixture path, or None when nothing was written (no model, or no
        round succeeded).
    """
    record = target.record
    listing: Any = None
    if record.discovery_route is not None and not target.model_override:
        status, listing = _request(target, f"{target.base_url}/{record.discovery_route}", None)
        if status != 200:
            print(f"  ! GET /{record.discovery_route} -> HTTP {status}")
    model = choose_model(target, listing)
    if not model:
        print(f"  SKIP: no model (set {_env_name(record, 'MODEL')})")
        return None
    print(f"  model: {model}")
    url = f"{target.base_url}/chat/completions"
    chat_status, chat_body = _request(target, url, target.payload(model, PLAIN_MESSAGES, stream=False, tools=False))
    tool_status, tool_body = (None, None)
    if record.native_tools:
        tool_status, tool_body = _request(target, url, target.payload(model, TOOL_MESSAGES, stream=False, tools=True))
    stream_status, events = _stream(target, url, target.payload(model, PLAIN_MESSAGES, stream=True, tools=False))
    stream_complete = stream_status == 200 and bool(events) and events[-1] == "[DONE]"
    print(f"  plain HTTP {chat_status}  tool HTTP {tool_status}  stream HTTP {stream_status}"
          f" ({len(events)} events, complete: {stream_complete})")
    if chat_status != 200 and tool_status != 200 and not stream_complete:
        print("  FAILED: no round succeeded -- no fixture written (check the key, model and URL)")
        return None
    if stream_status == 200 and not stream_complete:
        print("  ! the stream was cut off before [DONE]; its replay will fail until recaptured")
    listed = (listing or {}).get("data") if isinstance(listing, dict) else None
    fixture = {
        "server": record.key,
        "base_url": target.base_url_display,
        "model": model,
        "statuses": {"plain": chat_status, "tool": tool_status, "stream": stream_status},
        "chat_response": chat_body,
        "tool_call_response": tool_body,
        "stream_events": events,
        "models_response": (
            None if listed is None
            else {"count": len(listed), "sample": listed[:MAX_LISTED_MODELS]}
        ),
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    text, echoed = target.redact(json.dumps(fixture, indent=2, ensure_ascii=False))
    if echoed:
        print("  ! the provider echoed the credential in a response; it was redacted")
    path = FIXTURE_DIR / f"{record.key}.json"
    path.write_text(text + "\n", encoding="utf-8")
    gaps = {level: keys for level, keys in uncovered_keys(record, json.loads(text)).items() if keys}
    print(f"  wrote {path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path}")
    print(f"  uncovered keys: {gaps or 'none'}")
    return path


PROBE_KEY = "tldw-probe-invalid-key"  # deliberately fake: proves how a bad key is answered
NOAUTH_DIR = FIXTURE_DIR / "noauth"


def _plain_request(url: str, body: dict[str, Any] | None, headers: dict[str, str], timeout: float) -> dict[str, Any]:
    """One probe request; never carries a real credential."""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(url, data=data, method="POST" if body is not None else "GET",
                                     headers={"Content-Type": "application/json", "User-Agent": USER_AGENT, **headers})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw, status = response.read().decode("utf-8", errors="replace"), response.status
    except urllib.error.HTTPError as error:
        with error:
            raw, status = error.read().decode("utf-8", errors="replace"), error.code
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return {"status": -1, "body": type(error).__name__}
    try:
        return {"status": status, "body": json.loads(raw)}
    except json.JSONDecodeError:
        return {"status": status, "body": raw[:2000]}


def probe_without_key(record: ProviderRecord) -> Path | None:
    """Probe one preset's shipped URL with no key and with a fake key.

    Records how the model listing answers (and every model id when it is
    public) and how the chat route answers a missing and a bad key. No real
    credential is ever sent, so nothing is spent.

    Args:
        record: The preset to probe.

    Returns:
        The probe fixture path, or None for a per-account preset without a URL.
    """
    base = (os.environ.get(_env_name(record, "BASE_URL"), "").strip() or record.default_base_url or "").rstrip("/")
    if not base:
        print(f"[{record.key}] skip: per-account URL (set {_env_name(record, 'BASE_URL')})")
        return None
    fake = ({"api-key": PROBE_KEY} if record.auth_scheme == "api_key_header"
            else {"Authorization": f"Bearer {PROBE_KEY}"})
    listing = _plain_request(f"{base}/{record.discovery_route or 'models'}", None, {}, 30.0)
    listed = listing["body"].get("data") if isinstance(listing["body"], dict) else None
    # A real model id, so a provider that checks the model before the key still
    # answers the key question: override, else the public listing, else a seed.
    override = os.environ.get(_env_name(record, "MODEL"), "").strip()
    listed_model = choose_model(Target(record, {}), {"data": listed}) if isinstance(listed, list) else None
    seeds = _config_seeds(record)
    model = override or listed_model or (seeds[0] if seeds else "probe-model")
    chat_body = {"model": model, "messages": [{"role": "user", "content": "hi"}],
                 record.max_tokens_key or "max_tokens": 1}
    if listing["status"] == 200 and isinstance(listed, list):
        # Keep what the discovery parser checks (object-ness and each id), not
        # the bulky pricing/metadata.
        listing = {"status": 200, "entries": [
            {"id": entry.get("id")} if isinstance(entry, dict) else entry for entry in listed]}
    elif listing["status"] == 200:
        listing = {"status": 200, "body": str(listing["body"])[:500]}
    fixture = {
        "server": record.key,
        "base_url": base if base == (record.default_base_url or "").rstrip("/") else PER_ACCOUNT,
        "probe_model": model,
        "listing": listing,
        "chat_no_key": _plain_request(f"{base}/chat/completions", chat_body, {}, 30.0),
        "chat_bad_key": _plain_request(f"{base}/chat/completions", chat_body, fake, 30.0),
        "probed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    NOAUTH_DIR.mkdir(exist_ok=True)
    path = NOAUTH_DIR / f"{record.key}.json"
    path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    ids = listing.get("entries")
    print(f"[{record.key}] listing {listing['status']}{f' ({len(ids)} models)' if ids is not None else ''}"
          f"  no key {fixture['chat_no_key']['status']}  bad key {fixture['chat_bad_key']['status']}")
    return path


def main(argv: list[str] | None = None) -> int:
    """Run the capture or probe CLI.

    Args:
        argv: Command-line arguments; ``sys.argv[1:]`` when None.

    Returns:
        The process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("providers", nargs="*", help="record keys to capture (default: every available)")
    parser.add_argument("--keys-file", type=Path, default=DEFAULT_KEYS_FILE)
    parser.add_argument("--list", action="store_true", help="show who would run, and why not")
    parser.add_argument("--no-auth", action="store_true",
                        help="probe every preset's shipped URL without a real key (no tokens spent)")
    args = parser.parse_args(argv)
    try:
        keys_file = validate_path_simple(args.keys_file, reject_shell_metacharacters=False)
    except ValueError:
        parser.error("--keys-file is not a usable path")
    env = {**read_keys_file(keys_file), **os.environ}
    records = engine_presets()
    unknown = set(args.providers) - {r.key for r in records}
    if unknown:
        parser.error(f"unknown provider(s): {', '.join(sorted(unknown))}")
    if args.no_auth:
        chosen = [r for r in records if not args.providers or r.key in args.providers]
        probed = sum(probe_without_key(record) is not None for record in chosen)
        print(f"probed {probed} of {len(chosen)} provider(s) without a key")
        return 0
    targets = [Target(r, env) for r in records if not args.providers or r.key in args.providers]
    captured = 0
    for target in targets:
        reason = target.skip_reason()
        if args.list or reason:
            print(f"[{target.record.key}] {'ready' if reason is None else 'skip: ' + reason}")
            continue
        print(f"[{target.record.key}]")
        captured += capture(target) is not None
    if not args.list:
        print(f"captured {captured} of {len(targets)} provider(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
