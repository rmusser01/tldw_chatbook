#!/usr/bin/env python3
"""Live smoke: does every shipped model id still answer a generate call?

TASK-34100.2 AC#6 (gap-03). The 2026-10-02 first-run review found all eight
shipped Google ids retired upstream: each was still *listed* by Google's
models.list, so a model list could not catch it -- only a generate call
returned the 404 ("no longer available to new users"). This script makes one
1-output-token call per shipped default model and per curated catalog id and
reports any id the provider says is gone.

It reads the shipped template straight from
``tldw_chatbook/Config_Files/default_config.toml`` (nothing from the app is
imported, so no profile, config file or keychain is touched) and the API keys
from the environment:

* Google: ``GEMINI_API_KEY`` or ``GOOGLE_API_KEY``
* OpenAI: ``OPENAI_API_KEY``
* Anthropic: ``ANTHROPIC_API_KEY``

A provider without a key is skipped and says so. Keys are never printed.

Outcomes per id: ``ok`` (answered), ``retired`` (404: the provider no longer
serves it -- this fails the run), ``quota`` (429: the id exists, this key may
not use it now), ``unavailable`` (5xx: a transient provider outage such as
Google's 503 "high demand" -- retry later), ``rejected`` (another 4xx, with the
provider's message), ``key`` (401/403: the key, not the id -- the
provider's remaining ids are skipped) and ``unreachable`` (no HTTP answer at
all: a network error or a timeout, with the error type -- never counted as
retired; retry later).

A provider's shipped default is held to more: it is what a first chat sends
to, so a default that gives no answer (``unavailable``, ``unreachable``) is
tried ``DEFAULT_ATTEMPTS`` times and, if it never answers, fails the run.
TASK-34100.2 review v5r2 (G2C-R3-1): Google's default ``gemini-flash-latest``
timed out or answered 503 on every run for five weeks while this script
exited 0. A catalog id that gives no answer is named in the summary and does
not fail the run.

Usage::

    python Helper_Scripts/smoke_shipped_model_ids.py [--providers google,openai]

Exit status: 0 when no shipped id is retired and every default answered, 1
when an id is retired or a default never answered, 2 on a usage or
template-parse error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

#: The shipped default config template: a packaged data file since
#: TASK-36945.1 (it used to be config.py's ``CONFIG_TOML_CONTENT`` literal).
_TEMPLATE_TOML = (
    Path(__file__).resolve().parents[1]
    / "tldw_chatbook"
    / "Config_Files"
    / "default_config.toml"
)
_TIMEOUT_SECONDS = 30
#: Tries for a provider's shipped default before its silence fails the run.
DEFAULT_ATTEMPTS = 3
#: Verdicts that are no answer at all (an id that may still be served).
_SILENT = frozenset({"unavailable", "unreachable"})
#: Template section name -> ([providers] list key, env vars holding a key).
_PROVIDERS = {
    "google": ("Google", ("GEMINI_API_KEY", "GOOGLE_API_KEY")),
    "openai": ("OpenAI", ("OPENAI_API_KEY",)),
    "anthropic": ("Anthropic", ("ANTHROPIC_API_KEY",)),
}


@dataclass(frozen=True)
class Outcome:
    """One id's verdict and the provider's own (bounded) words for it."""

    model: str
    verdict: str
    detail: str = ""


def shipped_template() -> dict:
    """Parse the shipped template (``Config_Files/default_config.toml``) without importing config.py."""
    text = _TEMPLATE_TOML.read_bytes().decode("utf-8")
    # Placeholders config.py substitutes at import time.
    text = text.replace("__DEFAULT_TRANSCRIPTION_PROVIDER__", "auto")
    text = text.replace("__DEFAULT_SPLASH_DURATION__", "1")
    return tomllib.loads(text)


def shipped_ids(template: dict, section: str) -> list[str]:
    """The provider's default model first, then its curated catalog ids."""
    list_key, _env = _PROVIDERS[section]
    default = template.get("api_settings", {}).get(section, {}).get("model")
    ids = [default] if isinstance(default, str) and default else []
    ids += [str(model) for model in template.get("providers", {}).get(list_key, [])]
    return list(dict.fromkeys(model for model in ids if model.strip()))


def _request(section: str, model: str, key: str) -> urllib.request.Request:
    if section == "google":
        return urllib.request.Request(
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent",
            data=json.dumps(
                {
                    "contents": [{"role": "user", "parts": [{"text": "Say OK"}]}],
                    "generationConfig": {"maxOutputTokens": 1},
                }
            ).encode(),
            headers={"x-goog-api-key": key, "Content-Type": "application/json"},
        )
    if section == "anthropic":
        return urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(
                {
                    "model": model,
                    "max_tokens": 1,
                    "messages": [{"role": "user", "content": "Say OK"}],
                }
            ).encode(),
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
        )
    return urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(
            {
                "model": model,
                "max_completion_tokens": 1,
                "messages": [{"role": "user", "content": "Say OK"}],
            }
        ).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )


def _provider_message(body: bytes) -> str:
    """The provider's own error message, bounded to one short line."""
    try:
        payload = json.loads(body or b"{}")
    except ValueError:
        return ""
    error = payload.get("error") if isinstance(payload, dict) else None
    message = error.get("message") if isinstance(error, dict) else ""
    return " ".join(str(message or "").split())[:160]


def probe(section: str, model: str, key: str) -> Outcome:
    """Make one 1-output-token call and classify the answer."""
    try:
        with urllib.request.urlopen(  # noqa: S310 -- fixed https hosts above
            _request(section, model, key), timeout=_TIMEOUT_SECONDS
        ):
            return Outcome(model, "ok")
    except urllib.error.HTTPError as error:
        detail = _provider_message(error.read())
        verdict = {404: "retired", 429: "quota", 401: "key", 403: "key"}.get(
            error.code, "unavailable" if error.code >= 500 else "rejected"
        )
        if verdict == "rejected" and "max_tokens" in detail.lower():
            # A 1-token cap some reasoning models refuse: the id is served.
            verdict = "ok"
        return Outcome(model, verdict, f"HTTP {error.code} {detail}".strip())
    except (urllib.error.URLError, TimeoutError) as error:
        return Outcome(model, "unreachable", type(error).__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--providers",
        default=",".join(_PROVIDERS),
        help="comma-separated subset of: " + ", ".join(_PROVIDERS),
    )
    args = parser.parse_args(argv)
    sections = [part.strip() for part in args.providers.split(",") if part.strip()]
    unknown = [section for section in sections if section not in _PROVIDERS]
    if unknown:
        print(f"unknown provider(s): {', '.join(unknown)}", file=sys.stderr)
        return 2
    try:
        template = shipped_template()
    except (OSError, ValueError, tomllib.TOMLDecodeError) as error:
        print(f"could not read the shipped template: {error}", file=sys.stderr)
        return 2

    retired: list[str] = []
    silent_defaults: list[str] = []
    silent: list[str] = []
    for section in sections:
        _list_key, env_vars = _PROVIDERS[section]
        key = next((os.environ[name].strip() for name in env_vars if os.environ.get(name, "").strip()), "")
        ids = shipped_ids(template, section)
        if not key:
            print(f"{section}: skipped ({' or '.join(env_vars)} not set; {len(ids)} ids)")
            continue
        print(f"{section}: {len(ids)} shipped ids")
        default = template.get("api_settings", {}).get(section, {}).get("model")
        for model in ids:
            attempts = DEFAULT_ATTEMPTS if model == default else 1
            for _attempt in range(attempts):
                outcome = probe(section, model, key)
                if outcome.verdict not in _SILENT:
                    break
            print(f"  {outcome.verdict:<11} {model}  {outcome.detail}".rstrip())
            if outcome.verdict == "retired":
                retired.append(f"{section}/{model}")
            elif outcome.verdict in _SILENT:
                (silent_defaults if model == default else silent).append(
                    f"{section}/{model}"
                )
            if outcome.verdict == "key":
                print(f"  {section}: the key was refused; remaining ids not checked")
                break
    if silent:
        print("Did not answer (retry later): " + ", ".join(silent))
    if silent_defaults:
        print(
            f"Shipped default did not answer ({DEFAULT_ATTEMPTS} tries): "
            + ", ".join(silent_defaults)
        )
    if retired:
        print("RETIRED shipped ids: " + ", ".join(retired))
        return 1
    print("No shipped id is retired.")
    return 1 if silent_defaults else 0


if __name__ == "__main__":
    sys.exit(main())
