# Network TLS Trust Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a global `[network] ssl_verify` ternary (verify / off / custom-CA-additive) that governs TLS trust for all in-scope outbound HTTP/HTTPS/WebSocket traffic, with an F9 settings surface and fail-safe-to-verification-on semantics.

**Architecture:** One new helper module `tldw_chatbook/Utils/tls_trust.py` owns the policy (normalization, additive SSL contexts, merged-bundle cache for requests, client factories, warn-once/metrics). Shared client seams adopt the factories; ~50 long-tail client constructions get one-line policy threading; `tldw_api/client.py` gains a standalone-safe `ssl_verify` constructor param; the F9 settings screen gains a Network category.

**Tech Stack:** Python ≥3.11 (stdlib `ssl`, `tomllib` for tests), certifi, httpx 0.28.1, requests, aiohttp, websockets ≥14, Textual (settings screen), pytest (asyncio auto mode).

**Spec:** `Docs/superpowers/specs/2026-08-29-network-tls-trust-policy-design.md` — the plan argues from the spec; executors read both.

## Global Constraints

- **No new runtime dependencies.** certifi is already a transitive dep of httpx/requests; everything else stdlib.
- **Fail-safe direction is ALWAYS verification-on.** Any invalid value (bad type, missing/unreadable file, corrupt PEM, bundle-write failure) logs an error and yields default verification. The setting can never silently disable verification.
- **Custom-CA semantics are ADDITIVE** (certifi + custom both trusted). httpx callers must never receive a bare custom-CA **path** as `verify=` (httpx would load it as the *only* bundle) — pass `bool` or the additive `SSLContext` via `httpx_verify()`.
- **Subscriptions feeds keep their existing per-feed `ssl_verify` DB flag** — no changes to `Subscriptions/` in this plan.
- **Out of scope:** MCP client, Playwright, `Model_Artifacts/` HF downloads, Notes git push, `Web_Server` outbound, `truststore`/OS-trust-store mode, env-var override.
- **No new screen keybindings** (per ADR-031); the Network category reuses the existing `s` save action. Footer hints must not advertise unimplemented actions.
- **Tests are targeted, never full-suite** unless the user asks: run only the files listed per task. Commands use `.venv/bin/python -m pytest` from repo root. `asyncio_mode = "auto"` (no flags needed); markers must exist in `pyproject.toml` (`unit`, `ui` already registered).
- Commits: conventional-commit style (`feat:`, `test:`, `docs:`), one commit per task step as shown.

---

### Task 1: ADR-079, backlog task, and the `[network]` config template

**Files:**
- Create: `backlog/decisions/079-network-tls-trust-policy.md`
- Modify: `tldw_chatbook/config.py` (inside `CONFIG_TOML_CONTENT`, the template string starting at line 2921; insert after the `[web_security]` block that ends at line 3737, before `[image_generation]` at line 3739)
- Create: `Tests/Utils/test_tls_trust.py`

**Interfaces:**
- Consumes: nothing (this is the foundation task).
- Produces: config key `network.ssl_verify` (default `true`) in the default template; ADR `backlog/decisions/079-network-tls-trust-policy.md`; backlog task (record its printed ID as `<TASK_ID>` for later tasks).

- [ ] **Step 1: Create ADR-079** with exactly this content:

```markdown
# ADR-079: Network TLS Trust Policy (corporate DPI networks)

**Status:** Accepted
**Date:** 2026-08-29
**Spec:** Docs/superpowers/specs/2026-08-29-network-tls-trust-policy-design.md

## Context

Corporate TLS-inspection/DPI networks re-sign outbound HTTPS with a corporate
root CA that lives in the OS trust store. None of this app's transports
(requests, httpx, aiohttp, websockets) consult the OS store — they verify
against certifi/OpenSSL default paths only — so every HTTPS call fails with
`SSLCertVerificationError` in intercepted networks, and no setting exists to
express trust. ~50 inline client constructions across four transports plus one
OpenAI-SDK site; no shared seam today.

## Decision

One global config knob, `[network] ssl_verify = true | false | "/path/ca.pem"`:

- `true` (default): verify against the default bundle.
- `false`: verification disabled — insecure escape hatch, warned loudly.
- path: **additive** trust — the custom CA is trusted *in addition to* certifi,
  never as a replacement (selective interception is the common corp topology).

Implementation: shared helper `tldw_chatbook/Utils/tls_trust.py` (normalization,
additive SSL contexts, merged-PEM cache for requests, client factories,
warn-once + metrics). Shared seams (Console gateway, tldw_api client, image
gen, TTS, model catalog, evals) adopt factories; the long tail threads
`session.verify` / `ssl=` / injected `http_client=`. `tldw_api/client.py` stays
standalone (Apache-2.0): it gains an `ssl_verify` constructor param and the app
passes the resolved policy in. F9 Settings gains a Network category.

**Fail-safe direction is always verification-on**: invalid value, missing/
unreadable file, corrupt PEM, or bundle-write failure → default verification,
with an error log stating the remedy.

## Considered and rejected

- **Global startup injection** (env vars / `ssl.SSLContext` monkeypatching):
  requests respects `REQUESTS_CA_BUNDLE`, httpx uses certifi regardless, aiohttp
  and websockets ignore both; no stack supports disabling verification via env.
  Monkeypatching silently alters TLS for out-of-scope libraries.
- **Replace-semantics custom CA**: breaks every non-intercepted public endpoint
  under selective interception. (A team wanting "corp CA only" can export a
  bundle containing only the corp root — same knob.)
- **Per-provider granularity / insecure-host allowlist**: interception is
  network-wide; per-host verification is unsupported natively by any of the
  four stacks (would need mounted adapters / event hooks in each).
- **`truststore` (OS trust store) now**: new dependency; deferred as an
  additive follow-up mode if users still struggle to obtain the corp CA.
- **Unifying Subscriptions feeds into the global knob**: the existing per-feed
  `ssl_verify` flag keeps working; revisit later.

## Consequences

- With `false`, API keys and conversation content are interceptable by anyone
  on the network path — surfaced as a settings warning and a once-per-process
  log + metric.
- Policy binds at client construction; loop-cached clients (Console gateway)
  pick up changes after restart. Per-call sessions pick changes up immediately.
- New outbound-HTTP code should use `build_httpx_async_client` /
  `build_httpx_client` / `build_requests_session` so the policy applies by
  construction.
```

- [ ] **Step 2: File the backlog task** (record the printed task ID — later tasks reference it as `<TASK_ID>`):

```bash
backlog task create "Network TLS trust policy (corp DPI)" \
  -d "Corporate TLS-inspection networks break every HTTPS call because no transport consults the OS trust store and no setting expresses trust. Add one global ternary [network] ssl_verify with additive custom-CA semantics, plumbed through a shared helper and every in-scope outbound client." \
  --ac "Config [network] ssl_verify ternary normalizes leniently and fails safe to verification-on" \
  --ac "tls_trust helper tests cover coercion, additive contexts, merged bundle, and factories" \
  --ac "Shared httpx seams, requests long tail, aiohttp, websockets, and the OpenAI-SDK site honor the policy" \
  --ac "tldw_api client exposes an ssl_verify constructor param and the app bootstrap passes the policy" \
  --ac "F9 Network category saves valid values, rejects bad paths, and warns when verification is relaxed" \
  --ac "ADR-079 records the decision and rejected alternatives"
```

- [ ] **Step 3: Add the `[network]` section to the default config template.** In `tldw_chatbook/config.py`, find the `[web_security]` block (ends with `allowed_hosts = []` at line 3737) and insert immediately after it, before `[image_generation]`:

```toml
[network]
# TLS trust for outbound HTTP/HTTPS/WebSocket (LLM providers, content fetching).
#   true                verify against the default bundle (default)
#   false               DISABLE verification (insecure; only for TLS-inspecting
#                       corporate networks where you cannot obtain the CA)
#   "/path/to/ca.pem"   ALSO trust this CA bundle (corporate root CA) — additive.
# Windows paths: use a literal single-quoted string ('C:\certs\corp.pem').
ssl_verify = true
```

- [ ] **Step 4: Write the template test** — create `Tests/Utils/test_tls_trust.py`:

```python
"""Tests for the app-wide TLS trust policy (Utils/tls_trust.py) + config template."""
import tomllib

import tldw_chatbook.config as config_module


def test_default_config_template_has_network_ssl_verify():
    parsed = tomllib.loads(config_module.CONFIG_TOML_CONTENT)
    assert parsed["network"]["ssl_verify"] is True
```

- [ ] **Step 5: Run it and verify it passes** (it must — the template is data, not logic; if it fails the section was misplaced):

Run: `.venv/bin/python -m pytest Tests/Utils/test_tls_trust.py -v`
Expected: `1 passed`

- [ ] **Step 6: Commit**

```bash
git add backlog/decisions/079-network-tls-trust-policy.md tldw_chatbook/config.py Tests/Utils/test_tls_trust.py
git commit -m "docs: ADR-079 + [network] ssl_verify config template for TLS trust policy"
```

---

### Task 2: `Utils/tls_trust.py` — setting normalization, fail-safe, warn-once

**Files:**
- Create: `tldw_chatbook/Utils/tls_trust.py`
- Test: `Tests/Utils/test_tls_trust.py` (append)

**Interfaces:**
- Consumes: `get_cli_setting` from `tldw_chatbook.config`; `log_counter` from `tldw_chatbook.Metrics.metrics_logger`.
- Produces: `tls_verify_setting() -> bool | str` (normalized; `str` only when the path exists) and `warn_tls_policy() -> None` (idempotent per mode). Later tasks rely on exactly these names/signatures.

- [ ] **Step 1: Write the failing tests** — append to `Tests/Utils/test_tls_trust.py`:

```python
import ssl as _ssl
from pathlib import Path

import pytest
from loguru import logger

import tldw_chatbook.Utils.tls_trust as tls_trust


@pytest.fixture(autouse=True)
def _clean_warn_state():
    tls_trust._warned_modes.clear()
    yield
    tls_trust._warned_modes.clear()


@pytest.fixture
def _set_ssl_config(monkeypatch):
    def _install(value):
        monkeypatch.setattr(
            tls_trust,
            "get_cli_setting",
            lambda section, key=None, default=None: (
                value if (section, key) == ("network", "ssl_verify") else default
            ),
        )

    return _install


@pytest.mark.parametrize(
    "raw,expected",
    [
        (True, True),
        (False, False),
        ("true", True),
        ("1", True),
        ("ON", True),
        ("false", False),
        ("0", False),
        ("no", False),
        ("OFF", False),
        ("", True),
        ("   ", True),
        (5, True),          # unsupported type -> fail safe
        (None, True),       # unsupported type -> fail safe
        (["x"], True),      # unsupported type -> fail safe
    ],
)
def test_tls_verify_setting_coercion(_set_ssl_config, raw, expected):
    _set_ssl_config(raw)
    assert tls_trust.tls_verify_setting() is expected


def test_tls_verify_setting_existing_path_string(tmp_path, _set_ssl_config):
    ca = tmp_path / "corp.pem"
    ca.write_text("# corp")
    _set_ssl_config(str(ca))
    assert tls_trust.tls_verify_setting() == str(ca)


def test_tls_verify_setting_missing_path_fails_safe(tmp_path, _set_ssl_config):
    _set_ssl_config(str(tmp_path / "missing.pem"))
    assert tls_trust.tls_verify_setting() is True


def test_tls_verify_setting_missing_path_logs_error(tmp_path, _set_ssl_config, capsys):
    _set_ssl_config(str(tmp_path / "missing.pem"))
    messages: list[str] = []
    sink_id = logger.add(messages.append, level="ERROR")
    try:
        tls_trust.tls_verify_setting()
    finally:
        logger.remove(sink_id)
    assert any("ssl_verify" in m and "existing file" in m for m in messages)


def test_warn_tls_policy_once_per_mode(_set_ssl_config):
    _set_ssl_config(False)
    messages: list[str] = []
    sink_id = logger.add(messages.append, level="WARNING")
    try:
        tls_trust.warn_tls_policy()
        tls_trust.warn_tls_policy()
    finally:
        logger.remove(sink_id)
    warnings = [m for m in messages if "DISABLED" in m]
    assert len(warnings) == 1
    assert "API keys" in warnings[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest Tests/Utils/test_tls_trust.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tldw_chatbook.Utils.tls_trust'` (or collection error of the same kind).

- [ ] **Step 3: Create `tldw_chatbook/Utils/tls_trust.py`** with:

```python
"""App-wide TLS trust policy for outbound HTTP/HTTPS/WebSocket clients.

One config knob, ``[network] ssl_verify``:

- ``true`` (default) -> verify against the default bundle (certifi).
- ``false``          -> verification DISABLED (insecure escape hatch for
                        TLS-inspecting corporate networks).
- ``"/path/ca.pem"`` -> ALSO trust this CA bundle (corporate root CA) —
                        ADDITIVE: certifi + custom, never replace.

Fail-safe direction is ALWAYS verification-on: any invalid value (bad type,
missing/unreadable file, unparseable PEM, bundle-write failure) logs an error
with the remedy and yields default verification.

Governance: backlog/decisions/079-network-tls-trust-policy.md and
Docs/superpowers/specs/2026-08-29-network-tls-trust-policy-design.md.
"""
from __future__ import annotations

from pathlib import Path

from loguru import logger

from ..Metrics.metrics_logger import log_counter
from ..config import get_cli_setting

_TRUE_STRINGS = frozenset({"true", "1", "on"})
_FALSE_STRINGS = frozenset({"false", "0", "no", "off"})

_warned_modes: set[str] = set()


def tls_verify_setting() -> bool | str:
    """Normalized ``[network] ssl_verify``.

    Returns:
        ``True`` (verify on), ``False`` (verification off), or the string
        path of an EXISTING CA-bundle file. Never raises.
    """
    value = get_cli_setting("network", "ssl_verify", True)
    if isinstance(value, bool):
        result: bool | str = value
    elif isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE_STRINGS or not lowered:
            result = True
        elif lowered in _FALSE_STRINGS:
            result = False
        else:
            path = Path(value.strip()).expanduser()
            if path.is_file():
                result = str(path)
            else:
                logger.error(
                    f"[network] ssl_verify path {str(path)!r} is not an existing"
                    " file; falling back to default certificate verification."
                    " Remedy: point ssl_verify at an existing CA bundle (PEM)"
                    " file."
                )
                result = True
    else:
        logger.error(
            f"[network] ssl_verify has unsupported type"
            f" {type(value).__name__}; falling back to default certificate"
            " verification."
        )
        result = True
    _maybe_warn(result)
    return result


def warn_tls_policy() -> None:
    """Warn (once per process per mode) + metric when verification is relaxed."""
    _maybe_warn(tls_verify_setting())


def _maybe_warn(setting: bool | str) -> None:
    if setting is True:
        return
    if setting is False:
        mode, message = "off", (
            "TLS certificate verification is DISABLED"
            " ([network] ssl_verify = false). API keys and conversation"
            " content can be intercepted by anyone on the network path."
            " Restore ssl_verify = true unless this is required by a"
            " TLS-inspecting corporate network."
        )
    else:
        mode, message = "custom_ca", (
            f"TLS verification additionally trusts custom CA bundle"
            f" {setting!r} ([network] ssl_verify). Ensure this is your"
            " organisation's root CA."
        )
    log_counter(f"network_tls_verify_{mode}")
    if mode in _warned_modes:
        return
    _warned_modes.add(mode)
    logger.warning(message)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest Tests/Utils/test_tls_trust.py -v`
Expected: `14 passed` (13 coercion params + template test + the rest — all green; any FAIL means the normalization table diverges from the test matrix).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Utils/tls_trust.py Tests/Utils/test_tls_trust.py
git commit -m "feat: tls_trust setting normalization with fail-safe verify-on and warn-once"
```

---

### Task 3: Additive trust — `ssl_context_for_transport()` and the requests merged bundle

**Files:**
- Modify: `tldw_chatbook/Utils/tls_trust.py` (append)
- Test: `Tests/Utils/test_tls_trust.py` (append)

**Interfaces:**
- Consumes: `tls_verify_setting()` (Task 2), `get_user_data_dir` from `tldw_chatbook.Utils.paths`, `certifi.where()`.
- Produces:
  - `ssl_context_for_transport() -> None | bool | ssl.SSLContext` (`None` default / `False` off / additive context) — consumed by aiohttp and websockets call sites (Tasks 5, 8).
  - `requests_verify() -> bool | str` — consumed by every requests call site (Tasks 5, 7) and `build_requests_session` (Task 4).

- [ ] **Step 1: Write the failing tests** — append:

```python
import certifi


def _context_certs(ctx: "_ssl.SSLContext") -> set[bytes]:
    return {bytes(der) for der in ctx.get_ca_certs()}


_CUSTOM_PEM = (
    "-----BEGIN CERTIFICATE-----\n"
    "MIICIDCCAcYCCQDceGLIPeXd0zAKBggqhkjOPQQDAjAeMRwwGgYDVQQDDBN0bHMt\n"
    "dHJ1c3QtcGxhbi10ZXN0MB4XDTI2MDgyOTIyMTQ1M1oXDTM2MDgyNjIyMTQ1M1ow\n"
    "HjEcMBoGA1UEAwwTdGxzLXRydXN0LXBsYW4tdGVzdDCCAUswggEDBgcqhkjOPQIB\n"
    "MIH3AgEBMCwGByqGSM49AQECIQD/////AAAAAQAAAAAAAAAAAAAAAP//////////\n"
    "/////zBbBCD/////AAAAAQAAAAAAAAAAAAAAAP///////////////AQgWsY12Ko6\n"
    "k+ez671VdpiGvGUdBrDMU7D2O848PifSYEsDFQDEnTYIhucEk2pmeOETnSa3gZ9+\n"
    "kARBBGsX0fLhLEJH+Lzm5WOkQPJ3A32BLeszoPShOUXYmMKWT+NC4v4af5uO5+tK\n"
    "fA+eFivOM1drMV7Oy7ZAaDe/UfUCIQD/////AAAAAP//////////vOb6racXnoTz\n"
    "ucrC/GMlUQIBAQNCAARJY3gkP7zefsi/pnJW3KSsqc5nUiDQaLk/pB+yUHyazyqn\n"
    "S8AbLvsD1yhRO0B1rWN4VE4ghed8tZcclprS9j38MAoGCCqGSM49BAMCA0gAMEUC\n"
    "ICWp+dTRy9tkb1JSpx3yInFXId3QEjaL3DBQ9yI+/RFAAiEA+PfkQVSpmC0qJ80f\n"
    "SU8n1MnQXxWjOLJNSSPjSCbZBe4=\n"
    "-----END CERTIFICATE-----\n"
)
# A real (throwaway-key) self-signed certificate generated during planning:
# loading requires a parseable PEM body — a fake base64 blob would raise, and
# a file with no PEM block at all also raises SSLError ("no certificate or
# crl found"), which the helper's (OSError, ssl.SSLError) catch converts to
# the fail-safe verify-on path.


def test_ssl_context_default_returns_none(_set_ssl_config):
    _set_ssl_config(True)
    assert tls_trust.ssl_context_for_transport() is None


def test_ssl_context_off_returns_false(_set_ssl_config):
    _set_ssl_config(False)
    assert tls_trust.ssl_context_for_transport() is False


def test_ssl_context_additive_contains_certifi_plus_custom(
    tmp_path, _set_ssl_config
):
    ca = tmp_path / "corp.pem"
    ca.write_text(_CUSTOM_PEM)
    _set_ssl_config(str(ca))
    ctx = tls_trust.ssl_context_for_transport()
    assert isinstance(ctx, _ssl.SSLContext)
    certifi_only = _context_certs(
        _ssl.create_default_context(cafile=certifi.where())
    )
    merged = _context_certs(ctx)
    assert certifi_only < merged  # strictly more certs than certifi alone


def test_ssl_context_corrupt_pem_fails_safe(tmp_path, _set_ssl_config):
    ca = tmp_path / "corp.pem"
    ca.write_text(
        "-----BEGIN CERTIFICATE-----\ngarbage body\n-----END CERTIFICATE-----\n"
    )
    _set_ssl_config(str(ca))
    assert tls_trust.ssl_context_for_transport() is None


def test_requests_verify_bool_passthrough(_set_ssl_config):
    _set_ssl_config(False)
    assert tls_trust.requests_verify() is False
    _set_ssl_config(True)
    assert tls_trust.requests_verify() is True


def test_requests_verify_custom_ca_yields_merged_bundle(
    tmp_path, monkeypatch, _set_ssl_config
):
    ca = tmp_path / "corp.pem"
    ca.write_text(_CUSTOM_PEM)
    _set_ssl_config(str(ca))
    monkeypatch.setattr(
        tls_trust, "get_user_data_dir", lambda: tmp_path / "user_data"
    )
    merged_path = tls_trust.requests_verify()
    assert isinstance(merged_path, str)
    body = Path(merged_path).read_text()
    assert "BEGIN CERTIFICATE" in body
    # merged bundle loads cleanly as a CA store (comment header tolerated)
    ctx = _ssl.create_default_context(cafile=merged_path)
    assert _context_certs(ctx)


def test_merged_bundle_regenerates_when_custom_changes(
    tmp_path, monkeypatch, _set_ssl_config
):
    data_dir = tmp_path / "user_data"
    ca = tmp_path / "corp.pem"
    ca.write_text(_CUSTOM_PEM)
    _set_ssl_config(str(ca))
    monkeypatch.setattr(tls_trust, "get_user_data_dir", lambda: data_dir)
    first = tls_trust.requests_verify()
    ca.write_text(_CUSTOM_PEM + _CUSTOM_PEM)  # content (and mtime) change
    second = tls_trust.requests_verify()
    assert Path(second).read_text() != Path(first).read_text()


def test_merged_bundle_reused_when_sources_unchanged(
    tmp_path, monkeypatch, _set_ssl_config
):
    ca = tmp_path / "corp.pem"
    ca.write_text(_CUSTOM_PEM)
    _set_ssl_config(str(ca))
    monkeypatch.setattr(
        tls_trust, "get_user_data_dir", lambda: tmp_path / "user_data"
    )
    first = Path(tls_trust.requests_verify())
    first_mtime = first.stat().st_mtime_ns
    second = Path(tls_trust.requests_verify())
    assert second == first
    assert second.stat().st_mtime_ns == first_mtime  # not rewritten
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest Tests/Utils/test_tls_trust.py -v -k "ssl_context or requests_verify or merged_bundle"`
Expected: FAIL with `AttributeError: ... has no attribute 'ssl_context_for_transport'`.

- [ ] **Step 3: Append to `tldw_chatbook/Utils/tls_trust.py`** (add `os`, `ssl`, `tempfile`, `Any` to the imports at the top of the file: `import os`, `import ssl`, `import tempfile`, and `from typing import Any`):

```python
_MERGED_BUNDLE_NAME = "merged-ca-bundle.pem"


def _additive_context(custom_ca: str) -> ssl.SSLContext:
    """Context trusting certifi's bundle PLUS ``custom_ca`` (never replace)."""
    import certifi

    context = ssl.create_default_context(cafile=certifi.where())
    context.load_verify_locations(cafile=custom_ca)
    return context


def ssl_context_for_transport() -> None | bool | ssl.SSLContext:
    """Trust value for aiohttp ``TCPConnector(ssl=...)`` / websockets ``connect(ssl=...)``.

    Returns:
        ``None`` for default verification, ``False`` when verification is
        disabled, or an ADDITIVE ``ssl.SSLContext`` for a custom CA. Never
        raises; load failures fail safe to ``None``.
    """
    setting = tls_verify_setting()
    if setting is True:
        return None
    if setting is False:
        return False
    try:
        return _additive_context(setting)
    except (OSError, ssl.SSLError) as exc:
        logger.error(
            f"[network] ssl_verify bundle {setting!r} could not be loaded"
            f" ({exc}); falling back to default certificate verification."
        )
        return None


def _merged_bundle_path() -> str:
    """Path to a cached PEM containing certifi + the custom CA.

    Regenerated (atomic tmp + ``os.replace``) whenever either source's
    ``(mtime_ns, size)`` changes — a comment header records the fingerprint,
    and OpenSSL's PEM reader ignores non-PEM lines.
    """
    import certifi

    setting = tls_verify_setting()
    assert isinstance(setting, str)
    cache_dir = Path(get_user_data_dir()) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    merged = cache_dir / _MERGED_BUNDLE_NAME
    sources = (Path(certifi.where()), Path(setting))
    fingerprint = ";".join(
        f"{p}|{p.stat().st_mtime_ns}|{p.stat().st_size}" for p in sources
    )
    header = f"# tls-trust-sources: {fingerprint}\n"
    if merged.is_file() and merged.read_text(errors="replace").startswith(header):
        return str(merged)
    body = header + "".join(p.read_text() + "\n" for p in sources)
    fd, tmp = tempfile.mkstemp(dir=cache_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(body)
        os.replace(tmp, merged)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return str(merged)


def requests_verify() -> bool | str:
    """``verify=`` value for requests sessions/requests (bool or merged-bundle path)."""
    setting = tls_verify_setting()
    if setting is True or setting is False:
        return setting
    try:
        return _merged_bundle_path()
    except (OSError, UnicodeDecodeError) as exc:
        logger.error(
            f"[network] merged CA bundle could not be written ({exc});"
            " falling back to default certificate verification."
        )
        return True
```

Also change the import line `from ..config import get_cli_setting` to add the paths re-export:
`from ..Utils.paths import get_user_data_dir` is WRONG (that's this package) — use `from .paths import get_user_data_dir` (sibling module inside `Utils`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest Tests/Utils/test_tls_trust.py -v`
Expected: all pass (`22 passed` total in this file so far).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Utils/tls_trust.py Tests/Utils/test_tls_trust.py
git commit -m "feat: additive TLS trust contexts + requests merged-bundle cache"
```

---

### Task 4: Client factories + `httpx_verify()`

**Files:**
- Modify: `tldw_chatbook/Utils/tls_trust.py` (append)
- Test: `Tests/Utils/test_tls_trust.py` (append)

**Interfaces:**
- Consumes: `tls_verify_setting()`, `ssl_context_for_transport()`, `requests_verify()` (Tasks 2-3).
- Produces (all later seam tasks call these):
  - `httpx_verify() -> bool | ssl.SSLContext` — never a bare path string.
  - `build_httpx_async_client(**kwargs) -> httpx.AsyncClient`
  - `build_httpx_client(**kwargs) -> httpx.Client`
  - `build_requests_session(**kwargs) -> requests.Session`

- [ ] **Step 1: Write the failing tests** — append:

```python
import httpx
import requests as _requests


def _ssl_context_of(client) -> "_ssl.SSLContext":
    return client._transport._pool._ssl_context  # httpx 0.28 / httpcore layout


def test_httpx_verify_never_returns_bare_path(tmp_path, _set_ssl_config):
    ca = tmp_path / "corp.pem"
    ca.write_text(_CUSTOM_PEM)
    _set_ssl_config(str(ca))
    value = tls_trust.httpx_verify()
    assert isinstance(value, _ssl.SSLContext)  # never the bare path


def test_build_httpx_client_injects_policy(_set_ssl_config):
    _set_ssl_config(False)
    client = tls_trust.build_httpx_client()
    try:
        assert _ssl_context_of(client).verify_mode == _ssl.CERT_NONE
    finally:
        client.close()


def test_build_httpx_client_explicit_verify_wins(_set_ssl_config):
    _set_ssl_config(False)
    client = tls_trust.build_httpx_client(verify=True)
    try:
        assert _ssl_context_of(client).verify_mode != _ssl.CERT_NONE
    finally:
        client.close()


def test_build_httpx_client_default_is_verification(_set_ssl_config):
    _set_ssl_config(True)
    client = tls_trust.build_httpx_client()
    try:
        assert _ssl_context_of(client).verify_mode == _ssl.CERT_REQUIRED
    finally:
        client.close()


async def test_build_httpx_async_client_injects_policy(_set_ssl_config):
    _set_ssl_config(False)
    client = tls_trust.build_httpx_async_client()
    try:
        assert _ssl_context_of(client).verify_mode == _ssl.CERT_NONE
    finally:
        await client.aclose()


def test_build_requests_session_injects_policy(_set_ssl_config):
    _set_ssl_config(False)
    session = tls_trust.build_requests_session()
    assert session.verify is False


def test_build_requests_session_explicit_kwargs_forwarded(_set_ssl_config):
    _set_ssl_config(True)
    session = tls_trust.build_requests_session()
    assert session.verify is True  # bool passthrough, no merged file written
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest Tests/Utils/test_tls_trust.py -v -k "httpx or build_"`
Expected: FAIL with `AttributeError: ... has no attribute 'httpx_verify'`.

- [ ] **Step 3: Append to `tldw_chatbook/Utils/tls_trust.py`** (add imports `import httpx` and `import requests as _requests` at the top):

```python
def httpx_verify() -> bool | ssl.SSLContext:
    """``verify=`` value for httpx clients.

    ``bool`` or the additive ``SSLContext`` — NEVER a bare custom-CA path,
    which httpx would load as the only trusted bundle (replace semantics).
    """
    setting = tls_verify_setting()
    if setting is True or setting is False:
        return setting
    context = ssl_context_for_transport()
    return context if isinstance(context, ssl.SSLContext) else True


def build_httpx_async_client(**kwargs: Any) -> httpx.AsyncClient:
    """``httpx.AsyncClient`` with the app TLS trust policy applied by default.

    Callers may override with an explicit ``verify=`` (it wins).
    """
    kwargs.setdefault("verify", httpx_verify())
    return httpx.AsyncClient(**kwargs)


def build_httpx_client(**kwargs: Any) -> httpx.Client:
    """``httpx.Client`` with the app TLS trust policy applied by default."""
    kwargs.setdefault("verify", httpx_verify())
    return httpx.Client(**kwargs)


def build_requests_session(**kwargs: Any) -> _requests.Session:
    """``requests.Session`` with the app TLS trust policy applied by default."""
    session = _requests.Session(**kwargs)
    session.verify = requests_verify()
    return session
```

- [ ] **Step 4: Run the full helper test file**

Run: `.venv/bin/python -m pytest Tests/Utils/test_tls_trust.py -v`
Expected: all pass (`29 passed` total in this file so far).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Utils/tls_trust.py Tests/Utils/test_tls_trust.py
git commit -m "feat: TLS-trust-aware client factories for httpx and requests"
```

---

### Task 5: Shared httpx seams adopt the factories

**Files (Modify — one-line-ish edits each):**
- `tldw_chatbook/Chat/console_provider_gateway.py:1271` (`_new_owned_http_client`)
- `tldw_chatbook/TTS/base_backends.py:120`
- `tldw_chatbook/Image_Generation/http_client.py:119`
- `tldw_chatbook/LLM_Provider_Catalog/openai_compatible_model_discovery.py:822`
- `tldw_chatbook/Evals/word_bench/capture_client.py:205`
- `tldw_chatbook/Chat/local_server_discovery.py:499` and `:549`
- `tldw_chatbook/Tools/web_tool_impls.py:873`, `:925`, `:1746`
- Test: `Tests/Chat/test_console_gateway_tls_trust.py` (new)

**Interfaces:**
- Consumes: `build_httpx_async_client`, `build_httpx_client` (Task 4).
- Produces: all listed seams now construct policy-aware clients. No new public API.

- [ ] **Step 1: Write the failing seam test** — create `Tests/Chat/test_console_gateway_tls_trust.py`:

```python
"""The Console gateway's owned HTTP client honors the app TLS trust policy."""
import ssl

import pytest

import tldw_chatbook.Utils.tls_trust as tls_trust
from tldw_chatbook.Chat import console_provider_gateway as gateway_mod


@pytest.fixture
def _set_ssl_config(monkeypatch):
    def _install(value):
        monkeypatch.setattr(
            tls_trust,
            "get_cli_setting",
            lambda section, key=None, default=None: (
                value if (section, key) == ("network", "ssl_verify") else default
            ),
        )

    return _install


@pytest.mark.parametrize(
    ("config_value", "expected_mode"),
    [(False, ssl.CERT_NONE), (True, ssl.CERT_REQUIRED)],
)
def test_gateway_client_honors_tls_policy(_set_ssl_config, config_value, expected_mode):
    _set_ssl_config(config_value)
    client = gateway_mod.ConsoleProviderGateway._new_owned_http_client()
    ctx = client._transport._pool._ssl_context
    assert ctx.verify_mode == expected_mode
```

`_new_owned_http_client` (verified) is a `@staticmethod` on `ConsoleProviderGateway` (the class at `console_provider_gateway.py:1271`), so calling it off the class is correct.

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest Tests/Chat/test_console_gateway_tls_trust.py -v`
Expected: FAIL — the built client's context is `CERT_REQUIRED` even under `ssl_verify = false`, proving the seam isn't wired yet.

- [ ] **Step 3: Apply the seam edits.** In each file add the import `from ..Utils.tls_trust import build_httpx_async_client` (or `build_httpx_client`; use absolute `from tldw_chatbook.Utils.tls_trust import ...` if that file's import style is absolute — match the file's existing local-import style), then:

  - `console_provider_gateway.py:1271` — replace the body:
    ```python
    @staticmethod
    def _new_owned_http_client() -> httpx.AsyncClient:
        return build_httpx_async_client(
            timeout=httpx.Timeout(
                connect=GENERATION_CONNECT_TIMEOUT_SECONDS,
                read=GENERATION_READ_TIMEOUT_SECONDS,
                write=GENERATION_READ_TIMEOUT_SECONDS,
                pool=GENERATION_READ_TIMEOUT_SECONDS,
            )
        )
    ```
  - `TTS/base_backends.py:120` — `self.client = build_httpx_async_client(timeout=60.0)`
  - `Image_Generation/http_client.py:119` — `return build_httpx_client(` (keep the existing `timeout=`, `follow_redirects=`, `max_redirects=` arguments unchanged)
  - `LLM_Provider_Catalog/openai_compatible_model_discovery.py:822` — `async with build_httpx_async_client(timeout=timeout_seconds) as active_client:`
  - `Evals/word_bench/capture_client.py:205` — `self._client = build_httpx_async_client(**kwargs)` (the existing `transport`/`timeout` kwargs flow through unchanged)
  - `Chat/local_server_discovery.py:499` and `:549` — `client = http_client or build_httpx_async_client(timeout=timeout)`
  - `Tools/web_tool_impls.py:873`, `:925`, `:1746` — replace `httpx.Client(` with `build_httpx_client(` keeping the argument lists unchanged.

- [ ] **Step 4: Run the seam test and the touched modules' existing tests**

Run: `.venv/bin/python -m pytest Tests/Chat/test_console_gateway_tls_trust.py -v`
Expected: PASS.

Run: `.venv/bin/python -m pytest Tests/Evals/ Tests/TTS/ Tests/Image_Generation/ -q -k "capture or http_client or base_backend"`
Expected: PASS (guards against constructor-signature regressions in the touched modules; if a module has no matching tests, an empty selection is fine — `--strict-markers` doesn't affect `-k`).

- [ ] **Step 5: Grep-verify no bare client constructions remain in the touched files**

Run:
```bash
grep -n "httpx.AsyncClient(\|httpx.Client(" \
  tldw_chatbook/Chat/console_provider_gateway.py \
  tldw_chatbook/TTS/base_backends.py \
  tldw_chatbook/Image_Generation/http_client.py \
  tldw_chatbook/LLM_Provider_Catalog/openai_compatible_model_discovery.py \
  tldw_chatbook/Evals/word_bench/capture_client.py \
  tldw_chatbook/Chat/local_server_discovery.py \
  tldw_chatbook/Tools/web_tool_impls.py
```
Expected: only occurrences inside `tldw_chatbook/Utils/tls_trust.py`-style factory definitions, type annotations (`-> httpx.AsyncClient`), or comments — no live `httpx.AsyncClient(...)`/`httpx.Client(...)` constructor calls. (web_tool_impls line 1016's comment mentioning `httpx.Client()` stays as a comment.)

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Chat/console_provider_gateway.py tldw_chatbook/TTS/base_backends.py \
  tldw_chatbook/Image_Generation/http_client.py \
  tldw_chatbook/LLM_Provider_Catalog/openai_compatible_model_discovery.py \
  tldw_chatbook/Evals/word_bench/capture_client.py \
  tldw_chatbook/Chat/local_server_discovery.py tldw_chatbook/Tools/web_tool_impls.py \
  Tests/Chat/test_console_gateway_tls_trust.py
git commit -m "feat: shared httpx seams construct TLS-trust-aware clients"
```

---

### Task 6: `tldw_api` client `ssl_verify` param + app bootstrap wiring

**Files:**
- Modify: `tldw_chatbook/tldw_api/client.py` (constructor + `_get_client` at :1145)
- Modify: `tldw_chatbook/runtime_policy/bootstrap.py:227` and `:231`
- Test: `Tests/tldw_api/test_client_ssl_verify.py` (new; create `Tests/tldw_api/` if absent — check first with `ls Tests/` and place beside existing tldw_api tests if a differently named dir exists)

**Interfaces:**
- Consumes: `httpx_verify()` (Task 4) on the app side only.
- Produces: `TLDWAPIClient.__init__(..., ssl_verify: bool | str | ssl.SSLContext = True)` forwarded verbatim as the httpx `verify=`. The Apache-licensed client does NOT import `tldw_chatbook.Utils` — app callers pass the resolved value.

- [ ] **Step 1: Write the failing test** — create `Tests/tldw_api/test_client_ssl_verify.py`:

```python
"""TLDWAPIClient forwards its ssl_verify param into the underlying httpx client."""
import ssl

from tldw_chatbook.tldw_api.client import TLDWAPIClient


async def test_client_ssl_verify_false_disables_verification():
    client = TLDWAPIClient(base_url="https://example.invalid", ssl_verify=False)
    try:
        http = await client._get_client()
        assert http._transport._pool._ssl_context.verify_mode == ssl.CERT_NONE
    finally:
        await client.aclose()


async def test_client_ssl_verify_default_is_verification():
    client = TLDWAPIClient(base_url="https://example.invalid")
    try:
        http = await client._get_client()
        assert http._transport._pool._ssl_context.verify_mode == ssl.CERT_REQUIRED
    finally:
        await client.aclose()
```

If `TLDWAPIClient` requires more constructor args — it does not; the verified signature is `__init__(self, base_url: str, token: Optional[str] = None, timeout: float = 300.0, connect_timeout: Optional[float] = None)` — add `ssl_verify` as the fifth parameter after `connect_timeout`.

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest Tests/tldw_api/test_client_ssl_verify.py -v`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'ssl_verify'`.

- [ ] **Step 3: Edit `tldw_api/client.py`.** In `__init__` add the parameter (next to the existing `timeout`/`connect_timeout` params) and store it:

```python
            ssl_verify: bool | str | "ssl.SSLContext" = True,
```

(the file has `from __future__ import annotations`; add `import ssl` to the stdlib imports if not present), and in the body alongside the timeout assignments:

```python
        self.ssl_verify = ssl_verify
```

In `_get_client` (line ~1145) add `verify=self.ssl_verify,` to the `httpx.AsyncClient(...)` call. Add one docstring line in `__init__`: `ssl_verify: TLS trust for the client (True/False/CA path/SSLContext), forwarded to httpx verify.`

Then in `runtime_policy/bootstrap.py` add the import `from ..Utils.tls_trust import httpx_verify` (match that file's local-import style) and change both constructions at :227 and :231:

```python
        client = TLDWAPIClient(base_url=resolved_endpoint, ssl_verify=httpx_verify())
```

```python
    return TLDWAPIClient(base_url=resolved_endpoint, token=resolved_auth_token, ssl_verify=httpx_verify())
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest Tests/tldw_api/ -q`
Expected: PASS (existing tldw_api client tests keep passing — default behavior unchanged).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/tldw_api/client.py tldw_chatbook/runtime_policy/bootstrap.py Tests/tldw_api/test_client_ssl_verify.py
git commit -m "feat: tldw_api client ssl_verify param wired from app TLS policy"
```

---

### Task 7: requests long tail — `session.verify` threading

**Files (Modify — mechanical, one line per site):**

| File | Sites | Edit |
|---|---|---|
| `tldw_chatbook/LLM_Calls/LLM_API_Calls.py` | 15 × `requests.Session()` + 1 direct `requests.post` (inside `get_openai_embeddings`, def at :444, post at :475) | add `session.verify = requests_verify()` after each Session(); add `verify=requests_verify()` kwarg to the direct post |
| `tldw_chatbook/LLM_Calls/hosted_chat.py` | 1 × Session() at :533 | same one-liner (covers `moonshot.py`/`zai.py`, which route through hosted_chat) |
| `tldw_chatbook/LLM_Calls/qwencloud.py` | 1 × Session() at :1168 | same one-liner |
| `tldw_chatbook/LLM_Calls/Local_Summarization_Lib.py` | 15 × Session() | same one-liner |
| `tldw_chatbook/LLM_Calls/Summarization_General_Lib.py` | 16 × Session() | same one-liner |
| `tldw_chatbook/Web_Scraping/WebSearch_APIs.py` | 3 × Session() (:2718, :3089, :3759) | same one-liner |

**Interfaces:**
- Consumes: `requests_verify()` (Task 3).
- Produces: no new API; behavior only.

- [ ] **Step 1: Write the failing spot test** — append to `Tests/Utils/test_tls_trust.py`:

```python
def test_get_openai_embeddings_passes_tls_policy(_set_ssl_config, monkeypatch):
    """Representative direct-requests.post site threads verify= through."""
    import tldw_chatbook.LLM_Calls.LLM_API_Calls as llm_calls

    captured: dict = {}

    class _FakeResponse:
        status_code = 200

        def json(self):
            return {"data": [{"embedding": [0.0, 1.0]}]}

    def _fake_post(url, **kwargs):
        captured.update(kwargs)
        captured["url"] = url
        return _FakeResponse()

    monkeypatch.setattr(llm_calls.requests, "post", _fake_post)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    _set_ssl_config(False)
    llm_calls.get_openai_embeddings("hello", "text-embedding-3-small")
    assert captured.get("verify") is False
```

If the call raises a missing-API-key error before posting, check how `get_openai_embeddings` resolves its key (see `resolve_provider_api_key` in `tldw_chatbook/config.py`) and set the matching env var via `monkeypatch.setenv` — do not change the production code under test.

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest Tests/Utils/test_tls_trust.py::test_get_openai_embeddings_passes_tls_policy -v`
Expected: FAIL — `captured.get("verify")` is `None` because the post doesn't pass `verify=` yet.

- [ ] **Step 3: Apply the threading edits.** For every file in the table above:

1. Add the import near the other local imports, matching the file's style (`from tldw_chatbook.Utils.tls_trust import requests_verify` — absolute form is what `LLM_Calls/` files use for local imports, e.g. `from tldw_chatbook.config import ...`).
2. Directly after each `session = requests.Session()` line, add `session.verify = requests_verify()` at the same indentation.
3. Only in `LLM_API_Calls.py` `get_openai_embeddings`: add `verify=requests_verify(),` to the `requests.post(...)` call at :475.

- [ ] **Step 4: Grep-verify every Session() site got the line**

```bash
for f in tldw_chatbook/LLM_Calls/LLM_API_Calls.py tldw_chatbook/LLM_Calls/hosted_chat.py \
         tldw_chatbook/LLM_Calls/qwencloud.py tldw_chatbook/LLM_Calls/Local_Summarization_Lib.py \
         tldw_chatbook/LLM_Calls/Summarization_General_Lib.py tldw_chatbook/Web_Scraping/WebSearch_APIs.py; do
  echo "$f sessions=$(grep -c 'requests.Session()' $f) wired=$(grep -c 'session.verify = requests_verify()' $f)"
done
```
Expected: `sessions == wired` on every line (15/15, 1/1, 1/1, 15/15, 16/16, 3/3).

Run: `.venv/bin/python -m pytest Tests/Utils/test_tls_trust.py -v`
Expected: all pass.

- [ ] **Step 5: Run the LLM_Calls-adjacent targeted tests**

Run: `.venv/bin/python -m pytest Tests/LLM_Calls/ -q -k "embeddings or hosted or qwencloud or summarization"`
Expected: PASS (or empty selection if names differ — then run `.venv/bin/python -m pytest Tests/LLM_Calls/ -q` if that directory is small; if it is large, keep to `-k` selection).

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/LLM_Calls/ tldw_chatbook/Web_Scraping/WebSearch_APIs.py Tests/Utils/test_tls_trust.py
git commit -m "feat: thread TLS trust policy through requests long tail"
```

---

### Task 8: aiohttp, websockets, and OpenAI-SDK seams

**Files (Modify):**
- `tldw_chatbook/Web_Scraping/Article_Scraper/crawler.py:185` and `:350`
- `tldw_chatbook/Media_Creation/swarmui_client.py:139-141`
- `tldw_chatbook/LLM_Calls/realtime/transport.py:113`
- `tldw_chatbook/Local_Ingestion/OCR_Backends.py:826`
- Test: `Tests/LLM_Calls/test_realtime_tls_trust.py` (new)

**Interfaces:**
- Consumes: `ssl_context_for_transport()` (Task 3), `build_httpx_client` (Task 4).
- Produces: no new API.

- [ ] **Step 1: Write the failing test** — create `Tests/LLM_Calls/test_realtime_tls_trust.py`:

```python
"""WsTransport passes the app TLS policy to websockets.connect for wss:// URLs."""
import asyncio
import types

import pytest

import tldw_chatbook.Utils.tls_trust as tls_trust
from tldw_chatbook.LLM_Calls.realtime import transport as transport_mod


class _FakeWebsockets(types.SimpleNamespace):
    def __init__(self):
        captured = {}
        self.captured = captured

        async def connect(url, **kwargs):
            captured["url"] = url
            captured["kwargs"] = kwargs
            return object()

        self.connect = connect


@pytest.fixture
def _set_ssl_config(monkeypatch):
    def _install(value):
        monkeypatch.setattr(
            tls_trust,
            "get_cli_setting",
            lambda section, key=None, default=None: (
                value if (section, key) == ("network", "ssl_verify") else default
            ),
        )

    return _install


@pytest.mark.parametrize(
    ("url", "config_value", "ssl_expected"),
    [
        ("wss://example.invalid/rt", False, False),
        ("ws://example.invalid/rt", False, None),  # never passes ssl for ws://
        ("wss://example.invalid/rt", True, None),  # default policy -> no ssl kwarg
    ],
)
async def test_transport_passes_tls_policy(_set_ssl_config, url, config_value, ssl_expected):
    _set_ssl_config(config_value)
    fake = _FakeWebsockets()
    t = transport_mod.WsTransport()
    t._ws = None
    monkeypatched_connect = fake.connect
    orig = transport_mod._websockets
    transport_mod._websockets = lambda: fake
    try:
        await t.connect(url, headers={})
    finally:
        transport_mod._websockets = orig
    kwargs = fake.captured["kwargs"]
    if ssl_expected is None:
        assert "ssl" not in kwargs
    else:
        assert kwargs["ssl"] is ssl_expected
```

(Note: `_websockets` is the module's lazy-import helper referenced at `transport.py:41`; `WsTransport` is the class whose `connect` is at :95. If the helper's name differs, use whatever `connect()` calls at :113 to obtain the module.)

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest Tests/LLM_Calls/test_realtime_tls_trust.py -v`
Expected: FAIL — `"ssl" not in kwargs` fails for the `wss` + `False` case (transport doesn't pass ssl yet).

- [ ] **Step 3: Apply the edits.**

`realtime/transport.py` — replace the connect line at :113 (`self._ws = await websockets.connect(url, additional_headers=headers)`) with:

```python
            ssl_arg = ssl_context_for_transport()
            if ssl_arg is not None and not url.startswith("wss://"):
                # websockets rejects a non-None ssl argument for ws:// URIs.
                ssl_arg = None
            connect_kwargs: dict = {"additional_headers": headers}
            if ssl_arg is not None:
                connect_kwargs["ssl"] = ssl_arg
            self._ws = await websockets.connect(url, **connect_kwargs)
```

with import `from ...Utils.tls_trust import ssl_context_for_transport` adjusted to the file's import style (it imports `tldw_chatbook.*` absolute elsewhere).

`crawler.py:185` and `:350` — `async with aiohttp.ClientSession() as session:` becomes:

```python
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(ssl=ssl_context_for_transport())
    ) as session:
```

`swarmui_client.py:139-141` — `connector = aiohttp.TCPConnector(limit=10)` becomes `connector = aiohttp.TCPConnector(limit=10, ssl=ssl_context_for_transport())`.

`OCR_Backends.py:826` — becomes:

```python
                    self.client = OpenAI(
                        api_key=api_key,
                        base_url=base_url,
                        http_client=build_httpx_client(timeout=60.0),
                    )
```

with `from ..Utils.tls_trust import build_httpx_client, ssl_context_for_transport` imports added to each edited file (match each file's import style; crawler/swarmui import `ssl_context_for_transport` only, OCR imports `build_httpx_client` only).

- [ ] **Step 4: Run tests + grep-verify**

Run: `.venv/bin/python -m pytest Tests/LLM_Calls/test_realtime_tls_trust.py -v`
Expected: PASS (all three parametrized cases).

Run:
```bash
grep -n "TCPConnector" tldw_chatbook/Web_Scraping/Article_Scraper/crawler.py tldw_chatbook/Media_Creation/swarmui_client.py
grep -n "http_client=build_httpx_client" tldw_chatbook/Local_Ingestion/OCR_Backends.py
```
Expected: both crawler sites and the swarmui site show `ssl=ssl_context_for_transport()`; OCR shows the injected http_client.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/LLM_Calls/realtime/transport.py tldw_chatbook/Web_Scraping/Article_Scraper/crawler.py \
  tldw_chatbook/Media_Creation/swarmui_client.py tldw_chatbook/Local_Ingestion/OCR_Backends.py \
  Tests/LLM_Calls/test_realtime_tls_trust.py
git commit -m "feat: TLS trust policy on aiohttp, websockets, and OpenAI-SDK seams"
```

---

### Task 9: F9 Settings — Network category

**Files:**
- Create: `tldw_chatbook/UI/Screens/settings_network_defaults.py`
- Modify: `tldw_chatbook/UI/Screens/settings_config_models.py:10` (`SettingsCategoryId`)
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py` — six edit points: `_category_summaries()` (:3328 area), `_category_groups()` (:3517 area), `_render_detail_pane` dispatcher (:15271 area), handlers + staging methods (near the appearance handlers at :18092 area), `__init__` state (after `self._settings_drafts` init at :2473), save branch in `action_settings_save_category` (:20765, appearance branch at :21316), import block (near :275)
- Test: `Tests/UI/test_settings_network_defaults.py` (new), `Tests/UI/test_settings_network_category.py` (new)

**Interfaces:**
- Consumes: `SettingsValidationResult` from `settings_config_models.py:50`; `SettingsConfigAdapter` from `settings_config_adapter.py`; pilot-test harnesses `_build_test_app` (`Tests/UI/app_factory.py:166`), `DestinationHarness`, `_settle_settings`, `_click_settings_category`, `_active_destination_screen` (copy the import block from `Tests/UI/test_settings_category_sweep.py`).
- Produces: `load_network_tls`, `validate_network_tls`, `build_network_save_sections`, `network_ssl_toml_value`, `SettingsNetworkTLS` (mode strings exactly `"verify" | "off" | "custom-ca" | "invalid"`).

- [ ] **Step 1: Create `settings_network_defaults.py`**:

```python
"""Load/validate/save model for the Settings "Network" category ([network]).

Mirrors settings_appearance_defaults.py: pure functions over a config
mapping so they unit-test without an app.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .settings_config_models import SettingsValidationResult

_TRUE_STRINGS = frozenset({"true", "1", "on"})
_FALSE_STRINGS = frozenset({"false", "0", "no", "off"})


@dataclass(frozen=True)
class SettingsNetworkTLS:
    mode: str  # "verify" | "off" | "custom-ca" | "invalid"
    ca_bundle_path: str = ""
    raw: object = None


def load_network_tls(app_config: Mapping[str, Any]) -> SettingsNetworkTLS:
    network = app_config.get("network") if isinstance(app_config, Mapping) else None
    value = network.get("ssl_verify", True) if isinstance(network, Mapping) else True
    if value is True:
        return SettingsNetworkTLS("verify")
    if value is False:
        return SettingsNetworkTLS("off")
    if isinstance(value, str):
        lowered = value.strip().lower()
        if not lowered or lowered in _TRUE_STRINGS:
            return SettingsNetworkTLS("verify", raw=value)
        if lowered in _FALSE_STRINGS:
            return SettingsNetworkTLS("off", raw=value)
        path = Path(value.strip()).expanduser()
        if path.is_file():
            return SettingsNetworkTLS("custom-ca", ca_bundle_path=str(path), raw=value)
        return SettingsNetworkTLS("invalid", ca_bundle_path=value.strip(), raw=value)
    return SettingsNetworkTLS("invalid", raw=value)


def validate_network_tls(values: SettingsNetworkTLS) -> SettingsValidationResult:
    if values.mode == "invalid":
        return SettingsValidationResult(
            False, "ssl_verify value is invalid — choose a mode and save."
        )
    if values.mode == "custom-ca":
        raw = values.ca_bundle_path.strip()
        if not raw:
            return SettingsValidationResult(
                False, "Custom CA bundle requires a file path."
            )
        path = Path(raw).expanduser()
        if not path.is_file():
            return SettingsValidationResult(
                False, f"CA bundle file not found: {path}"
            )
        if not os.access(path, os.R_OK):
            return SettingsValidationResult(
                False, f"CA bundle file is not readable: {path}"
            )
    return SettingsValidationResult(True, "")


def network_ssl_toml_value(values: SettingsNetworkTLS) -> bool | str:
    if values.mode == "off":
        return False
    if values.mode == "custom-ca":
        return str(Path(values.ca_bundle_path.strip()).expanduser())
    return True


def build_network_save_sections(
    values: SettingsNetworkTLS,
) -> dict[str, dict[str, Any]]:
    return {"network": {"ssl_verify": network_ssl_toml_value(values)}}
```

- [ ] **Step 2: Write unit tests** — create `Tests/UI/test_settings_network_defaults.py`:

```python
from tldw_chatbook.UI.Screens.settings_network_defaults import (
    SettingsNetworkTLS,
    build_network_save_sections,
    load_network_tls,
    network_ssl_toml_value,
    validate_network_tls,
)


def test_load_defaults_to_verify():
    assert load_network_tls({}).mode == "verify"


def test_load_bool_off():
    assert load_network_tls({"network": {"ssl_verify": False}}).mode == "off"


def test_load_lenient_strings():
    assert load_network_tls({"network": {"ssl_verify": "off"}}).mode == "off"
    assert load_network_tls({"network": {"ssl_verify": "1"}}).mode == "verify"


def test_load_custom_ca(tmp_path):
    ca = tmp_path / "corp.pem"
    ca.write_text("# ca")
    loaded = load_network_tls({"network": {"ssl_verify": str(ca)}})
    assert loaded.mode == "custom-ca"
    assert loaded.ca_bundle_path == str(ca)


def test_load_missing_path_is_invalid():
    loaded = load_network_tls({"network": {"ssl_verify": "/nope.pem"}})
    assert loaded.mode == "invalid"
    assert loaded.raw == "/nope.pem"


def test_load_unsupported_type_is_invalid():
    assert load_network_tls({"network": {"ssl_verify": 7}}).mode == "invalid"


def test_validate_custom_ca_requires_existing_readable_file(tmp_path):
    missing = validate_network_tls(SettingsNetworkTLS("custom-ca", "/nope.pem"))
    assert not missing.valid
    empty = validate_network_tls(SettingsNetworkTLS("custom-ca", "  "))
    assert not empty.valid
    ca = tmp_path / "corp.pem"
    ca.write_text("# ca")
    ok = validate_network_tls(SettingsNetworkTLS("custom-ca", str(ca)))
    assert ok.valid
    assert validate_network_tls(SettingsNetworkTLS("verify")).valid
    assert validate_network_tls(SettingsNetworkTLS("off")).valid
    assert not validate_network_tls(SettingsNetworkTLS("invalid", raw="x")).valid


def test_build_sections_round_trip(tmp_path):
    ca = tmp_path / "corp.pem"
    ca.write_text("# ca")
    assert build_network_save_sections(SettingsNetworkTLS("off")) == {
        "network": {"ssl_verify": False}
    }
    assert build_network_save_sections(SettingsNetworkTLS("verify")) == {
        "network": {"ssl_verify": True}
    }
    assert build_network_save_sections(
        SettingsNetworkTLS("custom-ca", str(ca))
    ) == {"network": {"ssl_verify": str(ca)}}
```

- [ ] **Step 3: Run the unit tests** — pure functions, they should pass immediately (this file is data-shape logic written test-first in Step 1-2 order; if any assertion fails, fix the module, not the test):

Run: `.venv/bin/python -m pytest Tests/UI/test_settings_network_defaults.py -v`
Expected: all PASS.

- [ ] **Step 4: Wire the category into the screen.** Six edits to `settings_screen.py` plus the enum:

1. `settings_config_models.py` — inside `SettingsCategoryId`, next to `PRIVACY_SECURITY`, add: `NETWORK = "network"`.
2. `_category_summaries()` — next to the `PRIVACY_SECURITY` summary entry add:
   ```python
            SettingsCategorySummary(
                SettingsCategoryId.NETWORK,
                "Network",
                "TLS trust for outbound API traffic (corporate DPI networks).",
                "Guided",
            ),
   ```
3. `_category_groups()` — in the `"Data & Privacy"` tuple after `PRIVACY_SECURITY`, add `SettingsCategoryId.NETWORK`.
4. Import block (near `from .settings_appearance_defaults import ...` at :275) add:
   ```python
   from .settings_network_defaults import (
       SettingsNetworkTLS,
       build_network_save_sections,
       load_network_tls,
       validate_network_tls,
   )
   from .settings_config_adapter import SettingsConfigAdapter
   ```
   (If `SettingsConfigAdapter` is already imported, don't duplicate it.)
5. Near the appearance-mode constants, add the options list and warning text:
   ```python
   _NETWORK_TLS_MODE_OPTIONS: list[tuple[str, str]] = [
       ("verify", "Verify certificates (default)"),
       ("off", "Disable verification"),
       ("custom-ca", "Custom CA bundle"),
   ]
   ```
6. `__init__` (right after `self._settings_drafts` init at :2473): `self._network_pending: dict[str, object] = {}`.
7. `_render_detail_pane` dispatcher — add a branch:
   ```python
   elif category is SettingsCategoryId.NETWORK:
       yield from self._render_network_detail()
   ```
8. New screen methods (place near the appearance render/helpers):
   ```python
   def _render_network_detail(self) -> ComposeResult:
       values = load_network_tls(self._app_config_mapping())
       yield Static("Network", classes="destination-section settings-column-title")
       with Vertical(id="settings-network-card", classes="settings-focus-card"):
           yield Static(
               "TLS trust for outbound API traffic", classes="destination-section"
           )
           if values.mode == "invalid":
               yield Static(
                   f"Config has an invalid [network] ssl_verify value"
                   f" ({values.raw!r}); default verification is in use until"
                   " it is fixed.",
                   id="settings-network-invalid-row",
                   classes="settings-network-error",
               )
           with Horizontal(classes="settings-input-row settings-select-row"):
               yield Static("Certificate verification", classes="settings-input-label")
               yield Select(
                   _NETWORK_TLS_MODE_OPTIONS,
                   value="verify" if values.mode == "invalid" else values.mode,
                   id="settings-network-ssl-mode",
                   classes="settings-compact-select",
                   allow_blank=False,
                   compact=True,
               )
           with Horizontal(classes="settings-input-row"):
               yield Static("CA bundle path", classes="settings-input-label")
               yield Input(
                   value=values.ca_bundle_path if values.mode == "custom-ca" else "",
                   id="settings-network-ca-path",
                   classes="settings-compact-input",
                   placeholder="/path/to/corp-ca.pem (used by 'Custom CA bundle')",
               )
           yield Static(
               self._network_warning_text(self._network_effective_mode()),
               id="settings-network-warning",
               classes="settings-network-warning",
           )

   def _network_effective_mode(self) -> str:
       loaded = load_network_tls(self._app_config_mapping())
       pending_mode = self._network_pending.get("mode")
       return pending_mode if isinstance(pending_mode, str) else loaded.mode

   @staticmethod
   def _network_warning_text(mode: str) -> str:
       if mode == "off":
           return (
               "Verification is DISABLED: API keys and conversation content"
               " can be intercepted by anyone on the network path."
           )
       if mode == "custom-ca":
           return (
               "Verification additionally trusts your custom CA bundle"
               " (corporate root CA)."
           )
       return ""

   def _update_network_warning(self) -> None:
       try:
           widget = self.query_one("#settings-network-warning", Static)
       except Exception:
           return
       widget.update(self._network_warning_text(self._network_effective_mode()))

   @on(Select.Changed, "#settings-network-ssl-mode")
   def handle_network_ssl_mode_changed(self, event: Select.Changed) -> None:
       event.stop()
       self._network_pending["mode"] = str(event.value or "verify")
       self._update_network_warning()

   @on(Input.Changed, "#settings-network-ca-path")
   def handle_network_ca_path_changed(self, event: Input.Changed) -> None:
       self._network_pending["ca_bundle_path"] = event.value

   def _network_effective_values(self) -> SettingsNetworkTLS:
       loaded = load_network_tls(self._app_config_mapping())
       mode = self._network_effective_mode()
       if mode in ("verify", "off", "invalid"):
           return SettingsNetworkTLS(mode, raw=loaded.raw)
       path = str(
           self._network_pending.get("ca_bundle_path", loaded.ca_bundle_path)
       )
       return SettingsNetworkTLS(mode, ca_bundle_path=path, raw=loaded.raw)
   ```
   Ensure `Select` and `Static` are imported (they are — the appearance category uses them).
9. Save branch — inside `action_settings_save_category` (:20765), immediately above the `if category is SettingsCategoryId.APPEARANCE:` branch at :21316, add:
   ```python
       if category is SettingsCategoryId.NETWORK:
           values = self._network_effective_values()
           validation = validate_network_tls(values)
           if not validation.valid:
               self.app.notify(validation.message, severity="error")
               return
           section_values = build_network_save_sections(values)
           saved = SettingsConfigAdapter().save_sections(section_values)
           self.app.notify(
               "Network TLS setting saved."
               if saved
               else "Failed to save Network TLS setting.",
               severity="information" if saved else "error",
           )
           if saved:
               self._network_pending = {}
           return
   ```

- [ ] **Step 5: Write the pilot test** — create `Tests/UI/test_settings_network_category.py`, importing the harness helpers exactly the way `Tests/UI/test_settings_category_sweep.py` does (copy its import block for `_build_test_app`, `DestinationHarness`, `_settle_settings`, `_click_settings_category`, `_active_destination_screen`):

```python
import pytest
from tldw_chatbook.UI.Screens.settings_config_adapter import SettingsConfigAdapter

pytestmark = pytest.mark.ui


async def test_network_category_rejects_missing_ca_and_saves_valid_one(tmp_path, monkeypatch):
    saved: list[dict] = []

    def _capture(sections):
        saved.append({k: dict(v) for k, v in dict(sections).items()})
        return True

    monkeypatch.setattr(SettingsConfigAdapter, "save_sections", staticmethod(_capture))
    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(120, 35)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "network")
        screen = _active_destination_screen(host)
        assert screen.query_one("#settings-network-ssl-mode") is not None

        screen._network_pending["mode"] = "custom-ca"
        screen._network_pending["ca_bundle_path"] = "/definitely/not/here.pem"
        screen.action_settings_save_category()  # verified sync: -> None
        assert saved == []  # invalid path rejected, nothing written

        ca = tmp_path / "corp.pem"
        ca.write_text("# corp ca")
        screen._network_pending["ca_bundle_path"] = str(ca)
        screen.action_settings_save_category()
        assert saved == [{"network": {"ssl_verify": str(ca)}}]
        assert screen._network_pending == {}  # draft cleared after successful save
```

- [ ] **Step 6: Run the new UI tests and the automatic category sweep**

Run: `.venv/bin/python -m pytest Tests/UI/test_settings_network_defaults.py Tests/UI/test_settings_network_category.py -v`
Expected: PASS.

Run: `.venv/bin/python -m pytest Tests/UI/test_settings_category_sweep.py -q`
Expected: PASS — the sweep auto-derives categories from `_category_summaries()`, so it now exercises the Network category's rendering at both sizes (this catches compose errors the focused test misses).

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/UI/Screens/settings_network_defaults.py tldw_chatbook/UI/Screens/settings_config_models.py \
  tldw_chatbook/UI/Screens/settings_screen.py Tests/UI/test_settings_network_defaults.py Tests/UI/test_settings_network_category.py
git commit -m "feat: F9 Network category for TLS trust (select + CA path + warnings)"
```

---

### Task 10: Final sweep, backlog close-out, and DoD wrap-up

**Files:**
- Modify: backlog task file for `<TASK_ID>` (via `backlog` CLI)
- No production code changes expected — this task verifies and closes.

- [ ] **Step 1: Completeness greps** — all must show the stated counts:

```bash
echo "--- requests long tail (sessions == wired) ---"
for f in tldw_chatbook/LLM_Calls/LLM_API_Calls.py tldw_chatbook/LLM_Calls/hosted_chat.py \
         tldw_chatbook/LLM_Calls/qwencloud.py tldw_chatbook/LLM_Calls/Local_Summarization_Lib.py \
         tldw_chatbook/LLM_Calls/Summarization_General_Lib.py tldw_chatbook/Web_Scraping/WebSearch_APIs.py; do
  echo "$f sessions=$(grep -c 'requests.Session()' $f) wired=$(grep -c 'session.verify = requests_verify()' $f)"
done
echo "--- aiohttp/websockets/SDK seams ---"
grep -c "ssl=ssl_context_for_transport()" tldw_chatbook/Web_Scraping/Article_Scraper/crawler.py tldw_chatbook/Media_Creation/swarmui_client.py
grep -c "ssl_context_for_transport()" tldw_chatbook/LLM_Calls/realtime/transport.py
grep -c "http_client=build_httpx_client" tldw_chatbook/Local_Ingestion/OCR_Backends.py
echo "--- shared httpx seams ---"
grep -c "build_httpx_async_client\|build_httpx_client" tldw_chatbook/Chat/console_provider_gateway.py \
  tldw_chatbook/TTS/base_backends.py tldw_chatbook/Image_Generation/http_client.py \
  tldw_chatbook/LLM_Provider_Catalog/openai_compatible_model_discovery.py \
  tldw_chatbook/Evals/word_bench/capture_client.py tldw_chatbook/Chat/local_server_discovery.py \
  tldw_chatbook/runtime_policy/bootstrap.py tldw_chatbook/Tools/web_tool_impls.py
```

Expected: sessions==wired per file (15/1/1/15/16/3); crawler 2, swarmui 1, realtime ≥1, OCR 1; every shared seam file ≥1.

- [ ] **Step 2: Targeted test sweep** (everything this plan touched — still not a full-suite run):

```bash
.venv/bin/python -m pytest Tests/Utils/test_tls_trust.py \
  Tests/Chat/test_console_gateway_tls_trust.py \
  Tests/tldw_api/test_client_ssl_verify.py \
  Tests/LLM_Calls/test_realtime_tls_trust.py \
  Tests/UI/test_settings_network_defaults.py \
  Tests/UI/test_settings_network_category.py \
  Tests/UI/test_settings_category_sweep.py -v
```
Expected: ALL PASS.

- [ ] **Step 3: Lint the touched source**

```bash
.venv/bin/python -m ruff check tldw_chatbook/Utils/tls_trust.py tldw_chatbook/UI/Screens/settings_network_defaults.py \
  tldw_chatbook/UI/Screens/settings_screen.py tldw_chatbook/UI/Screens/settings_config_models.py \
  tldw_chatbook/config.py tldw_chatbook/tldw_api/client.py tldw_chatbook/runtime_policy/bootstrap.py \
  tldw_chatbook/LLM_Calls/ tldw_chatbook/Web_Scraping/WebSearch_APIs.py \
  tldw_chatbook/Web_Scraping/Article_Scraper/crawler.py tldw_chatbook/Media_Creation/swarmui_client.py \
  tldw_chatbook/Local_Ingestion/OCR_Backends.py Tests/Utils/test_tls_trust.py
```
Expected: no findings (fix any it reports before continuing).

- [ ] **Step 4: Close out the backlog task** per AGENTS.md DoD:

```bash
backlog task edit <TASK_ID> -s Done --notes "Implemented per Docs/superpowers/plans/2026-08-29-network-tls-trust-policy.md. Helper: tldw_chatbook/Utils/tls_trust.py (normalization + additive contexts + merged bundle + factories + warn/metrics). Adoption: shared httpx seams (gateway/TTS/image-gen/catalog/evals/discovery/web tools), tldw_api ssl_verify ctor param + bootstrap, requests long tail (LLM_API_Calls/hosted_chat/qwencloud/summarization libs/web search), aiohttp (crawler/swarmui), websockets (realtime, ws:// guarded), OpenAI SDK (OCR). UI: F9 Network category. ADR: backlog/decisions/079-network-tls-trust-policy.md. Spec: Docs/superpowers/specs/2026-08-29-network-tls-trust-policy-design.md."
```

Then open the task file and: tick every `- [ ]` AC checkbox to `- [x]`, and verify the Implementation Notes section exists (the `--notes` above seeds it; extend if the editor wants the file richer). Decide the lessons question explicitly: if nothing here generalizes beyond this task (expected), record that decision in the notes ("no lessons entry — nothing generalized beyond this feature") rather than leaving it unconsidered.

- [ ] **Step 5: Final commit**

```bash
git add backlog/tasks/  # the edited <TASK_ID> task file
git commit -m "docs(backlog): close <TASK_ID> — network TLS trust policy delivered"
```

---

## Self-Review (completed during planning)

- **Spec coverage:** config schema §1 → Task 1 (+ normalization in Task 2, settings-side load in Task 9); helper module §2 → Tasks 2-4; adoption inventory §3 → Tasks 5 (shared seams), 6 (tldw_api), 7 (requests tail), 8 (aiohttp/websockets/SDK); settings UI §4 → Task 9; error handling §5 → fail-safe tests in Tasks 2-3, invalid-row in Task 9; testing §6 → per-task tests + Task 10 sweep; ADR §ADR → Task 1. Out-of-scope list honored (no Subscriptions/MCP/Playwright/Model_Artifacts/Web_Server edits anywhere).
- **Placeholder scan:** every code step carries real code; the one environment-conditional instruction (the OPENAI key env var in Task 7) names the exact anchor to check (`resolve_provider_api_key` in `tldw_chatbook/config.py`) rather than deferring design. The embedded test certificate is a real, parseable, throwaway-key self-signed PEM verified during planning (a fake base64 body would not load).
- **Type consistency:** `tls_verify_setting() -> bool | str`, `requests_verify() -> bool | str`, `ssl_context_for_transport() -> None | bool | ssl.SSLContext`, `httpx_verify() -> bool | ssl.SSLContext`, factory names consistent across Tasks 2-8; `SettingsNetworkTLS` mode strings (`verify|off|custom-ca|invalid`) consistent between loader, validator, options list, and pilot test.
