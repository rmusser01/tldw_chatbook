# Egress Log URL Redaction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent request credentials and URL details from reaching every egress-policy log line.

**Architecture:** Keep redaction local to the two logging boundaries in `egress.py`. A private parser renders only a safe HTTP(S) origin and falls back to a constant for malformed input; policy decisions and caller-visible errors do not change.

**Tech Stack:** Python 3.11, `urllib.parse`, Loguru, pytest

---

### Task 1: Redact URLs at both egress log sites

**Files:**
- Modify: `tldw_chatbook/Utils/egress.py`
- Test: `Tests/Utils/test_egress.py`
- Modify: `Docs/security/production-diagnostic-inventory.json`
- Modify: `backlog/tasks/task-1722 - Redact-credentials-from-egress-block-logs.md`

- [x] **Step 1: Write failing warning- and debug-path tests**

Add `test_egress_block_log_redacts_url_credentials`, patch `_resolve` to return `10.0.0.1`, capture `egress.logger.warning`, and evaluate a URL containing userinfo, a path, a token query, and a fragment.

Add `test_disabled_egress_log_redacts_url_credentials`, patch `get_cli_setting` to return `False` for the `enabled` key, capture `egress.logger.debug`, and evaluate the same credential-bearing URL.

For both messages, assert that `https://example.test:8443` remains present while the userinfo, path, query, fragment, and their secret values are absent.

- [x] **Step 2: Run the focused tests and verify the current leak**

Run: `pytest -q Tests/Utils/test_egress.py -k "test_egress_block_log_redacts_url_credentials or test_disabled_egress_log_redacts_url_credentials"`

Expected: FAIL because the current messages contain the full URL.

- [x] **Step 3: Implement the minimal log-label helper**

Add this private helper and call it from `_blocked` and the disabled branch of `_pre_resolution`:

```python
def _log_origin(url: str) -> str:
    """Return a credential- and query-free URL label for transport logs."""
    try:
        parsed = urlparse(url)
        host = parsed.hostname
        port = parsed.port
    except ValueError:
        return "<invalid-url>"
    if not host or parsed.scheme not in ("http", "https"):
        return "<invalid-url>"
    rendered_host = f"[{host}]" if ":" in host else host
    rendered_port = f":{port}" if port is not None else ""
    return f"{parsed.scheme}://{rendered_host}{rendered_port}"
```

- [x] **Step 4: Run the focused tests and full egress module**

Run: `pytest -q Tests/Utils/test_egress.py -k "test_egress_block_log_redacts_url_credentials or test_disabled_egress_log_redacts_url_credentials"`

Expected: PASS.

Run: `pytest -q Tests/Utils/test_egress.py`

Expected: PASS.

- [x] **Step 5: Run static checks and inspect the diff**

Regenerate the reviewed persistent-diagnostic inventory, verify that only the existing `egress.py` owner digest changes, and run its architecture gate:

Run: `python scripts/check_persistent_diagnostic_inventory.py --write`

Run: `pytest -q Tests/Architecture/test_persistent_diagnostic_inventory.py`

Run: `ruff check tldw_chatbook/Utils/egress.py Tests/Utils/test_egress.py`

Run: `ruff format --check tldw_chatbook/Utils/egress.py Tests/Utils/test_egress.py`

Run: `git diff --check`

Expected: all commands succeed.

- [x] **Step 6: Complete TASK-1722 bookkeeping and commit**

Check every acceptance criterion, add concise implementation notes, set the task to Done only after verification, then commit the task, design, plan, implementation, and tests together.
