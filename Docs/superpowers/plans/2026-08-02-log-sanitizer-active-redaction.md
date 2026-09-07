# Active Credential Redaction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the active credential sanitizer, separate credential redaction from model-name display validation, and remove the monitored URL from subscription diagnostics without breaking the existing public sanitizer imports.

**Architecture:** `Utils/log_sanitizer.py` composes the canonical sensitive-config-key predicate with an exact log/protocol field set, then applies a classify-first assignment scanner followed by bounded standalone rules. Ollama and Transformers model names move to bounded single-line input validation, while the subscription diagnostic omits its URL entirely. Tests exercise direct production functions and the full production app only; the existing installed-wheel probe verifies the packaged import path.

**Tech Stack:** Python 3.11+, `re`, `tomllib`, Textual production helpers, Loguru, pytest, Ruff, Backlog.md

**ADR required:** yes

**ADR path:** `backlog/decisions/029-local-private-data-boundary.md`

**Reason:** This task changes credential/privacy behavior at rendering and diagnostic boundaries. ADR-029 already forbids credentials and private values in persistent diagnostics and requires omission at the producer boundary; this plan implements that accepted decision without creating a new ADR.

**Approved design:** `Docs/superpowers/specs/2026-08-02-log-sanitizer-active-redaction-design.md`

---

## File responsibility map

- `tldw_chatbook/Utils/log_sanitizer.py`: public credential-redaction functions, private structured-field classifier, assignment scanner, and standalone credential rules.
- `tldw_chatbook/Utils/sensitive_config_keys.py`: unchanged canonical owner of shipped config-key sensitivity.
- `tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events_ollama.py`: keep structured payload redaction; validate model names for one-line display.
- `tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events_transformers.py`: validate discovered model names for one-line display.
- `tldw_chatbook/Subscriptions/monitoring_engine.py`: retain snapshot-pruning metadata while omitting the URL.
- `Tests/Utils/test_log_sanitizer.py`: dedicated sanitizer public-contract, structured-classification, scanner, false-positive, and idempotence tests.
- `Tests/Utils/test_security_enhancements.py`: retain only path-validation tests after moving the unrelated sanitizer class.
- `Tests/Utils/test_sensitive_config_keys.py`: unchanged canonical predicate tests; run as a regression owner.
- `Tests/ProductionApp/test_llm_destination_actions.py`: direct production helper and mounted-production-app model display/payload behavior.
- `Tests/Subscriptions/test_watchlist_snapshot_pruning.py`: real `URLMonitor._store_snapshot()` diagnostic omission proof.
- `Tests/Packaging/test_installed_distribution.py`: extend the existing isolated-wheel probe; do not add another wheel builder.
- `Docs/security/production-diagnostic-inventory.json`: first reconcile only
  the reviewed latest-dev owner/summary drift in a separate baseline commit;
  later update only the reviewed `monitoring_engine.py` digest relative to that
  commit.
- `backlog/tasks/task-856 - Decide-the-fate-of-Utils-log_sanitizer.py-wire-it-in-fixed-or-delete-it.md`: implementation evidence, checked acceptance criteria, ADR link, and closeout notes.

No reduced, fake, simplified, or test-only application may be created. Use direct functions where that is the sharper boundary and the real `TldwCli` tests already present in `Tests/ProductionApp` where mounted behavior matters.

---

### Task 1: Rebase, reconcile inventory, and implement structured-field classification

**Files:**

- Create: `Tests/Utils/test_log_sanitizer.py`
- Modify: `Tests/Utils/test_security_enhancements.py`
- Modify: `tldw_chatbook/Utils/log_sanitizer.py`
- Modify: `Docs/security/production-diagnostic-inventory.json` (reviewed
  latest-dev baseline reconciliation only)
- Reference: `tldw_chatbook/Utils/sensitive_config_keys.py`
- Reference: `backlog/decisions/029-local-private-data-boundary.md`

- [x] **Step 1: Refresh and verify the implementation base before code**

Run:

```bash
git fetch origin dev
git rebase origin/dev
git status --short --branch
git diff --stat origin/dev...HEAD -- \
  tldw_chatbook/Utils/log_sanitizer.py \
  tldw_chatbook/Utils/sensitive_config_keys.py \
  tldw_chatbook/Utils/input_validation.py \
  tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events_ollama.py \
  tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events_transformers.py \
  tldw_chatbook/Subscriptions/monitoring_engine.py \
  Tests/Utils/test_security_enhancements.py \
  Tests/Utils/test_sensitive_config_keys.py \
  Tests/ProductionApp/test_llm_destination_actions.py \
  Tests/Subscriptions/test_watchlist_snapshot_pruning.py \
  Tests/Packaging/test_installed_distribution.py \
  Docs/security/production-diagnostic-inventory.json
```

Expected: the worktree is clean except for committed task/spec/plan documents. If latest `dev` changed any listed file, stop and reconcile the spec and plan before writing tests.

- [x] **Step 2: Re-run the focused behavioral baseline**

Run:

```bash
../../.venv/bin/python -m pytest \
  Tests/Utils/test_security_enhancements.py \
  Tests/Utils/test_sensitive_config_keys.py \
  Tests/ProductionApp/test_llm_destination_actions.py::test_ollama_success_payloads_are_bounded_and_redacted \
  -q
```

Expected: 26 tests pass. If the latest-dev count changes, record the new green count; do not proceed through a failure.

- [x] **Step 3: Reconcile the reviewed latest-dev inventory baseline before production edits**

Run the checker before editing the manifest:

```bash
../../.venv/bin/python scripts/check_persistent_diagnostic_inventory.py
```

At latest-dev commit `ceede62fe46d7aa090df4a36307077e097d8c044`, expected:
exit 1 because subsequent production changes updated diagnostic owners without
committing the generated inventory. Compare committed and generated inventories
in memory; do not run `--write`. The reviewed drift is exactly:

- summary `466/1144/6851/6` becomes `467/1151/6854/6` for
  owner files/TASK-492 calls/TASK-494 calls/sink files;
- add `UI/Screens/provider_model_resolution.py` with one TASK-494 call;
- update the twenty existing owner entries listed below; and
- leave persistent-sink topology and `monitoring_engine.py` unchanged.

At that pinned commit the exact generated owner values are:

| Path suffix | Calls | Digest |
| --- | ---: | --- |
| `Chat/console_chat_controller.py` | 26 | `9c82b5bacfb585cb2344` |
| `Chat/rag_scope.py` | 4 | `c8311dec8573efc15c1a` |
| `DB/Subscriptions_DB.py` | 4 | `aeadec9e82211a3c903c` |
| `Event_Handlers/Chat_Events/chat_rag_events.py` | 50 | `52dfbbbb9df89723c853` |
| `Subscriptions/local_watchlists_service.py` | 3 | `db76c859421cc0197208` |
| `Subscriptions/watchlist_scope_service.py` | 2 | `8dc49325dd2b17b0a7f5` |
| `Tools/web_tool_impls.py` | 9 | `5130bcf118362c078f60` |
| `UI/Console_Modules/dictation.py` | 10 | `44188d4d4174a8174a0a` |
| `UI/Console_Modules/message.py` | 13 | `ee1a310018bae09826e6` |
| `UI/Navigation/main_navigation.py` | 1 | `5041017946de31086ddb` |
| `UI/Screens/chat_screen.py` | 142 | `3547979b9158631d0066` |
| `UI/Screens/provider_model_resolution.py` | 1 | `e5ceaaaff067fd372196` |
| `UI/Screens/settings_screen.py` | 29 | `877b708b259829710f8a` |
| `UI/Screens/watchlists_collections_screen.py` | 77 | `79ef1df299f4a6131333` |
| `UI/Watchlists_Modules/watchlists_backend_controller.py` | 1 | `f137bb3f8055907a77b7` |
| `UI/Wizards/FirstRunSetupWizard.py` | 21 | `79833587abfc0d80e29b` |
| `Widgets/Console/console_context_modal.py` | 3 | `f20db7c68f2499dcc7a2` |
| `Widgets/Console/console_transcript.py` | 7 | `c54d9ebc99f1d6f1baec` |
| `Widgets/splash_screen.py` | 19 | `08e9790faf88858630ea` |
| `app.py` | 293 | `8cee8c672b033ec6b723` |
| `config.py` | 103 | `5026476db34a50876706` |

Review the semantic diagnostic-call delta, not only digests. The accepted delta
must still be: seven metadata-only `Tools/web_tool_impls.py` debug calls for
robots.txt redirect and parsing failures; one provider/model-only catalog debug
call in `provider_model_resolution.py`; one constant-message watchlist-star
debug call with exception context; one reduced-motion card-name info call; and
otherwise only line movement/formatting. Persistent-sink topology is unchanged,
and the existing admission tests remain the proof that these ordinary records
do not reach disk. The `app.py` sink line numbers move from 6063/6091/6148 to
6064/6092/6149, and the `config.py` private append sink moves from line 4339 to
4352, without semantic sink changes. If latest `dev` differs, stop and
reconcile the new delta before patching anything.

Use `apply_patch` to change exactly the reviewed summary and owner entries.
Then run:

```bash
../../.venv/bin/python scripts/check_persistent_diagnostic_inventory.py
../../.venv/bin/python -m pytest \
  Tests/Architecture/test_persistent_diagnostic_inventory.py \
  -q
```

Expected: checker exit 0 with 467 owners, 1,151 TASK-492 calls, 6,854
TASK-494 calls, and six sink files; all three architecture tests pass.

Commit this prerequisite separately before any sanitizer production edit and
record the commit ID as the TASK-856 inventory reconciliation boundary:

```bash
git add Docs/security/production-diagnostic-inventory.json
git commit -m "chore(security): reconcile diagnostic inventory baseline"
```

- [x] **Step 4: Capture the reconciled diagnostic-inventory no-regression fingerprint**

Run this read-only command before production edits:

```bash
../../.venv/bin/python -c 'import hashlib, importlib.util, json; spec=importlib.util.spec_from_file_location("inventory_check", "scripts/check_persistent_diagnostic_inventory.py"); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); inventory=module.build_inventory(); path="tldw_chatbook/Subscriptions/monitoring_engine.py"; monitoring=next(entry for entry in inventory["owners"] if entry["path"] == path); inventory["owners"]=[entry for entry in inventory["owners"] if entry["path"] != path]; print(json.dumps(monitoring, sort_keys=True)); print(hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest())'
```

Expected monitoring entry on the currently approved base:

```json
{"call_count": 16, "diagnostic_digest": "f9ccee6989b39da1333b", "owner": "TASK-494", "path": "tldw_chatbook/Subscriptions/monitoring_engine.py", "reason": "remaining Chatbook production diagnostic owner"}
```

Expected non-monitoring SHA-256 after the reviewed
`ceede62fe46d7aa090df4a36307077e097d8c044` reconciliation:

```text
8fbd4266f14a51b9645626ba6f5ea624b00db65ac0baae0e4b98de1eaabc0fab
```

Record the actual output in the plan execution notes. If latest `dev` changes
either value, reconcile it before proceeding rather than forcing the old
baseline.

- [x] **Step 5: Move the existing sanitizer tests to their dedicated owner**

Create `Tests/Utils/test_log_sanitizer.py` with the existing `TestLogSanitizer` imports and test methods from `Tests/Utils/test_security_enhancements.py`. Remove only that class and its sanitizer imports from `test_security_enhancements.py`; keep every path-validation test unchanged.

Run:

```bash
../../.venv/bin/python -m pytest \
  Tests/Utils/test_security_enhancements.py \
  Tests/Utils/test_log_sanitizer.py \
  -q
```

Expected: all relocated baseline tests pass before new assertions are added.

- [x] **Step 6: Write failing structured-redaction tests**

Add imports for `tomllib`, `CONFIG_TOML_CONTENT`, `DEFAULT_APP_TTS_CONFIG`, and `is_sensitive_config_key`. Add a local recursive leaf-key iterator rather than importing a helper from another test module.

Add these tests:

```python
def test_real_shipped_sensitive_key_names_are_redacted() -> None:
    default_config = tomllib.loads(CONFIG_TOML_CONTENT)
    key_names = {
        str(key)
        for key in _iter_leaf_key_names(default_config)
        if is_sensitive_config_key(key)
    }
    key_names.update(
        key
        for key in DEFAULT_APP_TTS_CONFIG
        if is_sensitive_config_key(key)
    )
    assert {"api_key", "auth_token", "api_token"} <= key_names

    sentinels = {key: f"PRIVATE_CONFIG_{index}" for index, key in enumerate(sorted(key_names))}
    result = sanitize_dict(sentinels)

    assert set(result) == set(sentinels)
    assert all(value == "***REDACTED***" for value in result.values())


@pytest.mark.parametrize(
    "key",
    [
        "Authorization",
        "Proxy-Authorization",
        "cookie",
        "Set-Cookie",
        "credential",
        "credentials",
        "database_url",
        "connection-string",
        "dsn",
    ],
)
def test_log_protocol_fields_are_redacted_without_expanding_config_policy(key: str) -> None:
    assert not is_sensitive_config_key(key)
    assert sanitize_dict({key: "PRIVATE_PROTOCOL_VALUE"})[key] == "***REDACTED***"
```

Also add direct tests proving:

- `sanitize_dict({1: "safe", "x-api-key": "PRIVATE"})` does not raise and redacts only the sensitive value;
- the input dictionary/list and nested containers are not mutated;
- `deep=False` returns a new outer container, preserves nested dictionary/list identity, and still sanitizes a direct string value;
- `api_key_env_var`, `max_tokens`, and an ordinary key remain unchanged; and
- a sensitive key whose value is a dictionary or list replaces the entire container with the marker before recursion.

- [x] **Step 7: Run the structured tests and verify RED**

Run:

```bash
../../.venv/bin/python -m pytest Tests/Utils/test_log_sanitizer.py -k 'real_shipped or protocol_fields or non_string or deep_false or non_mutating or sensitive_container' -q
```

Expected: failures show missing real config/log fields and the current non-string-key `.lower()` exception. Ensure each new test reaches the intended assertion rather than failing during setup.

- [x] **Step 8: Implement the exact structured classifier**

In `tldw_chatbook/Utils/log_sanitizer.py`:

```python
from tldw_chatbook.Utils.sensitive_config_keys import is_sensitive_config_key

REDACTION_MARKER = "***REDACTED***"
_LOG_ONLY_SENSITIVE_FIELDS = frozenset(
    {
        "authorization",
        "proxy_authorization",
        "cookie",
        "set_cookie",
        "credential",
        "credentials",
        "database_url",
        "connection_string",
        "dsn",
    }
)


def _is_sensitive_log_key(key: object) -> bool:
    normalized = str(key).strip().lower().replace("-", "_")
    return is_sensitive_config_key(key) or normalized in _LOG_ONLY_SENSITIVE_FIELDS
```

Delete the drifting `SENSITIVE_FIELDS` set. Change `sanitize_dict()` to call `_is_sensitive_log_key(key)` and use `REDACTION_MARKER`. Preserve the existing type fallback, `deep` branching order, direct-string sanitization, and list recursion exactly. Do not broaden the public functions to tuples, arbitrary mappings, or new container types.

- [x] **Step 9: Run structured and canonical-predicate tests GREEN**

Run:

```bash
../../.venv/bin/python -m pytest \
  Tests/Utils/test_log_sanitizer.py \
  Tests/Utils/test_sensitive_config_keys.py \
  Tests/Utils/test_security_enhancements.py \
  -q
```

Expected: all structured and relocated baseline tests pass; string tests that have not yet been added are not part of this checkpoint.

- [x] **Step 10: Commit the structured classifier**

```bash
git add \
  Tests/Utils/test_log_sanitizer.py \
  Tests/Utils/test_security_enhancements.py \
  tldw_chatbook/Utils/log_sanitizer.py
git commit -m "fix(security): centralize structured credential fields"
```

---

### Task 2: Implement the classify-first string scanner and installed-wheel proof

**Files:**

- Modify: `Tests/Utils/test_log_sanitizer.py`
- Modify: `Tests/Packaging/test_installed_distribution.py`
- Modify: `tldw_chatbook/Utils/log_sanitizer.py`

- [x] **Step 1: Add failing assignment-scanner and standalone-rule tests**

First update the relocated baseline assertions to the approved neutral-marker
contract before adding the new RED cases:

- `Bearer sk-...` expects `Bearer ***REDACTED***`, not
  `Bearer ***OPENAI_KEY***`;
- standalone `sk-...` in `create_safe_log_message()` expects
  `***REDACTED***`, not `***OPENAI_KEY***`; and
- URL userinfo expects `https://***REDACTED***@example.com`, not the legacy
  `https://***:***@example.com` split marker.

Replace any other provider-specific marker expectation carried over from
`TestLogSanitizer` with `***REDACTED***`. These assertion changes are part of
the approved contract and must happen before the RED run, so a later GREEN run
cannot be blocked by stale expectations from the implementation being removed.

Add parameterized and focused cases with conspicuous fake sentinels:

```python
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('api_key="PRIVATE_QUOTED", safe="visible"', 'api_key="***REDACTED***", safe="visible"'),
        ("password=correct horse battery staple", "password=***REDACTED***"),
        ("max_tokens=42 api_key=PRIVATE_LATER", "max_tokens=42 api_key=***REDACTED***"),
        ("api_key=PRIVATE_QUERY&safe=visible", "api_key=***REDACTED***"),
        ("api_key=\nrefresh_token=PRIVATE_NEXT", "api_key=\nrefresh_token=***REDACTED***"),
    ],
)
def test_assignment_scanner_contract(raw: str, expected: str) -> None:
    assert sanitize_string(raw) == expected
```

Add separate tests proving:

- two quoted secrets on one line are both redacted while the safe field remains;
- an escaped quote does not close a quoted secret;
- an unterminated quoted value is redacted to the line boundary and scanning resumes on the next line;
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and every config-derived sensitive provider label redact both quoted and unquoted opaque sentinels;
- `api_key_env_var`, `max_tokens`, `claude-opus-4-20250514`, ordinary `Basic model configuration`, `NotBearer token`, and `not-bearer token` remain unchanged;
- standalone `Bearer PRIVATE_BEARER` retains `Bearer` but removes the complete credential;
- `https://user:PRIVATE_PASSWORD@example.test/private` contains neither username nor password afterward;
- full, case-sensitive fake `sk-proj-`, `sk-ant-api03-`, legacy `sk-`, and `AIza` values become the neutral marker with no surviving prefix/suffix;
- uppercase variants of those credential prefixes are not newly recognized as standalone shapes;
- a standalone-shaped key inside a quoted labeled value produces one marker and remains idempotent;
- already-redacted strings, dictionaries, and lists are idempotent;
- non-string input retains the current `str()` fallback;
- `create_safe_log_message()` formatting failure returns the sanitized template and never interpolates raw arguments; and
- `safe_log()` invokes its supplied callback exactly once with only the fully
  sanitized final message, for example by collecting calls from
  `safe_log(calls.append, "api_key={}", "PRIVATE_CALLBACK")` and asserting
  `calls == ["api_key=***REDACTED***"]`; and
- a 100,000-character non-matching string completes and remains unchanged without a wall-clock timing assertion; and
- a long line containing many quoted sensitive assignments produces the exact
  redacted output without repeatedly scanning the remainder of the line. Prove
  this with deterministic CR/LF search-work accounting on the real scanner,
  not a wall-clock threshold.

- [x] **Step 2: Extend the existing installed-wheel probe before implementation**

Inside `INSTALLED_PROBE` in `Tests/Packaging/test_installed_distribution.py`, import from the installed target only:

```python
from tldw_chatbook.Utils.log_sanitizer import sanitize_dict, sanitize_string

assert sanitize_string("claude-opus-4-20250514") == "claude-opus-4-20250514"
assert sanitize_dict({"x-api-key": "PRIVATE_INSTALLED_SENTINEL"}) == {
    "x-api-key": "***REDACTED***"
}
assert "PRIVATE_INSTALLED_SENTINEL" not in sanitize_string(
    'x-api-key="PRIVATE_INSTALLED_SENTINEL"'
)
```

Do not add a test application, editable install, second builder, or source-root fallback. The existing probe already excludes checkout/build roots and imports the real `TldwCli` from the installed wheel.

- [x] **Step 3: Run the new source and installed tests RED**

Run:

```bash
../../.venv/bin/python -m pytest Tests/Utils/test_log_sanitizer.py -q
```

Expected: scanner/false-positive tests fail against the old `SENSITIVE_PATTERNS` loop.

Run:

```bash
../../.venv/bin/python -m pytest \
  Tests/Packaging/test_installed_distribution.py::test_installed_wheel_loaders_entry_points_and_assets_are_immutable \
  -q
```

Expected: FAIL in the isolated child because the installed sanitizer rewrites `claude-*` and/or misses a required labeled value. Confirm the failure is behavioral, not a build/network/environment failure.

- [x] **Step 4: Implement the monotonic assignment scanner**

Replace `SENSITIVE_PATTERNS` with private compiled rules. Use a prefix-only assignment pattern so classification happens before value consumption:

```python
_ASSIGNMENT_PREFIX = re.compile(
    r"""
    (?<![A-Za-z0-9_.-])
    (?:
        (?P<quote>["'])(?P<quoted_key>[A-Za-z0-9_.-]+)(?P=quote)
        |
        (?P<plain_key>[A-Za-z0-9_.-]+)
    )
    [ \t]*[:=][ \t]*
    """,
    re.IGNORECASE | re.VERBOSE,
)
```

Implement private helpers with these exact responsibilities:

```python
def _line_end(text: str, start: int) -> int:
    """Return the first CR/LF index at or after start, or len(text)."""


def _after_line_break(text: str, index: int) -> int:
    """Advance over LF, CR, or one CRLF pair; end stays at len(text)."""


def _find_quoted_end(text: str, value_start: int, quote: str) -> tuple[int, bool]:
    """Return the closing-quote/line-end index and whether it closed."""


def _apply_replacements(text: str, spans: list[tuple[int, int]]) -> str:
    """Build one output from sorted non-overlapping spans and REDACTION_MARKER."""


def _redact_assignments(text: str) -> str:
    """Classify label prefixes first, collect replacement spans, and always advance."""
```

`_redact_assignments()` must:

1. search from a monotonic cursor;
2. resume at `match.end()` for non-sensitive labels;
3. skip horizontal whitespace already consumed by the prefix pattern;
4. leave an empty assignment unchanged and resume after its line break;
5. replace only contents between a matching quoted pair, respecting backslash escapes, then resume after the closing quote;
6. replace an unterminated quote from after the opening quote to the line boundary;
7. replace an unquoted sensitive value through the line boundary; and
8. resume after CR/LF/CRLF and continue until no candidate remains.

After a sensitive prefix, inspect the immediate value character for EOF or a
CR/LF boundary before any line search. Parse quoted values directly with
`_find_quoted_end()`; call `_line_end()` only for unquoted values. Dense quoted
assignments must therefore remain single-pass rather than rescanning the line
suffix for every value.

Do not use repeated whole-string concatenation in the scan loop. Collect spans and build once.

- [x] **Step 5: Implement standalone passes in the specified order**

Add:

```python
_URL_USERINFO = re.compile(r"(https?://)[^/?#\s\r\n]*@", re.IGNORECASE)
_BEARER = re.compile(
    r"(?<![A-Za-z0-9_-])(Bearer\s+)(\S+)",
    re.IGNORECASE,
)
_STANDALONE_CREDENTIALS = (
    re.compile(r"(?<![A-Za-z0-9_-])sk-proj-[A-Za-z0-9_-]{20,}(?![A-Za-z0-9_-])"),
    re.compile(r"(?<![A-Za-z0-9_-])sk-ant-api03-[A-Za-z0-9_-]{20,}(?![A-Za-z0-9_-])"),
    re.compile(r"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9]{20,}(?![A-Za-z0-9_-])"),
    re.compile(r"(?<![A-Za-z0-9_-])AIza[A-Za-z0-9_-]{35}(?![A-Za-z0-9_-])"),
)
```

`sanitize_string()` retains its non-string `str()` fallback, then applies:

1. `_redact_assignments()`;
2. URL-userinfo replacement preserving only scheme, marker, `@`, and authority;
3. Bearer replacement preserving the scheme and replacing its credential; and
4. each case-sensitive standalone family with `REDACTION_MARKER`.

No pattern recognizes `claude-*`. Do not add standalone Basic matching or speculative opaque-provider formats.

- [x] **Step 6: Run source tests GREEN**

Run:

```bash
../../.venv/bin/python -m pytest \
  Tests/Utils/test_log_sanitizer.py \
  Tests/Utils/test_sensitive_config_keys.py \
  Tests/Utils/test_security_enhancements.py \
  -q
```

Expected: all pass. Inspect exact outputs, not only absence assertions, for quoted/unquoted syntax and false-positive cases.

- [x] **Step 7: Run the installed-wheel proof GREEN**

Run:

```bash
../../.venv/bin/python -m pytest \
  Tests/Packaging/test_installed_distribution.py::test_installed_wheel_loaders_entry_points_and_assets_are_immutable \
  -q
```

Expected: PASS from the isolated installed target, with source/build roots excluded and the target hash unchanged.

- [x] **Step 8: Commit the scanner and wheel proof**

```bash
git add \
  Tests/Utils/test_log_sanitizer.py \
  Tests/Packaging/test_installed_distribution.py \
  tldw_chatbook/Utils/log_sanitizer.py
git commit -m "fix(security): redact labeled credentials without model false positives"
```

---

### Task 3: Correct production consumer boundaries and inventory ownership

**Files:**

- Modify: `Tests/ProductionApp/test_llm_destination_actions.py`
- Modify: `Tests/Subscriptions/test_watchlist_snapshot_pruning.py`
- Modify: `tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events_ollama.py`
- Modify: `tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events_transformers.py`
- Modify: `tldw_chatbook/Subscriptions/monitoring_engine.py`
- Modify: `Docs/security/production-diagnostic-inventory.json`

- [x] **Step 1: Write failing direct production-helper tests**

In `test_ollama_success_payloads_are_bounded_and_redacted`, strengthen the payload to include nested `x-api-key`/authorization fields and replace the credential-like model-name assertion with:

```python
names = ollama_events._safe_ollama_model_names(
    [
        {"name": "claude-opus-4-20250514"},
        {"name": "org/model\r\n\tinjected"},
    ]
)
assert names == ["claude-opus-4-20250514", "org/model injected"]
```

Add a direct Transformers filesystem test using `tmp_path`:

```python
def test_transformers_model_scan_preserves_claude_ids_as_one_line(tmp_path: Path) -> None:
    model_root = tmp_path / "models--anthropic--claude-opus-4-20250514"
    model_root.mkdir()
    (model_root / "config.json").write_text("{}", encoding="utf-8")
    (model_root / "model.safetensors").touch()

    assert transformers_events.scan_transformers_local_models(tmp_path) == [
        "anthropic/claude-opus-4-20250514"
    ]
```

Add a second direct Transformers filesystem test for single-line normalization.
Because CR/LF/tab characters in a directory name are not portable to Windows,
skip only this test there with `@pytest.mark.skipif(os.name == "nt", ...)`:

```python
@pytest.mark.skipif(os.name == "nt", reason="Windows filenames reject CR/LF/tab")
def test_transformers_model_scan_normalizes_multiline_names(tmp_path: Path) -> None:
    model_root = tmp_path / "models--org--line\r\n\tname"
    model_root.mkdir()
    (model_root / "config.json").write_text("{}", encoding="utf-8")
    (model_root / "model.safetensors").touch()

    assert transformers_events.scan_transformers_local_models(tmp_path) == [
        "org/line name"
    ]
```

Use direct production functions; do not construct a test app.

- [x] **Step 2: Write the failing real snapshot-pruning diagnostic test**

In `Tests/Subscriptions/test_watchlist_snapshot_pruning.py`, use the existing real DB/source helpers and `URLMonitor._store_snapshot()`:

While touching this file, remove its pre-existing unused `_serve` import. A
latest-dev scoped Ruff baseline reports that single `F401`; `_serve` has no
references in the module, so removal is mechanical test hygiene rather than a
behavioral change.

```python
@pytest.mark.asyncio
async def test_prune_diagnostic_omits_the_monitored_url(monkeypatch) -> None:
    from types import SimpleNamespace
    from tldw_chatbook.Subscriptions import monitoring_engine

    db, _service, source_id = await _site_source(monkeypatch, [_page("seed")])
    private_url = (
        "https://PRIVATE_USER:PRIVATE_PASSWORD@example.test/PRIVATE_PATH"
        "?api_key=PRIVATE_QUERY#PRIVATE_FRAGMENT"
    )
    captured: list[str] = []

    def capture(message: str, *args) -> None:
        captured.append(message.format(*args))

    monkeypatch.setattr(
        monitoring_engine,
        "logger",
        SimpleNamespace(debug=capture),
    )
    monitor = monitoring_engine.URLMonitor(db)
    for index in range(_cap() + 1):
        await monitor._store_snapshot(
            source_id,
            private_url,
            {"text": f"revision {index}", "html": "", "headers": {}},
            fingerprint="fp",
        )

    message = next(text for text in captured if text.startswith("Pruned "))
    assert message == (
        f"Pruned 1 snapshot(s) for subscription {source_id}, "
        f"keeping the newest {_cap()}"
    )
    for forbidden in (
        private_url,
        "PRIVATE_USER",
        "PRIVATE_PASSWORD",
        "PRIVATE_PATH",
        "PRIVATE_QUERY",
        "PRIVATE_FRAGMENT",
    ):
        assert forbidden not in message
```

If the real logger is used by another `_store_snapshot()` branch during this test, give the replacement object only the methods actually observed; do not replace `URLMonitor` or the database with a simplified test implementation.

- [x] **Step 3: Run consumer tests RED**

Run:

```bash
../../.venv/bin/python -m pytest \
  Tests/ProductionApp/test_llm_destination_actions.py::test_ollama_success_payloads_are_bounded_and_redacted \
  Tests/ProductionApp/test_llm_destination_actions.py::test_transformers_model_scan_preserves_claude_ids_as_one_line \
  Tests/ProductionApp/test_llm_destination_actions.py::test_transformers_model_scan_normalizes_multiline_names \
  Tests/Subscriptions/test_watchlist_snapshot_pruning.py::test_prune_diagnostic_omits_the_monitored_url \
  -q
```

Expected: Ollama/Transformers fail because model names still use credential
redaction or retain CR/LF/tab whitespace; the subscription test fails because
the full sanitized URL is still present. On Windows only the explicitly
non-portable Transformers filename case is skipped.

- [x] **Step 4: Split Ollama and Transformers display validation from redaction**

In Ollama:

```python
from tldw_chatbook.Utils import input_validation as input_safety, log_sanitizer

# In _format_ollama_success_payload():
log_sanitizer.sanitize_dict(data)

# In _safe_ollama_model_names():
safe_name = " ".join(input_safety.sanitize_string(name, max_length=256).split())
```

In Transformers use one line for the import and one for the replacement:

```python
from tldw_chatbook.Utils.input_validation import sanitize_string as sanitize_input

display_name = " ".join(sanitize_input(display_name, max_length=256).split())
```

These line-count-neutral edits are deliberate: the inventory digest includes
diagnostic line numbers, and both modules contain logger calls after the edited
helpers. Preserve the existing logger line numbers so a presentation-only
change cannot manufacture unrelated owner-digest drift. Preserve the current
invalid/empty checks and result caps. Do not route displayed names back through
`log_sanitizer`.

Before changing `monitoring_engine.py`, re-run the fingerprint command from
Task 1, Step 4. Expected: both the non-monitoring SHA-256 and the monitoring
entry still exactly match the reconciled baseline. If they do not, fix the
line-count/scope error before proceeding.

- [x] **Step 5: Omit the subscription URL at the producer boundary**

Delete the dynamic log-sanitizer import and its stale Qodo comment. Change only the message and arguments:

```python
logger.debug(
    "Pruned {} snapshot(s) for subscription {}, keeping the newest {}",
    pruned,
    subscription_id,
    _SNAPSHOTS_KEPT_PER_URL,
)
```

Do not import `_log_origin()` from `Utils.egress`; the origin is unnecessary for this diagnostic.

- [x] **Step 6: Run the direct consumer tests GREEN**

Run the exact command from Step 3.

Expected: four selected tests pass using production helpers and the real
snapshot producer on POSIX. Windows reports three passed and the one explicitly
non-portable filename case skipped.

- [x] **Step 7: Prove TASK-856 changes only the monitoring inventory entry after reconciliation**

Re-run the read-only fingerprint command from Task 1, Step 4.

Expected:

- the non-monitoring SHA-256 exactly matches the recorded base hash;
- `owner`, `reason`, and `call_count: 16` remain unchanged; and
- only `diagnostic_digest` differs for `monitoring_engine.py`.

Patch only that digest in `Docs/security/production-diagnostic-inventory.json`
with `apply_patch`. Do not run the checker's blanket `--write` mode. Use the
recorded reconciliation commit—not `origin/dev`—as the comparison boundary for
TASK-856's inventory delta.

Inspect:

```bash
git diff <reconciliation-commit> -- \
  Docs/security/production-diagnostic-inventory.json
```

Expected: one digest line changes. The whole branch diff against `origin/dev`
also contains the separately reviewed upstream reconciliation and must not be
misreported as a TASK-856 sanitizer change.

- [x] **Step 8: Run the complete diagnostic-inventory gate GREEN**

Run:

```bash
../../.venv/bin/python scripts/check_persistent_diagnostic_inventory.py
```

Expected: exit 0 with the reviewed owner/sink summary.

Run:

```bash
../../.venv/bin/python -m pytest \
  Tests/Architecture/test_persistent_diagnostic_inventory.py \
  -q
```

Expected: all three architecture tests pass.

- [x] **Step 9: Commit the corrected consumers**

```bash
git add \
  Docs/security/production-diagnostic-inventory.json \
  Tests/ProductionApp/test_llm_destination_actions.py \
  Tests/Subscriptions/test_watchlist_snapshot_pruning.py \
  tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events_ollama.py \
  tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events_transformers.py \
  tldw_chatbook/Subscriptions/monitoring_engine.py
git commit -m "fix(privacy): separate display validation from diagnostics"
```

---

### Task 4: Run complete verification, review, and close TASK-856

**Files:**

- Modify: `Docs/superpowers/specs/2026-08-02-log-sanitizer-active-redaction-design.md`
- Modify: `Docs/superpowers/plans/2026-08-02-log-sanitizer-active-redaction.md`
- Modify: `backlog/tasks/task-856 - Decide-the-fate-of-Utils-log_sanitizer.py-wire-it-in-fixed-or-delete-it.md`
- Review: every file changed by Tasks 1–3

#### Latest-dev rebase amendment

After Tasks 1–3 passed their task-scoped reviews, `origin/dev` advanced and the
branch rebased cleanly onto `85a46bea8704d076fd6b544e56bead760fd3e9d9`.
None of TASK-856's scoped production/test files changed upstream, but five STT
executor diagnostics landed without an inventory update. The checked manifest
records `467/1151/6854/6`; generated state is `467/1151/6859/6` for owner
files/TASK-492 calls/TASK-494 calls/sink files.

The reviewed current-dev delta is exactly:

| Path suffix | Calls | Generated digest |
| --- | ---: | --- |
| `Local_Ingestion/audio_processing.py` | 50 | `e161bcc2fa635027a846` |
| `Local_Ingestion/local_file_ingestion.py` | 8 | `23248608ccde923a2339` |
| `Local_Ingestion/video_processing.py` | 80 | `1e6bc554c059b1bbb859` |
| `UI/Screens/library_screen.py` | 83 | `8a3d52f62abfd4600f79` |
| `app.py` | 298 | `e64f633a5515b2ad809f` |

Semantic AST call-multiset comparison shows the first four files have no added
or removed diagnostics. `app.py` adds five metadata/constant-only error calls:
callback marshal failure by callback name; dispatch failure by job ID and error
type; dispatch failure by job ID/provider/error type; and two constant shutdown
messages with exception context. Persistent-sink call shapes/digests are
unchanged; only the three later `app.py` sink lines move to 6635, 6663, and
6720. Existing admission tests remain the proof that these ordinary errors do
not reach persistent diagnostic storage.

Patch only those reviewed owner/summary/sink-line entries without `--write`,
commit them separately, and place that reconciliation immediately before the
Task 3 consumer commit. At that boundary `monitoring_engine.py` still has digest
`f9ccee6989b39da1333b`; the current non-monitoring fingerprint is
`a927b4bc7a229d3c3328a5336054c410aabdedfe5fd40219ab1152a9880763eb`.
Task 3 initially changed only the monitoring digest to
`3826b76482fd484ff194`, preserving the same non-monitoring fingerprint. The
scoped Ruff format repair subsequently changed only that source-sensitive
monitoring digest to the current `911bf9d65817bf259923`; owner, reason,
16-call count, sink topology, and non-monitoring fingerprint remain unchanged.
The final checker must report 467 owners, 1,151 TASK-492 calls, 6,859 TASK-494
calls, and six sink files.

#### Final-dev rebase amendment

After verified closeout, `origin/dev` advanced again and the branch rebased
cleanly onto `b030b0b73f217b955b298a45fce3a0256403447c`. The upstream Console
rail work did not modify any TASK-856 production/test file and did not add or
remove any diagnostic call. It moved existing calls in three owners, so the
generated counts remain `467/1151/6859/6` while these line-sensitive entries
change:

| Path suffix | Calls | Generated digest |
| --- | ---: | --- |
| `UI/Screens/chat_screen.py` | 142 | `35ccde87405a6c405f14` |
| `UI/Screens/settings_screen.py` | 29 | `0c5906f0cf31d049ca2b` |
| `config.py` | 103 | `97bc4277a86ca26bb903` |

AST call-multiset comparison is identical for all three files. Persistent-sink
shapes/digests are unchanged; only the `config.py` private append sink moves
from line 4352 to 4357. Reconcile exactly those entries in a third baseline
commit placed immediately before the Task 3 consumer commit. At that boundary
the monitoring digest remains `f9ccee6989b39da1333b`; TASK-856's head changes
only that digest to `911bf9d65817bf259923`. Both sides share the current
non-monitoring fingerprint
`5ce06a13eb48f8007eddfa92a0616b41e5122b89e6b2b7d494d4c81fb48723ac`.

- [x] **Step 1: Run the complete focused sanitizer/security suite**

```bash
../../.venv/bin/python -m pytest \
  Tests/Utils/test_log_sanitizer.py \
  Tests/Utils/test_security_enhancements.py \
  Tests/Utils/test_sensitive_config_keys.py \
  -q
```

Expected: all pass.

- [x] **Step 2: Run the full affected production-app and subscription modules**

```bash
../../.venv/bin/python -m pytest \
  Tests/ProductionApp/test_llm_destination_actions.py \
  Tests/Subscriptions/test_watchlist_snapshot_pruning.py \
  -q
```

Expected: all pass. These are the real production app/direct producer owners; do not substitute a reduced app.

- [x] **Step 3: Re-run the installed-wheel test from a clean build**

```bash
../../.venv/bin/python -m pytest \
  Tests/Packaging/test_installed_distribution.py::test_installed_wheel_loaders_entry_points_and_assets_are_immutable \
  -q
```

Expected: pass from the isolated installed target with target hashes unchanged.

- [x] **Step 4: Run static and syntax checks on every changed Python file**

```bash
../../.venv/bin/python -m ruff check \
  tldw_chatbook/Utils/log_sanitizer.py \
  tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events_ollama.py \
  tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events_transformers.py \
  tldw_chatbook/Subscriptions/monitoring_engine.py \
  Tests/Utils/test_log_sanitizer.py \
  Tests/Utils/test_security_enhancements.py \
  Tests/ProductionApp/test_llm_destination_actions.py \
  Tests/Subscriptions/test_watchlist_snapshot_pruning.py \
  Tests/Packaging/test_installed_distribution.py

../../.venv/bin/python -m ruff format --check \
  tldw_chatbook/Utils/log_sanitizer.py \
  tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events_ollama.py \
  tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events_transformers.py \
  Tests/Utils/test_log_sanitizer.py \
  Tests/Utils/test_security_enhancements.py \
  Tests/ProductionApp/test_llm_destination_actions.py \
  Tests/Packaging/test_installed_distribution.py

../../.venv/bin/python -m py_compile \
  tldw_chatbook/Utils/log_sanitizer.py \
  tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events_ollama.py \
  tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events_transformers.py \
  tldw_chatbook/Subscriptions/monitoring_engine.py
```

For each edited hunk in the two legacy files with recorded whole-file format
drift, derive the enclosing post-edit logical block from
`git diff --unified=0 origin/dev...HEAD -- <file>`, then run one range check per
edited block (Ruff accepts one range per invocation):

```bash
../../.venv/bin/python -m ruff format --check --range=<start>-<end> \
  tldw_chatbook/Subscriptions/monitoring_engine.py
../../.venv/bin/python -m ruff format --check --range=<start>-<end> \
  Tests/Subscriptions/test_watchlist_snapshot_pruning.py
```

Include the import block and the new test as separate snapshot-test ranges if
they are separate hunks. Record the exact ranges and passing output in the
execution notes.

Expected: lint, every full-file format check, every edited-range format check,
and syntax compilation pass. Do not accept a new formatting failure merely
because a file had unrelated baseline drift.

Recorded latest-dev format baseline: Ruff would reformat
`Tests/Subscriptions/test_watchlist_snapshot_pruning.py` and
`tldw_chatbook/Subscriptions/monitoring_engine.py` before TASK-856. Do not
mass-format either legacy file in this task; scoped lint must be green after the
unused import is removed. Every currently formatted changed Python file must
pass a full-file format check, and every edited block in the two legacy files
must pass an explicit range check.

- [x] **Step 5: Run hygiene and scope review**

```bash
git diff --check
git status --short
git diff --stat origin/dev...HEAD
git diff origin/dev...HEAD -- \
  tldw_chatbook/Utils/log_sanitizer.py \
  tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events_ollama.py \
  tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events_transformers.py \
  tldw_chatbook/Subscriptions/monitoring_engine.py
```

Review every changed line for:

- no secret fragment survives a marker;
- no `claude-*` or ordinary Basic prose false positive;
- scanner cursor always advances;
- no global logging hook or new config dependency;
- no URL value remains in the pruning diagnostic;
- no simplified/test-only app exists; and
- no inventory entry beyond the two separately reviewed latest-dev
  reconciliations and the TASK-856 monitoring digest is included, and no user
  change is disturbed.

- [x] **Step 6: Request independent code review and address verified findings**

Use `superpowers:requesting-code-review` with TASK-856, ADR-029, the approved spec/plan, the exact base/head inventory evidence, and all verification output. Apply `superpowers:receiving-code-review` before accepting any suggestion. Re-run affected tests after each correction and repeat review until no Critical or Important issue remains.

- [x] **Step 7: Complete Backlog and design/plan documentation**

Update TASK-856 only after code and reviews are complete:

- check all seven acceptance criteria;
- add concise `## Implementation Notes` covering the reviewed baseline
  reconciliation, classifier/scanner, consumer boundary changes,
  installed-wheel proof, subsequent one-digest TASK-856 inventory update,
  changed files, and ADR-029;
- record every verification count, both latest-dev inventory mismatches, both
  reconciliation commits, and the current reconciled base/head fingerprints;
- change the design status to `Implemented and verified`;
- check completed plan steps and document any deviation; and
- run `backlog task edit 856 -s Done` only after all other Definition-of-Done items are satisfied.

Run:

```bash
backlog task 856 --plain
```

Expected: status Done, all acceptance criteria checked, Implementation Plan and Implementation Notes present, and ADR-029 linked.

- [x] **Step 8: Commit closeout documentation**

```bash
git add \
  Docs/superpowers/specs/2026-08-02-log-sanitizer-active-redaction-design.md \
  Docs/superpowers/plans/2026-08-02-log-sanitizer-active-redaction.md \
  'backlog/tasks/task-856 - Decide-the-fate-of-Utils-log_sanitizer.py-wire-it-in-fixed-or-delete-it.md'
git commit -m "docs(security): close TASK-856 sanitizer repair"
```

- [x] **Step 9: Run final verification after the closeout commit**

Re-run the complete focused commands from Steps 1–5 plus the installed-wheel test. Confirm the worktree is clean, every implementation commit is based on the current `origin/dev`, and every required gate is green.

## Execution and deviations

- The first latest-dev diagnostic mismatch was reviewed at
  `ceede62fe46d7aa090df4a36307077e097d8c044`: checked
  `466/1144/6851/6` versus generated `467/1151/6854/6`. Commit
  rebased commit `2299da555` records only that reviewed pre-existing inventory
  reconciliation.
- After Tasks 1–3 reviewed cleanly, `origin/dev` advanced to
  `85a46bea8704d076fd6b544e56bead760fd3e9d9`. Scoped TASK-856 files were
  unchanged, but five upstream `app.py` diagnostics and related line movement
  changed the generated repository-wide manifest from checked
  `467/1151/6854/6` to `467/1151/6859/6`. Rebased commit `a25f5c792` records the
  separately reviewed rebase boundary immediately before the Task 3 consumer
  commit.
- After verified closeout, `origin/dev` advanced again to
  `b030b0b73f217b955b298a45fce3a0256403447c`. No TASK-856 file or diagnostic
  call changed, but Console rail line movement changed three owner digests and
  the `config.py` sink line. Rebased commit `2862505e7` records that third
  reviewed boundary immediately before the Task 3 consumer commit.
- At `2862505e7`, the monitoring owner is TASK-494 with 16 calls, digest
  `f9ccee6989b39da1333b`, and non-monitoring SHA-256
  `5ce06a13eb48f8007eddfa92a0616b41e5122b89e6b2b7d494d4c81fb48723ac`.
  Final generated state retains the owner, reason, count, fingerprint, and six
  sinks while changing only that digest to `911bf9d65817bf259923`.
- The first Task 4 verification exposed four branch-introduced Ruff format
  failures. Rebased commit `dce39f0d2` applied deterministic formatting only
  to the approved full files and edited legacy range, then patched only the
  resulting monitoring digest.
- Whole-branch review reported no Critical issues, two related Important
  scanner-complexity issues, and one Minor annotation mismatch. The added dense
  matched-input test deterministically measured 94,996,790 CR/LF-search work
  characters for 46,888 input characters before the fix. Rebased commit
  `1c1686cfa` routes quoted values directly to their quote scanner; scoped
  re-review found both Important issues resolved and no new Critical/Important
  issue. The compatible `sanitize_dict` non-string-key annotation Minor is
  explicitly deferred.
- Final pre-closeout gates had 77 sanitizer/security tests, four selected
  TASK-856 consumers, one isolated installed-wheel test, and three diagnostic
  architecture tests passing. Inventory was `467/1151/6859/6`; Ruff lint,
  full-file format, three legacy ranges, `py_compile`, and diff hygiene were
  green.
- The prescribed full affected-module command is not reported as passing. It
  returned **2 failed, 26 passed** on the branch, with exactly
  `test_llm_destination_action_census_is_complete_and_removed_controls_are_absent`
  and `test_production_llm_destination_owns_navigation_actions_and_recovery`.
  The identical clean-`origin/dev` command returned the same failure set at
  **2 failed, 23 passed**; the branch adds three passing TASK-856 tests.
- Rebased closeout commit `d70bd448f` recorded the verified design/task/lesson
  state. Follow-up commit `15d6d4b0c` checked completed Steps 8 and 9 after the
  post-commit gates; scoped re-review found the marker-only correction clean.
