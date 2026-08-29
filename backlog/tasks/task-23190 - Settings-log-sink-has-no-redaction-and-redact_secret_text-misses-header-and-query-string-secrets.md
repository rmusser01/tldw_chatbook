---
id: TASK-23190
title: >-
  Settings log sink has no redaction and redact_secret_text misses header and
  query-string secrets
status: Done
assignee: []
created_date: '2026-08-29 02:25'
updated_date: '2026-08-29 03:30'
labels:
  - security
  - settings
  - logging
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-23108 sends users to the log for failure detail ('Details are in Logs (F8)') and its user-facing paths were reduced to logging exception TYPE names only, precisely because message text is not safe to write there. The residual gap is the sink: the rotating file handler in Logging_Config.py applies no redaction (only the in-app Logs buffer runs redact_log_line), and redact_secret_text's _SECRET_ASSIGNMENT_PATTERN matches 'X = value' assignments only -- so an 'Authorization: Bearer sk-...' header or a '?key=<token>' URL reaches disk unchanged from any OTHER caller that logs exception text. Standing project rule (loguru diagnose incident) is to fix this at the sink rather than at each call site. Raised by the Qodo review of PR #2170 and by the TASK-23108 review round.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Secrets in header form ('Authorization: Bearer <token>') and query-string form ('?key=<token>', '?api_key=<token>') are redacted before reaching the rotating file sink
- [x] #2 Redaction is applied at the sink so it covers callers that do not opt in, not only paths that call redact_secret_text explicitly
- [x] #3 A test writes each secret shape through the real logging configuration and asserts the on-disk record contains no secret material
- [x] #4 Existing type-name-only call sites are unaffected and no diagnostic gains new interpolation
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Probe what actually reaches the rotating file sink today and what log_sanitizer already covers.
2. Add a redacting Formatter on the private file handler so every line it writes is sanitized (sink-side, no caller opt-in), reconciling an already-installed handler too.
3. Close the second half of the title: teach redact_secret_text the header (Bearer / 'X-Api-Key: v') and query-string ('?key=') shapes it misses, keeping the <redacted> marker stable.
4. On-disk tests through the real _configure_private_file_logging, plus a negative control and the layered no-write assertion.
5. Mutation-check both guards; run sanitizer/logging/settings suites, ruff, preflight.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Redaction is now a property of the private file sink, and `redact_secret_text`
recognises the two shapes it was blind to.

**The premise needed correcting first.** The description says an
`Authorization: Bearer sk-...` header "reaches disk unchanged from any OTHER
caller". Probed against the real configuration: it does not. The rotating
handler already carries `PersistentDiagnosticFilter`, which admits *only*
schema-validated ADR-029 metadata events -- an ordinary
`logger.error("Authorization: Bearer sk-live-abc123")` and a third-party
`httpx` record both produced zero bytes. So the sink had no redaction, but it
also had no unredacted secrets: what was missing was the second layer that
makes a future widening of that admission rule safe rather than a disclosure.
The fix is written and tested as that second layer, and
`test_unmarked_secret_bearing_record_is_not_written_at_all` pins the first one
so a widening becomes a visible change.

**`log_sanitizer` needed no new patterns.** Probed all four shapes through
`sanitize_string`/`redact_log_line` before writing anything: `Bearer` (via
`_BEARER`), `?key=` (via the bare-`key` entry `_LOG_ONLY_SENSITIVE_FIELDS`
gained in TASK-19558), `?api_key=` and `&token=` (via `is_sensitive_config_key`)
were all already covered, and a secret-free line passed through untouched. The
sink simply never called it.

**Approach and trade-offs.**

- `RedactingFileFormatter` wraps `redact_log_line` around the fully formatted
  record, rather than a `logging.Filter`. Two reasons, both load-bearing: a
  filter must mutate `record.msg`, and the record is shared with every other
  handler on the logger (and with whichever thread emitted it); and only the
  formatted string contains the `exc_info` traceback, where a credential inside
  `str(exc)` actually lives. The on-disk test for exception text fails against
  a msg-only redactor.
- Cost: 33.6 us/record vs 1.4 us for the plain formatter. Handler filters run
  before `emit`, so this is paid only on records that are actually written --
  a handful of metadata events per session today.
- `MAX_REDACTED_LINE_CHARS` is kept, not disabled. Truncation keeps strictly
  less data, its cut is token-aligned so it cannot slice a credential into an
  unmatchable fragment, and it is what bounds the per-record cost. Cost of
  keeping it: a >2,000-char record is truncated on disk too.
- The already-installed-handler branch reconciles the formatter for the same
  reason it already reconciles the filter -- otherwise a handler built by an
  earlier revision keeps writing in clear for the rest of the process.
- `redact_secret_text` keeps its `<redacted>` marker and its existing
  name vocabulary (many callers and tests assert the exact `KEY=<redacted>`
  shape). Three narrow additions: `[:=]` as separator (`X-Api-Key: <token>` is
  how every credential header is spelled), a `Bearer` prefix rule, and a bare
  `key` rule anchored to query-string position only. `key` is deliberately not
  added to the name vocabulary -- it is an ordinary English word and a TOML
  parse error says it -- so the `?`/`&` anchor is what proves it is a
  parameter name. Unlike the log sanitizer's redact-to-end-of-line behaviour
  this preserves the rest of the URL (`?key=<redacted>&cx=017`).

**Mutation checks** (all Edit-based restores). Disabling the redaction call in
`RedactingFileFormatter.format`: 6 failed / 19 passed, and the negative control
and admission-layer tests correctly stayed green. Removing the Bearer and
query-key rules: 2 of my parametrized cases failed. Narrowing the separator
back to `=`: the colon-header case failed. All green after restore.

**Verification.** `Tests/test_logging_private_files.py` 25 passed;
logging/diagnostics/sanitizer batch 167 passed; config/websearch/RAG privacy
batch 92 passed; `Tests/UI/test_settings_configuration_hub.py` +
`test_settings_provider_test_draft.py` 449 passed / 6 failed -- the identical
6 fail on the merge-base `4fb5d38d37` (save/revert and category-content
assertions, no redaction assertion among them). `preflight.sh` all green with
no diagnostic-inventory drift; ruff clean.

**Also checked, deliberately unchanged.** Loguru `diagnose` is `False` at every
`logger.add` in the package (`Logging_Config.py`, `__init__.py`,
`startup_logging.py`, `Metrics/logger_config.py`) -- no finding. Loguru's own
file sinks are already disabled in `Metrics/logger_config.py`;
`Chunking/engine/security_logger.py` can open one but only when handed an
explicit `log_file`, which no production caller does.

**Files.** `tldw_chatbook/Logging_Config.py`,
`tldw_chatbook/UI/Screens/settings_config_adapter.py`,
`Tests/test_logging_private_files.py`,
`Tests/UI/test_settings_configuration_hub.py`.
<!-- SECTION:NOTES:END -->
