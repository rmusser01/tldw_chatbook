---
id: TASK-41714
title: "Loguru calls pass %s format strings in about ten modules, so their values never reach the log"
status: Done
assignee:
- '@claude'
created_date: 2026-11-18 06:57
updated_date: 2026-11-20 10:28
labels:
- follow-up
- logging
priority: low
references:
- tldw_chatbook/app_navigation.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41500 and its review (round 16), widened by the filing review. Loguru does not substitute `%s`, so these calls print a literal `%s` and drop their values, for example: `app_navigation.py` (such as "Reusing installed screen instance ... route: %s"), `app.py` ~1252/4337/4369, `LLM_Calls/LLM_API_Calls.py` ~341 and five more, `Chat/console_runtime.py` ~6922, `UI/Screens/chat_screen.py` ~7607, `Library/local_media_chunk_tool_service.py` ~990, `UI/Settings_Modules/manual_sync.py` ~62, `Chat/prompt_template_manager.py` ~86, `UI/Wizards/first_run_appearance_step.py` ~179, `UI/Screens/scheduling/conflicts_tab.py` ~309 and `Event_Handlers/LLM_Management_Events/llm_management_events.py` ~128. `Tests/test_stdlib_logging_format_style.py` guards the opposite case.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every loguru call in the package that passes arguments uses `{}` placeholders or f-strings and logs its values
- [x] #2 A repo-wide test fails on a loguru call that passes `%s` with arguments, red before the change
<!-- AC:END -->

## Implementation Plan

1. Add a scope-aware AST scan to `Tests/test_stdlib_logging_format_style.py` (the opposite-case guard): it finds calls on a loguru logger (a name imported from loguru, a name assigned from one, or an `*.loguru_logger` attribute; a `logger` rebound by `logging.getLogger` in its function or module is stdlib) whose constant message has a `%` placeholder and that pass arguments. Plant a violation to show the scan can fail, then show the package scan red.
2. Convert every flagged site to `{}` placeholders, keeping the same values (no new fields, no tracebacks).
3. Show one converted site's value reaching a captured loguru sink.

## Implementation Notes

- **Guard.** `Tests/test_stdlib_logging_format_style.py` gained the opposite-case scan, `test_no_loguru_call_uses_percent_style_with_arguments`. It walks each package module scope by scope: a loguru logger is a name imported from loguru, a name assigned from one (`logger = logger.bind(...)`, `logger = getattr(app, "loguru_logger", _loguru_fallback_logger)`), or an `*.loguru_logger` attribute, through `.opt()/.bind()/.patch()`; a `logger` rebound by `logging.getLogger` in its function or module is stdlib. It flags a constant message with a `%` placeholder (and no `{`) when the call passes arguments. Not resolved: other attribute receivers (`module.logger`); `openai_protocol.py`'s `_owner.logger` was found by a wider receiver survey and converted by hand. Files without "loguru" and a `%` placeholder are not parsed (6.2 s serial at load 45).
- **Red/green.** Before the conversion the package scan failed listing 48 calls (24 in `app_navigation.py`; settings_screen's 28 `%s` calls are a stdlib `logger` and were rightly not listed). `test_the_loguru_scan_finds_a_planted_percent_style_call` pins module, nested-stdlib, `opt`, bound-name and `loguru_logger` cases.
- **Conversion.** 49 calls in 13 modules: `%s`/`%d` -> `{}`, same arguments, except three sites whose argument was raw exception text, which now log `type(exc).__name__`, so no new text reaches a persistent sink: `prompt_template_manager.safe_render` (a Jinja render error quotes the template, e.g. `'private_name' is undefined`), `local_media_chunk_tool_service` chunk-spec save (wraps raw store errors), and `openai_protocol` stream error (its traceback, already logged, keeps the detail). All other values are routes, screen class names, exception type names, model ids, counts and flags.
- **Value reaches a sink.** `Tests/DB/test_client_media_debug_logging.py` now asserts `Media search completed (mode=fts, limit=20, offset=0, sort=` and `Media search failed (error_type=OperationalError).` in a captured loguru sink. With `Client_Media_DB_v2.py` reverted to base both failed (the sink held `mode=%s, limit=%s, ...` and `error_type=%s`); with the change both pass, and the file's privacy assertions still hold.
- **Stdlib guard was red at base.** `test_no_stdlib_logger_uses_loguru_brace_style` failed before any change: `Client_Media_DB_v2.py` binds a stdlib `logger` inside one function, and the module-wide regex made every loguru `{}` call in the file an offender (3 rows at base, 8 after the conversion). `_brace_style_calls` now skips calls whose scope's `logger` is loguru; `test_a_loguru_module_with_one_stdlib_function_is_scanned_by_scope` was red without that skip (both lines reported) and green with it. All 7 tests in the file pass.
- **Pins updated.** `Tests/Chat/test_chat_functions.py`'s warning double formatted with `%` (so it hid this bug) and now uses `.format`; `Tests/UI/test_console_session_settings.py` pinned the raw `%s` template. Both nodes pass under a temporary `@private_profile_test` (removed); plain, they fail locally on the known `RecoveryRequired: raw_source_selection_changed` trap. `pytest Tests/Chat/test_chat_functions.py Tests/Library/test_media_chunk_tool_service.py Tests/DB/test_client_media_debug_logging.py <that session-settings node> -n 4` gave the same 26 failures and 230 passes at head and with the touched sources and tests reverted to base (diffed node lists); 25 are that trap (the session-settings node among them) and one is the stale reopen test below.
- **Diagnostic inventory.** 13 owner rows changed (same counts). Reviewed with `--statements --since cc1e0f8020`: every added statement interpolates only the metadata above; then `--write`. `test_reviewed_diagnostic_changes_are_metadata_only` still fails on two rows in files this task does not touch (`Agents/agent_service.py` 'on_child_settled consumer raised', `Chat/console_chat_store.py`, both "found 0"), so it is red at base.
- **Pre-existing, not this task:** `test_client_media_debug_logging.py::test_closed_connection_reopens_without_private_diagnostics` (expects "Media database connection was closed; reopening.", which no longer exists in `tldw_chatbook/DB`; red with base `Client_Media_DB_v2.py` too).
- Files: the 13 modules in the diff, `Docs/security/production-diagnostic-inventory.json`, `Tests/test_stdlib_logging_format_style.py`, `Tests/DB/test_client_media_debug_logging.py`, `Tests/Chat/test_chat_functions.py`, `Tests/UI/test_console_session_settings.py`.
- Checks: `./scripts/preflight.sh` passed (wrapper); `ruff check --select F` on the changed files passed; module, screen and Library size ratchets passed (serial; 2 failures in that run were the diagnostic inventory, since regenerated, and the base-red metadata-only rows above).
- **Review r17lg-rev1 minor 3.** The scan skipped any message containing `{`, so `logger.info("{} rows to %s", n, path)` escaped it. `_percent_message` now flags a `%`-placeholder message whose `str.format` fields are fewer than its positional arguments (`"{}% done", 50` still passes). `test_the_loguru_scan_flags_a_mixed_brace_and_percent_call` was red before (`[] == [(2, '{} rows to %s')]`); the package scan still finds no offender; the file passes 8 of 8.
- **Review r17lg-rev2 minor 1.** The minor-3 change dropped keyword-only calls: `_percent_message` returned early without positional arguments, so `logger.info("x %s", value=3)` and `logger.error("Failed: %s", exc_info=True)` passed. Keywords now count as arguments, as the original rule did. `test_the_loguru_scan_flags_keyword_only_percent_calls` was red before (`assert [] == [(2, 'x %s'), ...]`); the package scan still finds no offender; the file passes 9 of 9.
- **Review r17lg-rev3.** Counting keywords as arguments flagged correct calls, because loguru binds keywords no field names to `extra`: `logger.info("50% of {}", n, user=u)` and `logger.info("{} at 5% rate", n, extra=1)`. The rule is now: flag a `%`-placeholder message when its positional arguments outnumber its `{}` fields, or when it has no field at all and any argument, keywords included. `test_the_loguru_scan_spares_unused_keywords_beside_brace_fields` was red against the previous rule (both calls flagged) and is green; the keyword-only planted case is still flagged; the file passes 10 of 10 and the package scan is clean.
