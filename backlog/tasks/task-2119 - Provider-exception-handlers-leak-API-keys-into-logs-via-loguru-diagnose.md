---
id: TASK-2119
title: Provider exception handlers leak API keys into logs via loguru diagnose
status: Done
assignee: []
created_date: '2026-08-03 18:50'
updated_date: '2026-08-04 02:14'
labels:
  - security
  - llm-calls
  - observability
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
**A real API key was disclosed while verifying this. The Moonshot key in the repo root
was leaked in plaintext by an ordinary HTTP 429 during multi-provider verification on
2026-08-03 and must be rotated.**

`tldw_chatbook/LLM_Calls/LLM_API_Calls.py` has ~30 call sites where a provider request's
`RequestException` / generic exception handler calls
`logger.opt(exception=True).error(...)`. Loguru's `diagnose` option defaults to True and
dumps **every stack frame's local variables** alongside the traceback. In these handlers
the locals in scope include the raw request `headers` (carrying
`Authorization: Bearer <key>` / `x-api-key`) and `final_api_key`. So any transient
provider error during normal chat writes the user's key to the log sink in cleartext.

Confirmed affected by grep: OpenAI, Cohere, Moonshot, Z.AI. Live-confirmed for Moonshot
via a genuine 429. Google's and OpenRouter's specific error branches happen not to use
`opt(exception=True)` — coincidence, not design.

**Pre-existing, not introduced by the cost-ticker program:** `git blame` traces the
sampled sites to PR #707 (2026-07-19) and PR #1235 (2026-08-02).

Note this is a *different* surface from the one PR #1295 closed. That work made the
**debug payload logs** allowlist-shaped; these are **exception handlers**, where the
secret arrives via frame locals rather than via a logged payload dict — so no amount of
payload redaction touches it.

**Preferred fix is the class-killer, not 30 patches.** Frame locals should never reach a
persistent sink: set `diagnose=False` on the app's logger sink configuration, which
eliminates the entire category in one place regardless of which handler runs or what a
future contributor adds. Per-call-site `opt(exception=True, diagnose=False)` is the
fallback if a sink-level change is judged too broad, but it leaves the next new handler
exposed by default.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An exception raised inside a provider call with an API key in scope does NOT write the key value to any log sink (file or console), on both the sensitive and non-sensitive request paths
- [x] #2 The fix is applied at the sink/configuration level so a newly added exception handler is safe by default, not dependent on the author remembering a flag
- [x] #3 A regression test plants a sentinel key value in scope, forces a provider exception, captures log output, and asserts the sentinel appears nowhere
- [x] #4 Traceback/diagnostic value is preserved to the extent possible (the exception type, message, and stack are still logged — only frame-local dumping is suppressed)
- [x] #5 `tldw_chatbook/` swept for other `opt(exception=True)` / `exc_info=True` sites where credentials or full payloads are in scope; each fixed or justified
- [ ] #6 The exposed Moonshot key is rotated by the owner (tracked here for closure; not an agent action)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Fix at the loguru sink rather than the call sites
2. Cover the auto-init default sink that the incident script actually hit
3. Pin with a sentinel regression test plus a positive control
4. Sweep for any remaining sink registration
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed at the sink. All three live `logger.add()` registrations in the package now
pass `diagnose=False` explicitly, with `backtrace=True` kept so exception type,
message, and the stack of source lines still log — only the per-frame local-variable
dump goes away.

**Why the sink and not the call sites:** the task described ~30 handlers in
`LLM_API_Calls.py`, but a sweep found **1,055** `opt(exception=True)` sites across
`tldw_chatbook/`. Per-call-site patching was never viable, and any one missed site
reopens the hole. Three `logger.add()` calls cover all 1,055 by construction
(AC#5). The 160 stdlib `exc_info=True` sites need no change: stdlib's formatter
renders a traceback without frame locals.

**Third sink was the load-bearing one.** Beyond `Logging_Config.py` and
`Metrics/logger_config.py`, `tldw_chatbook/__init__.py` now replaces loguru's
auto-init default sink at package import. That sink is what the 2026-08-03 incident
actually hit: the verification script imported the provider adapters directly and
never called `configure_application_logging`, so the app's own hardened sink was
never installed. `configure_application_logging` calls `logger.remove()` before
adding its own sink, so the package-init stderr sink is discarded in the real app —
no duplicate TUI logging. `LOGURU_DIAGNOSE=0` is set alongside it as a default for
future `add()` calls, but is not relied on: it only binds if set before loguru's
module is first imported, and `Tests/conftest.py` imports loguru first.

**Test-quality note.** The first version of the sink-config pin asserted
`"diagnose=False" in <sliced source>`. Mutation testing showed it was **vacuous** —
the security rationale comment directly above the kwarg contains that same literal
string, so deleting the real argument still passed. Rewritten to parse the call with
`ast` and check actual keyword arguments; re-verified against three mutations
(remove the kwarg from either sink, flip it to `True`), each of which now fails.
The suite also carries a **positive control** asserting the sentinel *does* leak
through a `diagnose=True` sink, so the regression test cannot pass for the wrong
reason, and an out-of-process test reproducing the incident's exact import shape.
The sentinel test deliberately drives a real `ConnectionError` through `requests`'
internals rather than a mocked session — a fully-mocked `.post()` has no
intermediate frames and does not reproduce the leak at all.

**Post-fix gate sweep found a second, unrelated failure:** `Tests/Architecture/
test_persistent_diagnostic_inventory.py` pins a checked-in JSON snapshot
(`Docs/security/production-diagnostic-inventory.json`) of every production
diagnostic call and persistent-sink registration, keyed by source digest. It failed
after this fix — correctly: the `Logging_Config.py` sink's digest changed because
`diagnose=False, backtrace=True` were added to it. The rest of the diff (~200 of
~212 changed lines) was pre-existing drift already on `origin/dev`: the checked-in
snapshot predates PR #1235 (2026-08-02 16:54) by six minutes and had never been
regenerated since, so unrelated files (console_cost_tracker.py,
prompt_improvement_service.py, change_review_screen.py, several UI screen
refactors) were already out of sync before this branch existed. Reviewed the full
diff line by line for anything suspicious (a new `diagnose=True`, a new bare
`FileHandler`, etc.) — none found — then regenerated via the script's own
`--write` flag, the sanctioned path for an explicit, reviewed topology change.

**Modified:** `tldw_chatbook/__init__.py`, `tldw_chatbook/Logging_Config.py`,
`tldw_chatbook/Metrics/logger_config.py`, `Tests/Chat/test_sensitive_llm_logging.py`
(67 passing), `Docs/security/production-diagnostic-inventory.json` (regenerated).

**AC#6 remains open and is an owner action:** the disclosed Moonshot key still needs
rotating. Code changes cannot close that.

**PR #1305 Qodo round (all four findings accepted, mutation-verified):**
(1) package init narrowed from `logger.remove()` (wipes host-app sinks) to a
guarded `remove(0)` that replaces only loguru's auto-init default sink — a host
that configured loguru before importing the package keeps its sinks, pinned by a
new subprocess test that goes red if `remove(0)` reverts to `remove()`;
(2) tests made hermetic — the subprocess child now runs with `LOGURU_*` scrubbed
from its env, and the in-process ambient-env assertion (unfixable in-process:
the package is imported before any test body runs) was dropped in its favor;
(3) the subprocess check no longer reads `logger._core` private internals — it
behaviorally asserts a credential-shaped frame local stays out of the child's
stderr, with a bare-loguru positive-control variant proving the same script
shape DOES leak pre-fix (mutation: deleting the init block fails the test);
(4) `Args:` sections added to the new parameterized tests per file convention.
<!-- SECTION:NOTES:END -->
