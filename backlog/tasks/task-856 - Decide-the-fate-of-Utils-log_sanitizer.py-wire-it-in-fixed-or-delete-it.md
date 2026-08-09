---
id: TASK-856
title: 'Decide the fate of Utils/log_sanitizer.py: wire it in fixed, or delete it'
status: Done
assignee:
  - '@codex'
created_date: '2026-07-27 04:35'
updated_date: '2026-08-08 16:29'
labels:
  - security
  - config
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The original audit found no production importers, but that premise is stale on
current `dev`: `Utils/log_sanitizer.py` now has three production consumers.
Ollama uses it both to redact successful API payloads and to transform model
names, Transformers uses it to transform local model display names, and the
subscription monitor uses it before interpolating a URL into a pruning
diagnostic. The module therefore needs an explicit, correct runtime boundary
rather than deletion.

Its rules remain wrong in both directions. The `claude-*` pattern treats an
Anthropic model-ID prefix as a credential, so
`claude-opus-4-20250514` becomes `***ANTHROPIC_KEY***`. Meanwhile real-shaped
`sk-ant-api03-*` and `sk-proj-*` credentials are only partially matched or
survive because the generic `sk-` character class stops at their embedded
hyphens. `sanitize_dict()` also has a second, drifting secret-field list that
misses real shipped names such as `x-api-key`, `cohere_api_key`, `api_token`,
`secret_key`, and `refresh_token`.

Credential redaction and display-name validation are separate operations.
Model names must receive bounded single-line input validation without being
rewritten as secrets. Structured and labeled credential values must use the
shared sensitive-config-key classification plus HTTP/log-specific secret
fields. Ambiguous opaque provider tokens can only be safely recognized from
that context; attempting to identify every opaque token as a standalone
credential would corrupt ordinary identifiers. URL-bearing diagnostics must
omit private URL components instead of treating regex redaction as permission
to log them.

The latest `dev` baseline also contains reviewed diagnostic changes whose
generated inventory was not committed. TASK-856 must reconcile those exact
pre-existing owner entries in a separate baseline commit before changing
sanitizer production code. Its own inventory proof is then measured from that
reconciled commit, so upstream drift cannot be mistaken for or hidden inside
the subscription diagnostic change.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `Utils/log_sanitizer.py` remains importable under its existing public function names and redacts complete credential values without rewriting ordinary model identifiers such as `claude-opus-4-20250514`
- [x] #2 Structured redaction covers the app's real sensitive config key names, sourced in tests from shipped configuration rather than a duplicate sanitizer-owned list, plus authentication headers, cookies, credential containers, and connection-secret fields
- [x] #3 Labeled credentials from every configured provider are redacted even when their opaque value has no safely recognizable standalone format; high-confidence standalone key families are fully redacted rather than partially matched
- [x] #4 Ollama and Transformers model names use bounded single-line input validation, preserve legitimate `claude-*` identifiers, and do not rely on credential redaction for display safety
- [x] #5 Subscription snapshot-pruning diagnostics contain useful non-private metadata but omit the monitored URL and all URL credentials, paths, queries, and fragments
- [x] #6 Recursive, non-mutating dictionary/list behavior, non-string mapping keys, `deep=False`, formatting fallback, and the installed-wheel import path are covered without introducing a reduced or test-only application
- [x] #7 Pre-existing latest-dev diagnostic-inventory drift is reviewed and reconciled in a separate baseline commit, after which TASK-856 changes only `monitoring_engine.py`'s diagnostic digest while preserving every other generated inventory entry and the persistent-sink topology
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Rebase onto current origin/dev, rerun the focused baseline, review the exact pre-existing diagnostic-inventory drift, and reconcile only those entries in a separate baseline commit without using blanket write mode.
2. Capture the reconciled non-monitoring fingerprint, then move sanitizer tests to a dedicated module and use TDD to compose structured redaction from the canonical sensitive-config-key predicate plus exact log/protocol fields.
3. Use TDD to replace regex-first assignment handling with the monotonic classify-first scanner and bounded standalone rules; extend the existing installed-wheel probe.
4. Use direct production functions and the full production app to separate Ollama/Transformers display validation and omit the subscription URL at its diagnostic producer.
5. Prove only monitoring_engine.py’s reviewed diagnostic digest changes relative to the recorded reconciliation commit; do not use a blanket inventory rewrite.
6. Run focused, production-app, subscription, installed-wheel, diagnostic-inventory, lint, format, syntax, hygiene, and independent-review gates; complete task notes and status only after verified closeout.

ADR required: yes
ADR path: backlog/decisions/029-local-private-data-boundary.md
Reason: Implements ADR-029’s credential/privacy diagnostic boundary without introducing a new architectural decision.

Approved design: Docs/superpowers/specs/2026-08-02-log-sanitizer-active-redaction-design.md
Detailed plan: Docs/superpowers/plans/2026-08-02-log-sanitizer-active-redaction.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented and independently reviewed the active sanitizer rather than deleting
it. Structured redaction delegates shipped config-key classification to the
canonical predicate, composes the narrow HTTP/log-only field set, and uses a
classify-first scanner plus bounded standalone credential families. Ollama and
Transformers model names use bounded single-line display validation, while the
snapshot-pruning producer omits the monitored URL entirely.

Tests moved to `Tests/Utils/test_log_sanitizer.py` and cover public compatibility,
recursive/non-mutating containers, contextual and standalone credentials,
false-positive model identifiers, formatting fallback, installed-wheel imports,
and deterministic matched-input scan work. The final review caught a superlinear
quoted-match path: the pre-fix implementation searched 94,996,790 CR/LF characters
for 46,888 input characters. Rebased commit `1c1686cfa` made quoted values bypass
repeated line-end scans; the scoped re-review found no remaining Critical or
Important issue. The compatible `sanitize_dict`
non-string-key annotation mismatch remains an explicitly deferred Minor.

Diagnostic drift was reconciled without checker `--write` at three reviewed
boundaries: rebased commits `2299da555`, `a25f5c792`, and `2862505e7`. The last
boundary follows the final rebase onto
`b030b0b73f217b955b298a45fce3a0256403447c`. At `2862505e7`,
`monitoring_engine.py` was TASK-494 with 16 calls and digest
`f9ccee6989b39da1333b`; the final head changes only that digest to
`911bf9d65817bf259923`. The current non-monitoring SHA-256 remains
`5ce06a13eb48f8007eddfa92a0616b41e5122b89e6b2b7d494d4c81fb48723ac`, with
inventory `467/1151/6859/6` and unchanged sink topology.

ADR-029 is the governing privacy boundary; no new ADR was required. Changed
areas are the sanitizer and its focused/packaging tests, Ollama/Transformers
display helpers and production tests, subscription diagnostic/test, reviewed
inventory, approved design/plan, task closeout, and the two generalized lesson
entries.

Verification before closeout: 77 sanitizer/security tests, four selected
TASK-856 consumers, one installed-wheel test, and three inventory architecture
tests passed; Ruff lint, full-file format, three legacy ranges, `py_compile`,
inventory checker, and diff hygiene were green. The full affected-module command
is not claimed green: branch result was 2 failed/26 passed, exactly the same
failure set as clean `origin/dev` at 2 failed/23 passed
(`test_llm_destination_action_census_is_complete_and_removed_controls_are_absent`
and `test_production_llm_destination_owns_navigation_actions_and_recovery`).

Final PR refresh: the branch was rebased again onto `origin/dev` at
`ebeae144042c744d1639df2f192bda8d63aa78b6`. Exact generated comparison with
that base records inventory `470/1155/6890/6`; the base and branch differ only
in `monitoring_engine.py`'s digest (`65f3a0c1be12db1830f4` to
`28b1354fcd730b42d311`). Both retain 16 calls, the same owner/reason, six sink
files, and non-monitoring SHA-256
`284f2c81631ff4b06b67971e996258b2b831b731f097b05dae46fd11aea0e2f3`.
Fresh post-rebase verification retained 77 sanitizer/security passes, one
installed-wheel pass, eight inventory architecture passes, green static/range/
syntax/diff checks, and only the same two known full affected-module failures.

Lessons recorded: no-match input does not prove matched-scanner linearity; and a
rebase must rerun cross-cutting generated-manifest gates even when scoped files
are unchanged.
<!-- SECTION:NOTES:END -->
