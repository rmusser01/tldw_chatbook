---
id: TASK-40301
title: "A shared trace bundle leaves GitHub, Hugging Face, Google, Slack and webhook secrets unmasked that the log sanitizer masks"
status: Done
created_date: 2026-11-12 19:36
updated_date: 2026-11-13 03:33
assignee:
- '@claude'
labels:
- follow-up
- security
- privacy
- trace
- export
priority: high
references:
- tldw_chatbook/Chat/trajectory_export.py
- tldw_chatbook/Utils/log_sanitizer.py
- Tests/Chat/test_trajectory_export.py
dependencies:
- TASK-39149
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down (security review r13s-rev1 of TASK-39149). Checked by reading the code at the round-13 stack top 8b633d0101, 2026-11-12; line numbers are that tree's.

**Why.** Trace v2 bundles are made to be shared. The export builds from `TrajectorySnapshot`, not from stored traces that credentials-v2 already sanitized, so its own patterns are the only mask. Today a GitHub PAT, a Hugging Face token, a Google API key, a Slack token or a webhook secret in a message or tool result ships verbatim in every profile, while the same text is masked in logs.

**The gap.**
- `trajectory_export.py:300-320` (`_CREDENTIAL_VALUE_RES`, from 794128caf7) covers PEM private keys, bearer and basic credentials, `sk-`, `[rs]k_(live|test)_`, `AKIA`, URL user info and `key=`-style assignments.
- `log_sanitizer.py:104-138` (`_STANDALONE_CREDENTIALS`) also covers `AIza`, `gh[pousr]_`, `github_pat_`, `hf_`, `xox[baprs]-`, the AWS `ASIA/AGPA/AIDA/AROA` prefixes and bare JWTs (`eyJ...`). The export has none of these.
- `whsec_` (and `gsk_`) is in neither list.
- Two hand-kept lists drift; TASK-39149 fixed the last gap the same way.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An exported bundle masks every credential shape the log sanitizer masks, in previews, payload values and keys, for all three export profiles; a test is red before the fix
- [x] #2 `whsec_` and `gsk_` are masked in both places, or the notes name them out of scope with the reason
- [x] #3 The shapes masked today stay masked, and the notes say what happens when an older bundle holding a newly masked shape is imported
<!-- AC:END -->

## Implementation Plan

1. Re-check the premise on base 26b2e677dd: `_CREDENTIAL_VALUE_RES` in `trajectory_export.py` still lacks the sanitizer's `_STANDALONE_CREDENTIALS` shapes.
2. In `Utils/log_sanitizer.py`, add one public tuple of every credential value shape logs mask (private key block, URL user info, Bearer, the standalone and underscore-key tokens, plus a diagnostic-only tuple holding `whsec_` and `gsk_`, kept out of both stored trace projections so stored history is unchanged); `_sanitize_diagnostic_text` reads the token part of it.
3. `trajectory_export._CREDENTIAL_VALUE_RES` keeps its own shapes (AC#3) and appends that tuple, so export and logs share one source.
4. Tests in `Tests/Chat/test_trace_v2_collaboration.py` (fake token shapes, three profiles, preview, payload value and key; a drift pin; import of an older bundle holding a newly masked shape) plus a log-sanitizer case for `whsec_`/`gsk_`; show red first; lane the new export node next to TASK-39149's.

## Implementation Notes

A shared Trace bundle now masks every credential value shape the log sanitizer masks, from the sanitizer's own tuple.
- **Premise held on base 26b2e677dd**: `trajectory_export._CREDENTIAL_VALUE_RES` still had only its own list (PEM, Bearer/Basic, `sk-`, `[rs]k_(live|test)_`, `AKIA`, `scheme://user:pass@`, `key=` labels).
- **One source of truth**: `Utils/log_sanitizer.py` exports `CREDENTIAL_VALUE_PATTERNS`: the private-key block (also an unterminated one), URL user info (also user-only, such as `https://<token>@host`), Bearer with any token, and every token shape (`_STANDALONE_CREDENTIALS`, `_UNDERSCORE_KEY_CREDENTIALS`, and a new `_LOG_ONLY_CREDENTIALS`). `_sanitize_diagnostic_text` reads the same token tuple, so a shape added to any of those tuples reaches logs and bundles together. `_CREDENTIAL_VALUE_RES` is that tuple first (so the sanitizer's Bearer and user-info rules see the whole value), then the export's own shapes, which stay (AC#3). A test pins that every sanitizer pattern is in the export's tuple by identity.
- **AC#2**: `whsec_` (Stripe/Svix webhook signing secrets, base64 body with optional `=` padding) and `gsk_` (Groq) are masked in both places: logs and bundles. They are deliberately in neither stored trace projection (`sanitize_trace_credentials_v1`/`_v2`): those outputs are compared with stored history, so changing them needs a new credentials version (proposed follow-up). Like the other token rules, the body has no `_` and a length floor (24 and 20), so `whsec_mode` or `gsk_enabled_for_...` stays readable.
- **Scope**: value shapes. Label-driven masking stays as each side had it: the export masks credential-named keys (`_CREDENTIAL_KEY_RE`) and `api_key=`/`token=`/`password=`/`secret=` in text; logs also mask compound labels such as `client_secret=` in text (proposed follow-up).
- **AC#3, older bundles**: an older bundle that holds a newly masked shape verbatim (a GitHub or Hugging Face token, a JWT, a user-only URL user info, …) is refused on import, because import runs the same credential check over the whole document: "Trace v2 import blocked: credentials are forbidden in collaboration bundles; request a privacy-safe export". Exporting the trace again from the app gives a masked bundle that imports.
- **Tests** (fake values assembled from fragments, so no committed line holds a token-shaped literal):
  - `Tests/Chat/test_trace_v2_collaboration.py::test_every_shape_the_log_sanitizer_masks_is_masked_in_a_shared_bundle` (29 shapes x 3 profiles; each in a preview, a payload value and a payload key) also asserts the log sanitizer masks it; `test_the_bundle_masks_with_the_log_sanitizers_own_patterns`; `test_an_older_bundle_holding_a_newly_masked_shape_is_refused_on_import`.
  - `Tests/Utils/test_log_sanitizer.py::test_webhook_secrets_and_groq_keys_are_masked_in_logs_only` (both projections unchanged) and `test_identifiers_that_share_the_webhook_or_groq_prefix_stay_readable`.
  - Red before the fix: 71 failed, 23 passed (the 23: shapes the export already masked, such as `sk-`, `sk_live_` and `AKIA`, and the readable-identifier cases); e.g. `assert b'ghp_DONOTUSE…' not in b'{…}'`, import `DID NOT RAISE`, `'whsec_…' in 'webhook verify failed with whsec_…'`. Green after; the existing TASK-39149 nodes stay green.
- **Lane**: the three export nodes join the recovered-suites step after TASK-39149's (`derived-artifacts.yml`), 89 cases in 21.5 s serial with that step's flags, load 47 on 18 CPUs. `Tests/Utils/test_log_sanitizer.py` already runs whole there.
- **Runs** (run.sh, `-n 4`, load 50-70): the 40 non-Architecture test files that name `trajectory_export`, `trajectory_import` or the log sanitizer: 1502 passed, 8 failed, 1 error. With both changed sources at base, the same command over those nodes: 7 failed and the same collection error (`No module named tldw_chatbook.Chat.console_thinking`); the 8th, `test_installed_tiktoken_bundle_is_offline_and_immutable[direct-engine-first-sdist]`, passed at head on its own (190 s, a load flake). The 7 are `RecoveryRequired`, packaging and RAG-closure reds present at base.
- Files: `tldw_chatbook/Utils/log_sanitizer.py`, `tldw_chatbook/Chat/trajectory_export.py`, `Tests/Chat/test_trace_v2_collaboration.py`, `Tests/Utils/test_log_sanitizer.py`, `.github/workflows/derived-artifacts.yml`.
