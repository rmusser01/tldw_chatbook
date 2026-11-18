---
id: TASK-41510
title: "Trace export masks credentials by an older simple-label rule that hides prose and code, and ?key= URL query values survive bundles"
status: Done
assignee:
- '@claude'
updated_date: 2026-11-18 02:41
created_date: 2026-11-17 05:27
labels:
- follow-up
- security
- trace
priority: high
references:
- tldw_chatbook/Chat/console_trace_export.py
dependencies:
- TASK-40301
- TASK-41133
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the trace stream (round 15) after TASK-40301 and TASK-41133. Two gaps: (1) the export's older simple-label rule masks ordinary prose and code such as `token = oauth.fetch_token(url,)`, so a bundle is unreadable where nothing secret is present; the credential-shaped value rule (as used by TASK-40301) should decide instead; (2) `?key=` URL query values that are not `AIza...` survive shared bundles while logs mask them (a TASK-40301 leftover). TASK-41133 covers `whsec_`/`gsk_` and compound labels, not these.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `token = oauth.fetch_token(url,)` and similar prose or code is not masked, and a real credential-shaped value is
- [x] #2 A `?key=<value>` query value of any shape is masked in bundles as it is in logs
- [x] #3 Tests pin both, each red before the fix
<!-- AC:END -->

## Implementation Plan

1. Base check (45b1949ca3): `_credential_text` masks `token = oauth.fetch_token(url,)` whole through the older simple-label regex (last in `trajectory_export._CREDENTIAL_VALUE_RES`); `mask_url_query_keys` keeps `?key=python`, `?key=YOUR_API_KEY`, `?key={api_key}` and `?key=photos/...`, which logs mask.
2. `log_sanitizer.mask_credential_labels` also reads the labels the older rule covered (`token`, `password`, `passwd`, `secret`, `apikey`, `api key`, and a label ending in one after `.` or `-`) and judges their values with the look-alike rules compound labels use; the older regex leaves `_CREDENTIAL_VALUE_RES`.
3. `mask_url_query_keys` masks every non-empty `?key=` value but an emitted marker.
4. Logs: `_sanitize_diagnostic_text` calls neither function; `Tests/Utils/test_log_sanitizer.py` stays green unchanged.
5. Rows in `Tests/Chat/test_trace_v2_collaboration.py`: `_MUST_KEEP` gains prose and code under simple labels, `_MUST_MASK` gains real values under them, and the five `url-key-*` rows move from keep to mask. Red at base for the new rows; old-rule rows shown red under a temporary edit that drops the old rule with no replacement.

## Implementation Notes

Simple labels are judged as compound labels are, and a URL's `?key=` value is masked whatever its shape.

- **Base (45b1949ca3).** The last regex in `trajectory_export._CREDENTIAL_VALUE_RES` (`\b(api key|token|password|passwd|secret)\s*[:=]\s*\S+`) replaced label and first word with the marker: `token = oauth.fetch_token(url,)` became `[credential redacted]`, yet `password = os.getenv("DB_PASSWORD", "postgres")` kept `"postgres"` and `{"password": "..."}` (quoted key) was not masked. `mask_url_query_keys` kept `?key=python`, `YOUR_API_KEY`, `API_KEY`, `{api_key}` and `photos/2024/cat.jpg` (review r15tr-rev4/rev5 look-alikes).
- **AC#1.** The older regex is gone. `log_sanitizer._LABEL` (was `_COMPOUND_LABEL`) also matches one word, `api key`, and leading hyphens (`--client-secret=`, which only the older rule caught). `_credential_label_text` decides: a label the older rule covered (`token`, `password`, `passwd`, `secret`, `apikey`/`api key`, alone or after `.`/`-`, such as `db.password`, `settings.api_key`, `page-token`) or a compound credential label; other single words (`Authorization`, `cookie`) stay out, so `Authorization: Bearer <token>` is still kept. Values are judged by the existing look-alike rules (`_kept_look_alike`, whole-expression spans), so code, prose, `None` and annotations are kept and real values masked, label kept visible. `_inner_values_look_alike` uses the same predicate. A bare `label:` whose value sits on the next indented line (`password:\n  hunter2x`, which the older `\s*` read) is judged too (`_VALUE_BELOW`; a nested `name:` line, a list item, a comment or a blank line is not a value). This also applies to compound labels.
- **AC#2.** `mask_url_query_keys` masks every non-empty value but an emitted marker (trailing full stop or unbalanced closer stays visible). The five `url-key-*` look-alike rows moved from `_MUST_KEEP` to `_MUST_MASK`, reversing the r15tr-rev4/rev5 rulings per this task.
- **Logs unchanged.** `_sanitize_diagnostic_text` calls neither function. Probe: base vs head `sanitize_string`, `redact_log_line`, `sanitize_trace_credentials_v1/_v2/_v3`, `sanitize_dict`, `sanitize_list` over 2105 texts (both tables, look-alikes, sanitizer shapes, variants): 0 diffs. `Tests/Utils/test_log_sanitizer.py` green, unchanged.
- **No cache, linear.** `_LABEL_FACTS` admits nothing new (`db.password`, `api key` and single non-credential words are never cached); `seen` stays per call. New cache rows (a one-word secret as a label, `api key:`, `password:`) and three adversarial rows (200 KB of `a=`, of `token= `, of `api `) pass the existing bounds.
- **Base vs head probe** (21 simple labels x 13 values x 7 separators x 5 contexts): no credential value masked at base is kept at head.
- **Tests** (`Tests/Chat/test_trace_v2_collaboration.py`, laned nodes only): `_MUST_KEEP` +8 (`simple-code-call` is the AC's text), `_MUST_MASK` +19 (simple labels, quoted, dotted, hyphenated, inner, next-line, and six `?key=` shapes), `test_a_masked_run_stops_before_the_next_field` +2 (`value-below`, `heading-then-value`).
- **Red evidence.** At base: 16 failed / 349 passed on the two table tests (8 keep rows, 6 `?key=` rows, `simple-password-json`, `simple-code-default`). Rows the older rule already masked, under a temporary edit that drops it with no replacement (base sanitizer, head export): 24 failed, including the existing `dotted-label` and `probe8-cli-flag` rows. Targeted controls: `_VALUE_BELOW` off: 6 failed; `_is_credential_label` back in `_inner_values_look_alike`: `simple-inner-label` failed; `_LABEL_FACTS` caching every one-word label: the cache test failed (`[0, 12] == [0, 7]`); `_LABEL` without `-*`: `probe8-cli-flag` failed. Green after: 879 passed (`test_trace_v2_collaboration.py` + `test_log_sanitizer.py`, `-n 4`, load 58-65).
- Files: `tldw_chatbook/Utils/log_sanitizer.py`, `tldw_chatbook/Chat/trajectory_export.py`, `Tests/Chat/test_trace_v2_collaboration.py`.
