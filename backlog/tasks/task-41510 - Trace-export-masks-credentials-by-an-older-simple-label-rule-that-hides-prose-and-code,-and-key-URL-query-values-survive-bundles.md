---
id: TASK-41510
title: "Trace export masks credentials by an older simple-label rule that hides prose and code, and ?key= URL query values survive bundles"
status: Done
assignee:
- '@claude'
updated_date: 2026-11-18 04:15
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
- **Base vs head probe** (21 simple labels x 13 values x 7 separators x 5 contexts): no credential value masked at base is kept at head. **Wrong (review r16tr-rev1):** its contexts had no label after `].`/`).`/`?.`/`$.`, no value on the line after `=`, and no unindented or post-blank-line value, all of which leaked; see the fix round below for the corrected probe.
- **Tests** (`Tests/Chat/test_trace_v2_collaboration.py`, laned nodes only): `_MUST_KEEP` +8 (`simple-code-call` is the AC's text), `_MUST_MASK` +19 (simple labels, quoted, dotted, hyphenated, inner, next-line, and six `?key=` shapes), `test_a_masked_run_stops_before_the_next_field` +2 (`value-below`, `heading-then-value`).
- **Red evidence.** At base: 16 failed / 349 passed on the two table tests (8 keep rows, 6 `?key=` rows, `simple-password-json`, `simple-code-default`). Rows the older rule already masked, under a temporary edit that drops it with no replacement (base sanitizer, head export): 24 failed, including the existing `dotted-label` and `probe8-cli-flag` rows. Targeted controls: `_VALUE_BELOW` off: 6 failed; `_is_credential_label` back in `_inner_values_look_alike`: `simple-inner-label` failed; `_LABEL_FACTS` caching every one-word label: the cache test failed (`[0, 12] == [0, 7]`); `_LABEL` without `-*`: `probe8-cli-flag` failed. Green after: 879 passed (`test_trace_v2_collaboration.py` + `test_log_sanitizer.py`, `-n 4`, load 58-65).
- Files: `tldw_chatbook/Utils/log_sanitizer.py`, `tldw_chatbook/Chat/trajectory_export.py`, `Tests/Chat/test_trace_v2_collaboration.py`.

**Review r16tr-rev1 (fix round 1).**
- **Critical 1(a).** `_LABEL`'s lookbehind is `(?<![A-Za-z0-9_-])(?<![A-Za-z0-9_.-]\.)` (the reviewer's): a label may follow a `.` that does not end a name, so `users[0].password = "..."`, `get_user().token`, `this.cfg?.secret`, `row['x'].password`, `$.password:`, `  .password` and `users[0].client_secret` are judged, while `db.password` stays one label (`db.password = os.getenv("X")` and `self.token = None` kept).
- **Critical 1(b)(c), Minor 3.** `_VALUE_BELOW` now skips blank lines and needs no indentation, and excludes a nested `"name":` as well as `name:`, list items and comments. `_owns_value_below` decides whether the value below a label that ends its line is the label's: after `=` always (`const password =\n  "..."`); after `:` for a YAML key (only indentation or `- ` before the label) and otherwise unless the label's line opens a code block or list item (`if`/`elif`/`else`/`while`/`for`/`with`/`match`/`case`/`class`/`def`/`async`/`try`/`except`/`finally`, `1.`, `-`, `*`, `+`; `_BLOCK_OPENER`). **Deviation from Minor 3's wording**, which allowed the rule only for a label that starts its line: under that rule a mid-line label leaked, and in a bundle every row sits mid-line (`Request failed with ...`, `key ... in a tool result`), so a temporary edit to it turned 7 must-mask rows red (`next-line-prose` `my password:\nhunter2xyz`, `next-line-unindented`, `simple-yaml-next-line*`, `compound-yaml-next-line`). The block-opener test keeps all six Minor 3 shapes. It reads the label's line back with `rfind` bounded by the previous `\n`, once per line.
- **Important 2.** `_line_end` is one compiled `[\r\n]` search; `find("\r")` scanned to the end of a text with no CR once per line. The log path calls it too, so logs get faster with byte-identical output. CPU for `mask_credential_labels` on `"token: x9\n"` lines, round 1 vs now: 1.69/0.52 s at 100k, 5.60/1.02 s at 200k, 19.6/2.0 s at 400k (`sanitize_string`: 1.93/0.77, 6.03/1.54, 19.3/2.65).
- **Minor 4.** `_code_literals_look_alike` accepts a first argument or index that is the label's own name (`_label_name`): `password = request.form.get("password")` and `secret = os.getenv("secret")` are kept, like `request.form["password"]`. `client_secret=SecretStr("secret")` stays masked (the literal is not the label's name).
- **Minor 5.** The `_SIMPLE_CREDENTIAL_LABEL` comment now says the old rule matched after any non-word character.
- **Rows.** `_MUST_MASK` +14 (7 after-dot shapes, 3 after `=`, 3 unindented/after-blank/prose, 1 indented prose), `_MUST_KEEP` +5 (`db.password = os.getenv("X")`, `self.token = None`, two calls and one index naming the label), `test_a_masked_run_stops_before_the_next_field` +6 block cases (each Minor 3 shape stays and the field after it is masked; not `_MUST_KEEP` rows because a preview joins lines, and on one line `if token: return token` is masked, as at base) and `heading-then-value` replaced by `value-after-blank` (under ruling (c) a value after a blank line is judged, so `password:\n\nSee below.` now masks `See below`). Timing: adversarial `many-label-lines` (150k lines, 0.93 s at head) and a `label-lines` ratio case in `test_runs_after_a_value_scan_in_linear_time`.
- **Red evidence.** Against round 1's sanitizer (314ff9c5fa): 22 failed (13 must-mask, the 2 call-naming keep rows, `value-after-blank` and the 6 block cases); `many-label-lines` 5.36 s CPU against the 2 s bound and `label-lines` 1.55 s against 8 x 0.19 s. Controls: Minor 3's literal line-start rule, 7 failed (above); no own-name argument and no quoted-key exclusion, 3 failed. Green: `test_trace_v2_collaboration.py` + `test_log_sanitizer.py` 918 passed (`-n 4`, load 35-59).
- **Corrected probe** (22 labels x 13 values x 13 separators, including `:\n`, `:\n\n`, `=\n`, x 19 contexts, including `users[0].`, `get_user().`, `?.`, `$.`, `  .`, `const`, `my`, `- `, `if`, `class` and `1.`; 70,642 texts): 3,024 values masked at base are kept at head, every one a `label:` whose line opens a code block or list item (`if`, `class`, `1. Generate a`) with the value on a later line, which is Minor 3's carve-out. No other context keeps a value base masked.
- **Logs.** Base vs head over 2,215 texts (both tables, look-alikes, sanitizer shapes, CR, LF and CRLF line texts) x 5 log functions, plus `sanitize_dict`/`sanitize_list`: 0 diffs.

**Coordinator ruling on the carve-out (after fix round 1).** A credential-shaped value base masked must not be kept, so list items leave `_BLOCK_OPENER`: only code-block openers (`if`, `elif`, `else`, `while`, `for`, `with`, `match`, `case`, `class`, `def`, `async`, `try`, `except`, `finally`) keep the value below out. `1. Password:\n   hunter2xyz`, `- Your password:`, `* API key:` and `+ token:` with a value below are masked (`_MUST_MASK` +4: `list-numbered`, `list-dash-prose`, `list-star`, `list-plus`); `- password:` stays the YAML-key form. `1. Generate a token:\n   Go to Settings > Tokens` now masks the whole line below, not only the first word as base did: "Go" starts an expression that is masked whole (exact case `list-item-first-word`; the block case `block-list-item` is gone). Red at the previous commit (4c6535a161): those 5 failed; green after.
- **Probe re-run** (85,514 texts: the 19 contexts above plus `* `, `+ the `, `- Your ` and `  2) `): 2,016 values masked at base are kept at head, all under `if` or `class` lines (2 openers x 4 later-line separators x 252 label/value pairs). No list-marker or prose context keeps one.
- **Runs** (load 23-35): `test_trace_v2_collaboration.py`, `test_log_sanitizer.py`, the trajectory export/import files, `Tests/UI/test_trajectory_import_ui.py`, `Tests/CI/test_ci_queue_pressure_contract.py` (`-n 4`): 1119 passed. Log byte-identity probe: 2,235 texts, 0 diffs. Serial size and identity ratchets: 270 passed. `preflight.sh`: passed.
