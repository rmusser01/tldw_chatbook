---
id: TASK-41133
title: "Trace credential masking leftovers: a credentials-v3 projection for whsec_ and gsk_, and compound labels such as client_secret= in shared bundle text"
status: Done
created_date: 2026-11-15 02:20
updated_date: 2026-11-17 16:53
assignee:
- '@claude'
labels:
- follow-up
- security
- trace
priority: low
references:
- tldw_chatbook/Utils/log_sanitizer.py
- tldw_chatbook/Chat/trajectory_export.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 endpoint stream fu/r14-endpoint 6cb8e34461 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** TASK-40301 masked `whsec_` (Stripe/Svix webhook secrets) and `gsk_` (Groq) in logs and shared bundles, but deliberately not in the stored trace projections, because changing those outputs needs a new credentials version. Stored trace history therefore still holds them. Separately, shared bundle text masks `api_key=`/`token=`/`password=`/`secret=` but not compound labels such as `client_secret=`, which logs do mask.

**Evidence.** TASK-40301's notes (endpoint branch): AC#2 "They are deliberately in neither stored trace projection (`sanitize_trace_credentials_v1`/`_v2`) … (proposed follow-up)"; Scope: "logs also mask compound labels such as `client_secret=` in text (proposed follow-up)".
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A credentials-v3 stored-trace projection masks whsec_ and gsk_ values; existing v1/v2 history still verifies, and new traces use v3
- [x] #2 Shared bundle text masks compound credential labels the log sanitizer masks (client_secret=, refresh_token= and the like), with tests red first
<!-- AC:END -->

## Implementation Plan

1. Add `sanitize_trace_credentials_v3` (v2 plus `whsec_`/`gsk_`) in `Utils/log_sanitizer.py`, reusing the existing `whsec_`/`gsk_` patterns, and make `credentials-v3` the current `CREDENTIAL_FILTER_VERSION` with v1/v2 still readable; give v3 its own fixed legacy policy id.
2. Generalise the v1-only span widening so a stored v1 or v2 policy's frozen PII spans are widened over every key a newer projection masks.
3. Tests red first: v3 masks the two shapes and reports `credentials-v3`; stored v1 and v2 traces read, export and share without them and are not rewritten; v1/v2 projections stay byte-identical; a v2 legacy ledger keeps migrating under v3.
4. Split the log sanitizer's diagnostic label predicate into credential labels and PII labels, and have shared bundle text mask credential-shaped values under compound credential labels (`client_secret=`, `refresh_token=`) through that one predicate; user names, prose, templates and identifiers survive. Tests red first.
5. Covering tests, ratchets, ruff, preflight.

## Implementation Notes

**credentials-v3 (AC#1).** `sanitize_trace_credentials_v3` (`Utils/log_sanitizer.py`) is v2 plus the existing `whsec_`/`gsk_` patterns, renamed `_LOG_ONLY_CREDENTIALS` -> `_V3_KEY_CREDENTIALS`, so logs, shared bundles and stored traces use one pair of regexes. `CREDENTIAL_FILTER_VERSION` is now `credentials-v3` (`console_trace_redaction.py`), with v1 and v2 still readable and byte-identical. Every read, export and share of v1/v2 history goes through the generic readable-version paths TASK-39123 built, so it re-applies v3 with no other change. Two v1-only pieces needed generalising:
- The frozen-PII-span widening now covers every key a newer projection masks (`trace_credential_key_spans`, which replaces `trace_credentials_v2_spans`; `widen_over_newer_keys` replaces `widen_over_v2_credentials`). A v2 policy's span that cut a `gsk_` key would otherwise leave a fragment v3 no longer matches.
- `console_trace_legacy._LEGACY_POLICY_IDS` gains `credentials-v3` (`...099`). Without it a v2 legacy ledger hits TASK-40303's conflict again; the parametrised ledger test catches that.

The rule comment beside `_CREDENTIAL_PROJECTIONS` now names both of these, so the next version does not miss them.

**Compound labels in bundles (AC#2).** The log sanitizer's diagnostic label list is split into credential labels (`_DIAGNOSTIC_CREDENTIAL_LABELS`, the suffixes and the `x_*_token/_key` rule, read by `_is_credential_label`) and PII labels. `_is_sensitive_diagnostic_key` is their union, so log behaviour is unchanged. `mask_credential_labels` masks a value only under a compound label (it has a `_`, `-` or `.`) that `_is_credential_label` accepts, and only a credential-shaped value, always masked whole (rules below, as revised in review r15tr-rev1). `trajectory_export._credential_text` applies it after its regexes, so import refuses an older bundle that holds such a value, the same way it refuses other credential material. The result is idempotent: `client_secret=[credential redacted]` does not match again. Values the rule deliberately keeps:
- Code: `client_secret=client_secret`, `os.environ[...]`.
- Placeholders: `<your refresh token>`, `${CLIENT_SECRET}`.
- Prose: `client_secret: it comes from ...`.
- User names: `user_name=`, `username:`.
- Labels that are not credentials: `max_tokens`, `next_page_token`.

Known limits, after review r15tr-rev8. Under a compound credential label the bundle keeps only the allowlist. Every value, quoted or not, is judged with the whole expression it starts. That expression runs to a top-level `,` or `;`, an unbalanced closer, or the next field (`name=`, `name:`, `"name":`). It also runs over line ends while a bracket is open. It is kept only when the value is kept, every later literal is a look-alike, and no random-looking word sits outside quotes. These are the remaining ways a real secret could be kept:
- Under any label:
  - a placeholder (`your-word`, `YOUR_WORD`, `YourWordHere`, `example`, `none`, all-`x`/`*`);
  - a digit-free name or a phrase in angle brackets (`<letmein>` is kept by design);
  - a reference or template that names a credential;
  - a known-root path with no random-looking segment;
  - a dotted name of conventional segments;
  - an UPPER_SNAKE constant holding every label word;
  - code whose first arguments and indexes are names (an UPPER env name that is not 12+ letters with five consonants in a row, a compound key in a call, any lowercase key in an index) and whose other literals are look-alike values. So `SecretStr("HUNTER")` and `SecretStr("super_secret")` are kept.
- Under issued-secret and token labels only (not password labels or labels ending in `secret`/`secret_key`):
  - `changeme`;
  - a non-dotted call;
  - a value that is wholly a time or an ISO date, or starts with `scheme://` or `file:/`;
  - the label's own name, bare or versioned by at most two digits;
  - a snake name of credential words only;
  - a quoted sentence with no letter-and-digit word;
  - a function word followed by a word that is not token-shaped. Token-shaped means letters and digits, two case humps, or 12+ letters with five consonants in a row. So random letters with vowels between, such as `client_secret: the abcdefghijkl`, are still read as prose.
- A URL's `?key=` value: lowercase words or a `/`-path with no letter-and-digit word are kept.
- An unquoted value that runs on past its first word stops before the next `name=` field (`db_password=my pass=word here` leaves `pass=word here`).
- An escape-quoted value (`\"client_secret\": \"...\"`, JSON inside a string) is judged as a lone literal, with nothing after it.

Simple labels (`api_key=`, `token:`) stay with the export's older rule. A trailing full stop and a separator before the next field stay visible. `==` is never a separator, and a comparison's right operand is not a label, so `if client_secret == other_secret: return` is kept.

Masked by design although they are not secrets:
- Any call the grammar does not parse, masked whole: nested, with a number argument, or spread over lines. That includes a look-alike like a black-formatted `os.environ.get(\n    "CLIENT_SECRET",\n    "",\n)`.
- `tokens[0]`.
- `cfg.get("token")`, since a lowercase call key needs an underscore.
- An expression longer than 2 KB, masked to the end of the line the limit falls on.
- A label read as a condition without a comparison (`if client_secret: return`).
- A date or code under a strict label.
- Names with words besides the label's.

**Review r15tr-rev1.** The first rule missed real values and over-masked look-alikes, and it could leave a partial mask (`db_password=Tr0ub4dor&3` came out as `[credential redacted]&3`).
- **Value extent.** A value is now quoted whole, or bare up to whitespace, a quote, or a `&`/`;`/`,` that starts another field (`name=`) or ends the text. So a value is never cut inside itself. Labels may repeat separators (pydantic's `APP__CLIENT_SECRET`).
- **What counts as credential-shaped** (`_credential_shaped`): a digit (6+ characters), or mixed case (16+). A `-` or `/` alone does not count. A value is rejected when it starts like a path, reference or placeholder (`/`, `.`, `~`, `$`, `<`, `{`, `%`, `:`, `*`, `[`, `(`), when it is a call, index or time (`b64encode(`, `os.environ[`, `10:00`), a date or number, or a dotted name.
- **Red first** (run.sh, `-n 4`, load 81, before the rule change):
  - In `_LOG_SANITIZER_SHAPES`, 18 nodes for six new shapes (punctuation, JSON with `&`, a `&3` tail, a `%2F` tail, URL-encoded `refresh_token`, pydantic env) were red.
  - The new `test_label_look_alikes_survive_a_shared_bundle_and_an_older_one_imports` (placeholders, a Kubernetes secret name, paths, code, a date and a time, references) had 33 red nodes. It asserts the text survives export and that an older bundle holding it imports.
  - The query-shape fake gained digits (`GOCSPX-...a1B2`), as a real client secret has them.
  - After the change, the collaboration file gives 338 passed.
- **Reviewer probes** (run.sh): `probe2.py` and `probe_labels.py` show every listed false negative masked whole and every listed false positive kept. `_has_credential_material` is False for the placeholder and the path. Three cases still masked by the export's older simple-label rule, not by this one: `api_key=config.provider2.key`, `api_key=$OPENAI_API_KEY_2` and `X-Request-Token: abc`. `?key=` in a URL is still kept, because `key` is not a compound label. `probe_equiv.py` gives corpus 4001, 0 diffs, and the token and value pattern tuples are unchanged, so log output is byte-identical to base.
- The `sanitize_trace_credentials_v2` docstring now describes it as the stored v2 projection.
- The look-alike node is pinned in recovered-suites beside the label-list node.

**Red first.** At base cc6006fcf5 (run.sh, `-n 4`, load 72-81): 28 red in `Tests/Utils/test_log_sanitizer.py` plus the v2 ledger case. These cover v3 detection, stored v1/v2 traces holding `whsec_`/`gsk_` (read, export and share), v2 frozen spans, cut spans, and `test_webhook_secrets_and_groq_keys_are_masked_in_logs_and_from_v3`. In `test_trace_v2_collaboration.py`, 15 shape nodes for the five new compound-label shapes are red (the `X-Auth-Token:` header was already masked by the export's `\btoken` rule), and the label-list node fails with an AttributeError. Negative controls on the fixed tree, each reverted:
- Dropping the `whsec_` floor turns the identifier pin red.
- Dropping the credential-shape lookahead turns 6 look-alike nodes red (`client_secret=client_secret`, `os.environ[...]`).
- Skipping the label predicate turns the look-alikes and the label-list node red (`AssertionError: card_number`).
- A private label list in place of the shared one turns only the label-list node red.

**Covering runs.** `test_log_sanitizer`, `test_trace_v2_collaboration`, legacy migration and credential filter: 484 passed, then 246 passed. The 78-file covering set (every test naming the credential versions, `CredentialSanitizer`, trajectory export/import or the legacy ledger): non-UI gives 201 failed, 3814 passed, 8 errors at load 70-120. 198 of those nodes fail the same way with the base source swapped in. They are dev reds on no lane: project-instruction traces, failed/discarded tool runs, agent-call thinking and voice factory composition, with causes such as `PreparedProviderRequest(cache_boundary=)` and RecoveryRequired. The other 3 (archive recovery follow-up `[True]`, voice privacy custom PII timeout, response-rules collector 5 s timeout) pass at HEAD on their own (3 passed). The 3 UI files give 41 passed. `Tests/CI` gives 1640 passed. The module-size ratchet gives 207 passed. The ruff gate is OK (1090 findings under the 1105 baseline) and the markup check is OK.

**Lane.** `Tests/Utils/test_log_sanitizer.py` runs whole in architecture-guards. In the recovered-suites job (and `RECOVERED_SUITES_TARGETS`), the renamed ledger node `test_an_older_legacy_ledger_keeps_migrating_new_exchanges_under_the_current_version` replaces TASK-40303's, and `test_the_bundle_masks_compound_labels_with_the_log_sanitizers_own_labels` joins it. The new shape and look-alike parameters ride nodes already listed there. Both nodes pass under `--storage-lease-guard`.

**User Guide.** `Docs/User_Guide/console/semantic-trace-capture.md` already says recognised secret formats are removed and lists none, so it needed no change.

**Files.** `tldw_chatbook/Utils/log_sanitizer.py`, `tldw_chatbook/Chat/console_trace_redaction.py`, `tldw_chatbook/Chat/console_semantic_revision.py`, `tldw_chatbook/Chat/console_trace_legacy.py`, `tldw_chatbook/Chat/trajectory_export.py`, `Tests/Utils/test_log_sanitizer.py`, `Tests/Chat/test_trace_v2_collaboration.py`, `Tests/Chat/test_console_trace_legacy_migration.py`, `Tests/Chat/test_console_trace_credential_filter.py`, `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`.
- **Review-round covering run** (`-n 4`, load 50-70): the 39 files that touch the export, the import, the log sanitizer or the maintenance loop give 11 failed, 1570 passed.
  - 8 of the 11 fail the same way with the base source swapped in.
  - The other 3 pass at HEAD on their own: a bug-report child timing, a logging timing assertion (13.4 s against 2 s), and an sdist release check that timed out at 300 s.
  - Serial with lane flags: `Tests/Utils/test_log_sanitizer.py` 209 passed; the recovered-suites slice 310 passed.

**Review r15tr-rev2.**
- **Nested labels.** The rev1 rule took a value whole and never looked inside it, so `client_secret=...` inside another compound label's value (a JSON string, a URL fragment, a query string, a pipe list) went unmasked. `mask_credential_labels` now finds each label prefix and reads a value only for a credential label. When it does not mask, it continues right after the label, so any value is scanned again. That is the reviewer's recursion done iteratively, which stays clear of Python's recursion limit on chained labels.
- **Key formats.** Now masked:
  - Values starting with `/`, `.` or `~`. Only `./`, `../`, `~/` and lowercase `/seg/seg` paths are kept.
  - Values with a `:`. Only times (`\d{1,2}:\d{2}`) and URLs are kept, so the Telegram `123456789:AAH...` token and `Tr0ub4dor:3x` are masked.
  - Long dotted tokens (`ya29.<long>`).
- **Quoted values** may hold the other quote and spaces, so `"Tr0ub4d0r's..."` and a quoted passphrase are masked whole.
- **Bare values** drop an unbalanced trailing `)`/`]`/`}` and a trailing `.`, so the bracket or full stop is kept and only the secret is masked. Lowercase identifiers and capitalised word runs are kept.
- **Quadratic regex (pre-existing).** The export's URL user-info rule (`\b[a-z][a-z0-9+.-]*://...`) took 121 s of CPU on a 200 KB dash-separated text. It gains `(?<![a-z0-9+.-])`, as `_URL_CREDENTIAL_USERINFO` has.
- **Bounded reads.** A value over 2048 characters is not read (possessive `{1,2048}+`), so chained credential labels scan in linear time.
- **Red first** (load 31-40): 67 nodes.
  - 14 new `_LOG_SANITIZER_SHAPES` × 3 profiles: six nested texts from probe4/probe5, then base64 `/`, Azure `.`, `~`, Telegram, colon password, dotted token, apostrophe and passphrase.
  - 7 look-alikes × 3 profiles.
  - 3 bracket and full-stop cases.
  - The 200 KB dash-separated timing case, which took about 2 min before failing.
- **Negative controls**, each reverted:
  - No rescan turns the 18 nested nodes red.
  - No trimming turns the 3 bracket cases red.
  - No identifier and word-run rules turn the 21 look-alike nodes red.
  - No length cap turns the chained-credential-labels timing case red (7 min 47 s).
- **Probes** (run.sh): probe3, probe4 and probe5 give the outcomes above. The adversarial inputs run in 0.05-0.12 s; the dash-label input took 121 s before. probe2 and probe_labels are unchanged from rev1. `probe_equiv.py` gives corpus 4097, 0 diffs, and the pattern tuples are equal, so logs are byte-identical to base.
- The bracket and timing nodes are pinned in recovered-suites.
- **Rev2 evidence.**
  - Covering run (`-n 4`, the same 39 files, load 37-65): 8 failed, 1645 passed. The 8 are the nodes that fail identically at base (rev1's base comparison).
  - Serial with lane flags: the recovered-suites slice (every pinned Trace export node, the ledger node, the parking and projection files) gives 412 passed in 113 s at load 59-63; `Tests/Utils/test_log_sanitizer.py` gives 209 passed.
  - `Tests/CI` gives 1640 passed, 1 skipped; the module ratchet 207 passed; the ruff gate is OK.

**Review r15tr-rev3 (engineering ruling: mask by default).**
- **Design.** A value under a compound credential label is now masked unless it is on an explicit look-alike allowlist (`_kept_look_alike` in `log_sanitizer.py`):
  - placeholders (`your-...`, `<...>`, all-`x`/`*` runs, `changeme`, `example`, `REDACTED`, `None`/`null`/`true`/`false`, bracketed markers such as `[credential redacted]` and `[local path withheld]`);
  - references (`$VAR`, `${VAR}`, `%VAR%`, `{name}`, `{{ ... }}`, `${{ ... }}`, `%(name)s`);
  - known-root paths and drive paths;
  - code (a call or index; an identifier built from its label's words);
  - times, dates and URL schemes;
  - prose under a non-password label.
  Pure numbers, bcrypt `$2b$...`, `password123`, `summer2024`, `abc_123` and `CorrectHorseBatteryStaple` are masked.
- **Additions beyond the ruling's list**, each needed by a reviewer look-alike:
  - `None`/`null`/`true`/`false` (`access_token=None, max_retries=3`);
  - templates `{{...}}`, `${{...}}`, `%(x)s` and `{x}` (probe3);
  - a `file:/` URI (probe3);
  - a leading `:` Ruby symbol and kebab-case names (`oauth-client-secret`);
  - a bare lowercase word continued by prose (`client_secret: see section 4.2 of the guide`);
  - bracketed markers, so a second pass changes nothing and a safe-summary export's `[local path withheld]` is not masked again.
- **The identifier rule is narrower than the ruling's wording.** A value is kept when it holds every word of its label, or when it is compound and every word is a credential word. That keeps `client_secret_2`, `token_v2`, `refresh_token2`, `YourClientSecretHere` and `SECRET_KEY_V2`, and masks `password123` and the probe's `MySuperSecretPassphrase`; "includes a word of the label" would have kept those two.
- **Overflow.** A value over 2048 characters is masked whole up to its delimiter, found with a linear scan; a kept value is at most 2048 characters, so rescanning stays linear. The slowest of 16 adversarial 200 KB texts takes 0.075 s CPU, so the 2 s bound keeps over 20x headroom.
- **Escapes.** Quoted values honour `\` escapes (`_find_quoted_end`), and JSON-escaped labels and values (`{\"client_secret\": \"...\"}`) are read.
- **`?key=`.** The value of a URL's `?key=` is masked as logs mask it (`mask_url_query_keys`).
- **Tables.** `_MUST_MASK` (85 entries) and `_MUST_KEEP` (55 entries) in `Tests/Chat/test_trace_v2_collaboration.py` now hold every secret and look-alike from the rev1-rev3 probes, and are the contract.
  - Must-mask: masked with no head or tail left, a second pass changes nothing, an older bundle holding it is refused, and a full export does not hold it.
  - Must-keep: unchanged in all three profiles, and an older bundle holding it imports.
  - Moved from keep to mask: the session-token UUID and `refresh_token: rotated2024.` (the ruling masks `summer2024`). `secret1` and `tok_v2` were withdrawn.
  - Against the previous round's source, 45 table nodes were red (37 mask, 8 keep).
  - Negative controls, each reverted: keeping long values turns the 7 overflow entries red; plain quote matching turns the escaped-quote entry red; keeping quoted prose under password labels turns the passphrase entry red.
- **Rev3 evidence.**
  - Reviewer probes, all re-run: probe2 to probe7 and probe_labels. Every secret is masked and every look-alike kept, except the four the ruling moved (kwarg-two withdrawn; sentence-period and the UUID masked; yaml-flow is a real secret, masked with its `}` kept). `probe_equiv.py` gives corpus 4327, 0 diffs, so logs are byte-identical to base.
  - Covering run (`-n 4`, 39 files, load 22-52): 8 failed, 1651 passed. These are the same 8 that fail at base.
  - Serial with lane flags: the recovered-suites slice (14 targets) gives 418 passed in 37 s at load 20-26; `Tests/Utils/test_log_sanitizer.py` gives 209 passed.
  - `Tests/CI` gives 1640 passed, 1 skipped; the module ratchet 207 passed; the ruff gate is OK.

**Review r15tr-rev4 (allowlist tightened as the reviewer specified).**
- **Strict labels.** Password labels (`password`, `passphrase`, `passwd`, `pwd`) and user-chosen secret labels (`secret`, `secret_key`, `jwt_secret`, `app_secret`, `webhook_secret`, `session_secret`, `signing_secret`, `cookie_secret`) keep only:
  - placeholders other than `changeme`;
  - references (`$VAR` and `%VAR%` UPPER_SNAKE only, so `$hunter22` and `%Passw0rd%` are masked);
  - known-root paths;
  - dotted code;
  - UPPER_SNAKE constants holding every label word (`DB_PASSWORD_2`).
- **Identifiers.** A name must be lower_snake, UPPER_SNAKE, lower-kebab or camelCase, so `Password_123`, `Client_Secret_2024` and `Password-2024` are not names. The credential-words-only branch takes snake names only, not capitalised words.
- **Prose.** A quoted sentence counts only when no word holds a letter and a digit. A bare word counts only if it is in the function-word list (`it`, `see`, `the`, ...).
- **Loose shapes:**
  - A call or index must be the whole value, balanced, with empty, quoted or name arguments. It is read whole, quotes included (`resp.json()['access_token']`).
  - Times and ISO dates must be the whole value.
  - Schemes: lowercase `scheme://` or `file:/` only.
  - `your[-_][A-Za-z_-]+` only.
  - `{...}` and `{{...}}` only around a snake or dotted name; `<...>` only with no letter-and-digit word.
  - Bracketed markers only as the exact markers this code emits.
- **Other fixes.**
  - A YAML block scalar (`client_secret: |`) is its indented lines.
  - Escape-quoted values find their closing quote within a line end cached per line, so the scan is linear: 0.84 s at 800 KB before, a 4x text now under 8x the time. The ratio test is red without the cache (12.5x).
  - `?key=` values are judged as `api_key` values (`mask_url_query_keys` reuses `_trimmed_end`), so `?key=YOUR_API_KEY`, `?key=API_KEY` and `?key={api_key}` are kept.
- **Tables.** `_MUST_MASK` now has 143 rows and `_MUST_KEEP` 57.
  - Every probe8 secret was added to must-mask. probe8's look-alikes are the five `?key=` URLs; the three that fit the allowlist were added to must-keep.
  - Moved to must-mask by the exact rules: `secret_key=base64.b64encode(os.urandom(32))` (strict label; a call argument), `the secret_key: 2024-01-01` (strict label) and `tokens[0]` (numeric argument). Must-keep gained a non-strict date row and a `tokens[name]` row in their place.
- **probe9 gate.** The reviewer's probe9 stops at its first loosening, because rev4 replaced the prose-length rule it edits.
  - An adapted probe9 (same method) runs the reviewer's 5 loosenings whose code is unchanged (verbatim), its 3 others by intent on the rev4 code, and one loosening for each of the 16 rules rev4 added: 24 in all. Each turns at least one must-mask row red, after rows were added for them (including `long-signed-url` for "no overflow masking").
- **Red first.** Against the rev3 source, 54 of the round's nodes were red: 50 must-mask, 3 must-keep (the `?key=` look-alikes) and the linear-scan ratio.
- **Logs.** `probe_equiv.py` gives corpus 4457, 0 diffs.
- **Rev4 evidence** (after the coordinator restart):
  - Every rev4 item is in the source. 21643a41bd also narrows `{...}` to snake or dotted names, as specified.
  - probe8: 0 kept secrets (47 masked). `?key=YOUR_API_KEY`, `API_KEY` and `{api_key}` are kept; `?key=python` and `?key=photos/2024/cat.jpg` are masked.
  - The reviewer's probe9 still stops at its first loosening, because its prose-length fragment is gone. The adapted probe9 (`$FU/r15tr-probe9-r4.py`, 24 loosenings) gives baseline 0 red, and every loosening turns at least one must-mask row red.
  - Earlier probes: no secret kept in probe2, probe_labels, probe3 or probe6. The only look-alikes that change are the rev3 reclassifications plus `tokens[0]`, `code-b64` and the strict-label date (rev4). probe4 and probe5 mask every nested case, probe7 agrees with rev2 on every case, and `probe_equiv` gives corpus 4457, 0 diffs.
  - Covering run (`-n 4`, 39 files with UI, load 50-104): 9 failed, 1712 passed in 63 min. The 8 known base failures, plus `Tests/UI/test_console_context_modal.py::test_context_modal_none_preview_replaces_stale_loaded_state`, which passes 3 of 3 on its own and only imports `content_fingerprint`.
  - Serial with lane flags: the recovered-suites slice (15 targets) gives 480 passed in 207 s at load 64-172; `Tests/Utils/test_log_sanitizer.py` gives 209 passed. The CPU-time and ratio tests passed at that load.
  - `Tests/CI` gives 1640 passed, 1 skipped (load 57); the module ratchet 207 passed; the ruff gate is OK.

**Review r15tr-rev5.**
- **The reviewer's three fixes:**
  1. **Strict labels by shape.** Any label ending in `secret` or `secret_key` is strict (`_strict_label`), except names ending in `client_secret`, `consumer_secret`, `api_secret`, `aws_secret_access_key` or `secret_access_key`. This replaces the fixed user-chosen list.
  2. **Literals inside code.** A code value is kept only when every quoted literal in it is an env name, a lowercase key or a known path. Multi-argument calls are read whole, so `os.getenv("DB_PASSWORD", "...")` and `os.environ.get("CLIENT_SECRET", "...")` mask whole with no tail. `os.getenv('CLIENT_SECRET_2')`, `os.environ["CLIENT_SECRET"]` and `resp.json()['access_token']` stay kept.
  3. **`?key=`.** A value is judged as a strict `api_key` value, plus lowercase words or a `/`-path with no letter-and-digit word. `?key=python` and `?key=photos/2024/cat.jpg` are kept and import.
- **Further changes probe10's 0-kept gate needed:**
  - **Names.** An identifier counts only as the label's own name, versioned by at most two digits; a camelCase name needs the version. The credential-words branch is snake only, not kebab. This masks `myClientSecret99`, `accessToken2024`, `token-token`, `client_secret_prod` and `clientSecret`.
  - **Templates and placeholders.** References and templates must name a credential (`$ADMIN123`, `{hunter}` masked). `<...>` needs a phrase (`<letmein>` masked). `your-` placeholders keep one case (`Your-Password` masked).
  - **Function words.** A function word before a token makes the rest of the line the value (`client_secret: the <token>`).
  - **Unquoted passphrases.** Under a strict label, a masked unquoted value runs on to the next field or the line end, so `db_password: correct horse battery staple` leaves no word.
  - **Reclassified from keep to mask:** `oauth-client-secret` and `ClientSecretFromVaultConfig` (words besides the label's).
- **Linear scans.** `_LETTER_DIGIT_WORD` and `_RANDOM_SEGMENT` are anchored at word starts, so a long run is scanned once. The rest-of-line check after a function word is capped.
  - Unanchored, the 200 KB long-run and spaced-label cases failed their CPU bound.
  - The slowest adversarial case now takes 0.39 s CPU (best of three, load 47), over 5x under the 2 s bound.
  - probe11: the line-end cache gives a 4.1x ratio for 4x the text (11.6x without it); 8 MB takes 2.2 s.
- **Tables.** `_MUST_MASK` has 180 rows (every probe10 secret, the coordinator's four, and rows for new rules), `_MUST_KEEP` 57. Against the rev4 source, 35 table nodes were red (33 mask, 2 keep).
- **Adapted probe9** (`$FU/r15tr-probe9-r5.py`): 37 loosenings, which are the rev4 set (fragments updated where the code moved) plus one or more for each rev5 rule. Baseline is 0 red, and every loosening turns at least one must-mask row red.
- **Probes.**
  - probe10: 0 kept (33 masked). probe8: 0 kept (47 masked).
  - probe2, probe_labels, probe3 and probe6 keep no secret; probe3's fixed point holds. probe4 and probe5 mask all 7 nested cases. probe7 agrees with rev2.
  - `probe_equiv` gives corpus 4561, 0 diffs, so logs are byte-identical to base.
- **Rev5 evidence:**
  - Covering run (`-n 4`, 39 files with UI, load 33-65): 8 failed, 1750 passed in 28 min. These are the 8 that fail at base.
  - Serial with lane flags: the recovered-suites slice (15 targets) gives 517 passed in 135 s at load 46-64; `Tests/Utils/test_log_sanitizer.py` gives 209 passed.
  - `Tests/CI` gives 1640 passed, 1 skipped (load 43); the module ratchet 207 passed; the ruff gate is OK.
  - `./scripts/preflight.sh` passes (rc 0, load 37-40).

**Review r15tr-rev6.**
- **Gate correction:** probe10's KEPT rows `<letmein>`, `<MyPassword>` and `client_secret = clientSecret` are rows the reviewer declined, so they are not gates. They are now kept on purpose.
- **The fixes:**
  1. **Defaults inside code.** A literal passes as a name only as a first argument or an index, and only when it names a credential. Any other literal is judged as the label's value under the same strictness. So `os.getenv("DB_PASSWORD", "postgres")` is masked and a placeholder default is kept. The six probe12 rows are in `_MUST_MASK`, plus `admin_password` and `changeme` defaults.
  2. **`<...>` placeholders.** A digit-free name (`<[A-Za-z][A-Za-z_-]*>`) or a phrase is kept. `angle-word` and `angle-camel` moved to `_MUST_KEEP` with the seven README texts, and an older bundle holding each one imports.
  3. **Function-word span.**
     - The span extends only when the very next word is token-shaped (a letter-and-digit word, or mixed case like `XkmPvLwQ`). It stops before the next field, searched for after that word.
     - The strict passphrase run uses the same `_field_end`, which also leaves a separator before the field visible (`[credential redacted] | user=bob`).
     - The three probe12 lines are in `_MUST_KEEP`, and `test_a_masked_run_stops_before_the_next_field` pins the exact masked output.
  4. **`client_secret = clientSecret`** is kept again (the label's own camelCase name), and is in `_MUST_KEEP`.
  5. **Scan cost.** Only the next word is examined, no longer 2 KB ahead. `_words` and `_strict_label` are cached.
     - probe13 (1 MB, rev5 file vs head): 0.30/0.31 s, 0.25/0.26 s and 0.33/0.35 s, so back to rev5 cost.
     - `test_runs_after_a_value_scan_in_linear_time` adds the ratio pin (4x text under 8x time) for the function-word and passphrase chains.
     - The ratio alone passed on the rev5 code (2 KB per label is linear), so the test also asserts each chain costs under 3x a chain of one-word values. On rev5 that assertion failed: 4.14 s vs 3 x 0.68 s.
- **Tables.** `_MUST_MASK` has 187 rows and `_MUST_KEEP` 70. Against the rev5 source, 23 nodes were red: 7 mask, 13 keep, 3 exact-output. Both linear-time nodes were red too (run alone).
- **Adapted probe9** (`$FU/r15tr-probe9-r6.py`): 40 loosenings, with fragments updated and one or more for each rev6 rule.
  - The prose loosening now widens the caller's function-word test, because `_kept_look_alike` no longer repeats it.
  - Baseline is 0 red, and every loosening turns at least one must-mask row red.
- **Probes.**
  - probe12: no swallowed fields, and no difference from rev5 on the docs look-alikes. Its four remaining KEPT hunt rows each follow from a reviewer rule: two `?key=` lowercase words (rev5 rule 3), a function word before a lowercase word (rev6 rule 3), and `pass=` read as a field.
  - probe10: KEPT only for the three declined rows. probe8: 0 kept.
  - probe2, probe_labels, probe3, probe4-7 and probe11: as in rev5.
  - `probe_equiv` (rev6 corpus 4626): 0 diffs.
  - Slowest adversarial case: 0.32 s CPU (load 26).
- **Lane.** `test_a_masked_run_stops_before_the_next_field` and `test_runs_after_a_value_scan_in_linear_time` are pinned in the recovered-suites lane (workflow and `RECOVERED_SUITES_TARGETS`).
- **Rev6 evidence:**
  - Covering run (`-n 4`, 39 files with UI, load 28-45): 8 failed, 1775 passed in 22 min. These are the 8 that fail at base.
  - Serial with lane flags: the recovered-suites slice (17 targets) gives 542 passed in 83 s at load 30-33; `Tests/Utils/test_log_sanitizer.py` gives 209 passed.
  - `Tests/CI` gives 1640 passed, 1 skipped (load 30), after the lane edits. The module ratchet gives 207 passed, and the ruff gate is OK.
  - `./scripts/preflight.sh` passes (rc 0, load 33-45).

**Review r15tr-rev7.**
- **The ruling: one rule for literals near kept code.**
  - In `mask_credential_labels`, a credential-labelled value that is code (a call, an index, a dotted name, a reference or an identifier) is judged with the whole expression it starts.
  - `_expression_end` tokenises with quoted strings skipped whole. The expression ends at a top-level `,` or `;`, an unbalanced closer, or the next field.
    - `==` is not a field.
    - A ternary's `?` pairs with its `:`.
    - An expression over 2 KB runs to the line end.
  - The expression is kept only when the value is kept, `_code_literals_look_alike` passes on the whole expression, and no random-looking word sits outside its literals. Otherwise it is masked whole.
  - The rule covers fallbacks after `or`, `||` and `??`, `if ... else`, and concatenation, with no operator-specific code.
- **Deviations from the ruling, and additions to it:**
  - The expression also ends at a top-level `,`, `;` or unbalanced closer, not only at `_field_end`. That keeps `{"client_secret": os.environ["CLIENT_SECRET"], "user": "bob"}` (pinned).
  - Masked code is masked whole as well, so `prefix + "<token>"` leaves nothing behind.
  - An unparsed call (`f(g("..."))`) is masked whole to its matching closer and the rest of its expression.
  - The call grammar takes keyword arguments, so `os.environ.get("CLIENT_SECRET", default=None)` is kept, and `default="hunter2x"` and `value="GOCSPX-..."` are judged as values.
- **Names in a first argument or index.**
  - An UPPER env name passes whatever it names, so `os.getenv('MY_APP_CS')` (which rev6 began masking) and the reviewer's `os.getenv("X") or os.getenv("Y")` are kept.
  - In a call, a lowercase key needs an underscore, so `SecretStr("secret")` and `SecretStr("supersecretvalue")` are masked. A lowercase key in an index needs none.
- **Function words.** A word of 8 or more characters with five consonants in a row (`y` counts as a vowel) is token-shaped. This masks probe14's two rows and probe12's `see xkqzvbnmplwrtd`; `see HTTPS docs` is pinned as kept.
- **Caches.** `_words` is uncached. Labels go through the cached `_label_words`, and `_strict_label` is cached on label text only. `test_masking_never_caches_value_text` warms the label caches, masks new values, and asserts that no `lru_cache` in `log_sanitizer` or `trajectory_export` grew and that `_words` has no cache.
- **Tables.**
  - `_MUST_MASK` has 205 rows: the 4 fallbacks, 2 keyword, 2 concatenation, 3 function-word, and rows for the rule's own parts.
  - `_MUST_KEEP` has 78: the reviewer's three plus `MY_APP_CS`, `default=None`, a dict entry, a keyword argument before a field, and `HTTPS`.
  - `test_code_next_to_a_secret_is_masked_whole` pins 5 exact outputs: keyword default, keyword value, `SecretStr("secret")`, nested call, and fallback then statement.
  - Against the rev6 source, 22 nodes were red.
- **Adapted probe9** (`$FU/r15tr-probe9-r7.py`): 47 loosenings, with fragments updated and 11 new for rev7. Baseline is 0 red, and every loosening turns at least one must-mask row red. It now also prints keep breaks per loosening.
  - "Code read one argument at a time" was dropped. Since rev7 an unparsed call is masked whole, so multi-argument parsing only keeps look-alikes; reverting it breaks 2 must-keep rows and no must-mask row.
  - The 2 KB expression cap is a cost guard, not a mask rule. It is pinned by the new `kept-code-quoted-keys` adversarial text.
- **Probes.**
  - probe14: 0 kept secrets, and every hunt row is masked whole with no tail. Two lines need explaining:
    - It prints `KEPT` for `SecretStr("secret")` only because `secret` is part of the label `client_secret`. The output is `client_secret=[credential redacted]`, pinned exactly.
    - Its cache section raises an AttributeError, because `_words` has no `cache_info` by design; the new test covers that check.
  - probe14 over-masking: head keeps all but `see RFC6749 section 2.3.1`, which rev6 masked too.
  - probe15: the chain/plain ratio is 0.9 and 0.8 at head (rev6 5.9 and 8.5).
  - probe13: rev5 vs head 0.29/0.28 s, 0.25/0.26 s and 0.33/0.32 s. Value words are split only for compound names.
  - probe12: no swallowed fields. It still prints KEPT for the two `?key=` lowercase words and `pass=`.
  - probe10: KEPT only for the three declined rows. probe8: 0 kept.
  - probe2, probe_labels, probe3-7 and probe11 give the same results as rev6.
  - `probe_equiv` (rev7 corpus 4719): 0 diffs.
  - Adversarial: the slowest case is 0.32 s CPU (quoted-prose-1mb); the new code texts take at most 0.14 s (load 44).
- **Rev7 evidence:**
  - Covering run (`-n 4`, 39 files with UI, started at load 32): 31 failed, 10 errors and 1777 passed in 28 min.
    - At about 11:08 UTC the load spiked past 500 and macOS refused `~/Documents` reads ("Operation not permitted") on `.venv` and worktree files until 14:17 UTC.
    - Every failure and error beyond the 8 base failures was in four files that run subprocesses or time keystrokes: `Tests/Packaging/test_installed_distribution.py`, `Tests/Packaging/test_bug_report_import_deferral.py`, `Tests/Packaging/test_rag_boot_import_closure.py` and `Tests/Performance/test_console_keystroke_work_census.py`. They showed `PermissionError` or subprocess `init_import_site` failures.
  - Rerun of those four files once access was back (`-n 4`, load 50-155): 3 failed, 247 passed in 42 min, with no `PermissionError`.
    - The 3 failures are base failures from the known 8: `test_installed_wheel_loaders_entry_points_and_assets_are_immutable`, `test_profile_driven_rag_still_works_from_the_deferred_boot_state` and `test_search_modes_is_import_cheap_and_single_sourced`.
    - So the covering set fails only its 8 base failures.
  - Serial with lane flags: the recovered-suites slice (19 targets) gives 577 passed in 117 s at load 34; `Tests/Utils/test_log_sanitizer.py` gives 209 passed.
  - `Tests/CI` gives 1640 passed, 1 skipped (load 35). The module ratchet gives 207 passed, and the ruff gate is OK.
  - `./scripts/preflight.sh` passes (rc 0, load 34-37).

**Review r15tr-rev8.**
- **The ruling.**
  1. **The expression rule runs for every value.**
     - In `mask_credential_labels`, every credential-labelled value that is not prose (a function word before words) or escape-quoted is judged with the expression it starts. A quoted value's span begins at its opening quote.
     - The value is judged as before. What follows it is checked for literals with `_code_literals_look_alike` and for random-looking words outside quotes.
     - When it fails, the whole span is masked, quotes included.
     - A lone value, followed directly by a separator, closer, line end or field and with no bracket of its own, takes a fast path and is judged exactly as before.
     - This masks `"{}".format(...)`, `"%s" % ...`, `+` and implicit concatenation, `"" or ...`, `"" if dev else ...`, dict, list and tuple values, and lambdas.
  2. **Multiline.** `_expression_end` treats a line break at depth 0 as the end, runs over line breaks while a bracket is open, and keeps the 2 KB guard. Past it, the span runs to the end of the line the limit falls on. Both black-formatted calls are must-mask rows and exact-output pins (`client_secret = [credential redacted]`, and the next line stays visible).
  3. **`==`.** The label separator is `[ \t]*(?::|=(?!=))[ \t]*`.
     - A label right after `==`, `!=`, `<=` or `>=` is a comparison's operand, not a field. Without this, `other_secret: return` in `if client_secret == other_secret: return` was still masked and its import refused.
     - I did not have the reviewer's four comparison rows verbatim, so I built four from probe16's line: the operand line, `== ""`, `assert ... == "client_secret"` and `return ... == settings.CLIENT_SECRET`. I added `!=`, `<=` and `>=` rows too. All seven are in `_MUST_KEEP`, so each is kept and imports.
- **Additions:**
  - An UPPER env name that is 12+ letters with five consonants in a row is not a name (`SecretStr("XKQZVBNMPLWRTDXK")`, probe16).
  - The function-word letter rule now needs 12+ letters, so `see strengths of each option` (rev7 masked it) is kept and pinned.
  - A quoted key (`"user":`) ends an expression, as an unquoted field does. Otherwise a space-separated `"client_secret": "a b" "user": ...` line was masked whole: probe11's quoted-prose 4.6 MB came out as 40 bytes before this fix. Two must-keep rows pin it.
  - Unparsed calls stay masked whole. I tried judging their callee instead, but inside JSON text the closing quote of the enclosing string reads as an opening quote, and `refresh_token=tokens[0]` and `secret_key=base64.b64encode(os.urandom(32))` then leaked into the serialized bundle.
- **Unchanged look-alikes.** All of `_MUST_KEEP` passes, including the seven README rows. `probe_equiv` (rev8 corpus 4794) gives 0 diffs.
- **Tables.**
  - `_MUST_MASK` has 226 rows, all 21 probe16 hunt rows among them. `_MUST_KEEP` has 88 rows. `test_code_next_to_a_secret_is_masked_whole` has 7 pins.
  - Against the rev7 source, 24 nodes were red: 14 mask, 8 keep, 2 exact.
- **Adapted probe9** (`$FU/r15tr-probe9-r8.py`): 51 loosenings, with fragments updated and new ones for rev8. Baseline is 0 red, and every loosening turns at least one must-mask row red.
  - "expression rule only for non-quoted values" turns 6 rows red; "no multiline expressions" 2; "lone fast path ignores open brackets" 2; "env names may be random letters" 1.
  - The keep rules are pinned by must-keep rows, checked by reverting each: quoted-key fields (1 row), `==` as a separator (4), comparison operands as labels (4), random letters from 8 (1).
- **Probes.**
  - probe16: 0 kept of 21, each masked whole.
    - Over-masking: head keeps 11 of its 13 lines. The two masked are `Required for confidential clients`, which rev7 masked too, and `cfg.get("token")`, masked since rev7.
    - Cost: nested kept spans 0.01 s at 0.21 MB; dotted-head chain 0.37 s at 2 MB.
  - probe14: 0 kept; its `SecretStr("secret")` KEPT line is the label-substring artifact.
  - probe12: no swallowed fields.
  - probe10: KEPT only for the three declined rows. probe8: 0 kept.
  - probe2, probe_labels, probe3-7: as before.
  - probe11: ratio 4.0, passes. At MB scale, escaped 8.0 MB takes 1.39 s and quoted-prose 9.2 MB 1.66 s. Mixed-unclosed 6.8 MB takes 2.55 s against rev7's 1.76 s, because its quotes pair across labels, so each value has a tail to judge. It is linear.
  - probe13 (rev5/head): 0.31/0.32 s, 0.26/0.30 s and 0.35/0.37 s.
  - probe15 chain/plain: 1.2 and 1.0 (rev6 5.8 and 8.3).
  - Adversarial: the slowest case is 0.34 s CPU (load 57).
- **Rev8 evidence:**
  - Covering run (`-n 4`, 39 files with UI, load 36-100, no access loss): 9 failed, 1842 passed in 40 min. That is the 8 base failures plus `Tests/Packaging/test_bug_report_import_deferral.py::test_report_imports_and_no_console_open_keep_heavy_readers_deferred`.
    - That test failed inside its subprocess on `host.screen._capture is not None` after `wait_for_complete()`, a timing check.
    - Run alone, it passed 3/3 at head and 3/3 in `fu-r15-base` (cc6006fcf5), and it passed in the rev7 rerun. So it is a load flake.
  - Serial with lane flags: the recovered-suites slice (19 targets) gives 610 passed in 202 s at load 48-63; `Tests/Utils/test_log_sanitizer.py` gives 209 passed.
  - `Tests/CI` gives 1640 passed, 1 skipped (load 65-100). The module ratchet gives 207 passed, and the ruff gate is OK.
  - `./scripts/preflight.sh` passes (rc 0, load 42-46).
