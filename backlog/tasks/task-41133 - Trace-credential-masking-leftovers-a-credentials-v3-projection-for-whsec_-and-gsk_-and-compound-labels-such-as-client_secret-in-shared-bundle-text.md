---
id: TASK-41133
title: "Trace credential masking leftovers: a credentials-v3 projection for whsec_ and gsk_, and compound labels such as client_secret= in shared bundle text"
status: Done
created_date: 2026-11-15 02:20
updated_date: 2026-11-17 04:24
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

Known limits, after review r15tr-rev2. Under a compound credential label the bundle still keeps:
- a value with a digit but under 6 characters (`hu3er`);
- a value with no digit, unless it has mixed case and is 16+ characters (all-lowercase or all-uppercase letters, hex without digits, `AbCdEfGhIjKlMnO`);
- a purely numeric or date-shaped value (`12345678`, `2024-01-01`), or one holding a time (`10:00`);
- a value starting with `./`, `../`, `~/`, `$`, `<`, `{`, `%`, `:`, `*`, `[` or `(` (bcrypt `$2b$...`, `$ecret...`), or a lowercase absolute path (`/etc/ssl/x`);
- a value that starts like a call or index (`P4ssw0rd(1)`), or like a URL;
- a dotted name whose segments are identifiers of up to 20 characters;
- a lowercase identifier whose segments end in digits (`hunter2`, `secret1`, `client_secret_2`), and a run of capitalised words (`YourClientSecretHere`);
- a value over 2048 characters, a quoted value split by a line break, and JSON-escaped quotes (`\"client_secret\": ...`, which logs do not mask either).

Simple labels (`api_key=`, `token:`) stay with the export's older rule, and `?key=` in a URL is not masked. A UUID under a credential label (`session_token=550e8400-...`) is masked, because a session token's value is a credential.

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
