# Active Credential Redaction and Display Validation (TASK-856)

**Status:** Implemented and verified

**Task:** TASK-856

## Context

`tldw_chatbook.Utils.log_sanitizer` was originally recorded as dead code. That
is no longer true on current `dev`. It has three production consumers:

1. Ollama renders successful API payloads through `sanitize_dict()` and also
   passes model names through `sanitize_string()` before displaying them.
2. Transformers passes locally discovered model names through
   `sanitize_string()` before displaying them.
3. The subscription monitor passes a monitored URL through `sanitize_string()`
   before including the URL in a snapshot-pruning debug message.

The active sanitizer is incorrect in both directions. It classifies every
`claude-*` model ID as an Anthropic credential while failing to consume the
hyphenated portions of real-shaped `sk-ant-api03-*` and `sk-proj-*`
credentials. Its private `SENSITIVE_FIELDS` set also disagrees with the
canonical `is_sensitive_config_key()` predicate introduced by TASK-852 and
misses real shipped configuration names.

The three consumers expose a boundary error as well as regex defects:
credential redaction, untrusted display-name validation, and metadata-only
diagnostics are different operations. A single function must not be used as a
generic notion of "safe text."

## Goals

- Keep the sanitizer's existing public function imports stable.
- Reliably remove complete credential values from structured and labeled text.
- Avoid treating model identifiers or arbitrary opaque strings as credentials.
- Make display-name and diagnostic consumers use the boundary appropriate to
  their purpose.
- Preserve the sanitizer's recursive, non-mutating container contracts.
- Verify the installed package, not only the source checkout.

## Non-goals

- This task does not install a process-wide logging filter.
- It does not authorize payloads, URLs, or other private values to be written to
  persistent diagnostics merely because a redaction helper exists.
- It does not attempt to identify every unlabeled opaque provider token. Some
  providers issue values with no stable syntax that distinguishes them from an
  ordinary identifier.
- It does not rename the module or its existing public functions.
- It does not change provider configuration, credential storage, encryption,
  or application logging topology.

## Decision

Keep `Utils/log_sanitizer.py` as an explicit credential-redaction utility, but
narrow its claimed purpose. It is defense-in-depth for strings and structures
that a caller has already decided may be rendered. It is not a global privacy
policy and it is not a substitute for omitting private data at a diagnostic
boundary.

### Public compatibility

The following functions remain importable under their current names:

- `sanitize_string`
- `sanitize_dict`
- `sanitize_list`
- `sanitize_log_params`
- `create_safe_log_message`
- `safe_log`

The public container behavior remains:

- sanitization produces new dictionaries and lists rather than mutating input;
- a sensitive dictionary key replaces its entire value, including a container
  value, with the marker;
- `deep=True` recursively handles nested dictionaries and lists;
- `deep=False` leaves nested containers untouched but still sanitizes direct
  string values/items, matching the current contract; the returned outer
  dictionary/list is new while each untouched nested container retains its
  original identity;
- non-string dictionary keys do not raise;
- non-string input to `sanitize_string()` retains the current `str()` fallback;
- formatting failure in `create_safe_log_message()` retains the current safe
  fallback to the sanitized template; and
- `safe_log()` retains its one-rendered-message callback behavior.

All redactions use the single neutral marker `***REDACTED***`. Provider-specific
markers are removed because they add no protection and make sequential rule
behavior harder to reason about.

### Structured secret classification

`sanitize_dict()` must not maintain another copy of the app's config-key
inventory. A private log-field classifier will delegate config-shaped names to
`Utils.sensitive_config_keys.is_sensitive_config_key()` and add one exact set
of log/protocol names that are outside that predicate's config contract.
Log-only names are normalized with
`str(key).strip().lower().replace("-", "_")` and compared to:

```text
authorization
proxy_authorization
cookie
set_cookie
credential
credentials
database_url
connection_string
dsn
```

There are no substring or suffix rules for these log-only names. A future name
requires an explicit contract and regression rather than silently broadening
the set. Config-shaped names are passed to the canonical predicate in their
original spelling, so that predicate continues to own its `_env_var` exclusion
and other rules. This composition is intentional: shipped configuration names
belong to the canonical predicate, while the sanitizer owns protocol fields
that may appear in response or header dictionaries. Extending the config
predicate with HTTP-only concepts would incorrectly change encryption and
Settings privacy-posture behavior.

When a structured key is sensitive, its complete value is replaced before any
recursive traversal. Otherwise nested dictionaries/lists follow the existing
`deep` contract and direct strings pass through `sanitize_string()`.

### String redaction

String redaction combines a deterministic assignment scanner with a small
ordered set of precompiled standalone patterns. The scanner classifies a label
before deciding how much value text to consume. This ordering is load-bearing:
a regex that first consumes a non-sensitive assignment such as
`max_tokens=42 api_key=PRIVATE_SENTINEL` could skip the later sensitive label,
while a rule that stops every unquoted value at whitespace could expose the
remainder of a multi-word password.

The scanner records non-overlapping replacement spans and constructs the output
once. It does not repeatedly slice and concatenate the whole input. Standalone
patterns contain no nested ambiguous quantifiers. Label, HTTP scheme, URL
scheme, and Bearer matching is case-insensitive. Standalone credential-family
prefixes are case-sensitive (`sk-`, `sk-proj-`, `sk-ant-api03-`, and `AIza`)
to avoid expanding their false-positive surface.

The normative matching contract is:

1. A precompiled candidate-prefix pattern finds an unquoted label matching
   `[A-Za-z0-9_.-]+` or that same label surrounded by one matching pair of
   single or double quotes. Optional ASCII horizontal whitespace may surround
   a `:` or `=` separator. The label starts at the beginning of the string or
   after a character outside `[A-Za-z0-9_.-]`, so matching cannot begin halfway
   through a larger label. The label is unquoted and passed to the same private
   log-field classifier used by `sanitize_dict()`.
2. A non-sensitive candidate consumes no value text. Scanning resumes at the
   end of its separator, so a later sensitive assignment—including one nested
   inside a quoted descriptive value—remains discoverable.
3. For a sensitive candidate, a quoted scalar value starts with `'` or `"` and
   consumes through the same unescaped quote. Backslash plus the following
   character is part of the value and cannot close it. The quotes are preserved
   and only their contents are replaced. If a sensitive label starts a quoted
   value without a closing quote, sanitization fails closed through the first
   CR, LF, or end of string.
4. For a sensitive candidate, an unquoted scalar value consumes from the first
   non-horizontal-whitespace character through the first CR, LF, or end of
   string. Spaces, quotes, commas, semicolons, mapping punctuation, query
   separators, and fragments inside that span are redacted too. This
   deliberately prefers over-redacting the remainder of one diagnostic line to
   leaking an unquoted multi-word or punctuation-bearing credential. Callers
   that need adjacent fields preserved must pass structured data or quote the
   scalar value.
5. If a sensitive candidate has no value before CR, LF, or end of string, it is
   left unchanged; an empty assignment contains no credential bytes to expose.
6. Assignment matching is for scalar text only. Serialized nested containers
   are not parsed with regex; callers holding structured data use
   `sanitize_dict()`/`sanitize_list()` so a sensitive container value is
   replaced as a unit.
7. An independent Bearer match begins at the start of the string or after a
   character outside `[A-Za-z0-9_-]`, requires one or more whitespace
   characters after `Bearer`, and consumes the following non-whitespace
   credential. The scheme is retained and the complete credential is replaced.
   Matching cannot begin inside `NotBearer` or `not-bearer`. `Basic` is
   recognized only as part of a sensitive authorization assignment/header;
   treating the common English adjective as an independent scheme would
   corrupt ordinary log prose.
8. HTTP(S) URL userinfo matches from immediately after `://` through the last
   `@` before the next `/`, `?`, `#`, ASCII whitespace, CR, or LF. The userinfo
   is replaced as a unit; scheme and authority remain only when the caller is
   otherwise allowed to render the URL.
9. Standalone recognizable tokens use these exact families, evaluated from
   most specific to least specific:

   ```text
   sk-proj-     + at least 20 characters from [A-Za-z0-9_-]
   sk-ant-api03- + at least 20 characters from [A-Za-z0-9_-]
   sk-          + at least 20 characters from [A-Za-z0-9]
   AIza         + exactly 35 characters from [A-Za-z0-9_-]
   ```

   A standalone match requires both the preceding and following character, if
   present, not to be in `[A-Za-z0-9_-]`. The specific `sk-proj-` and
   `sk-ant-api03-` rules consume the maximal allowed run before the legacy
   `sk-` rule is considered. This prevents partial `sk-proj`/`sk-ant` matches.
   No rule recognizes `claude-*`.

Scanner progress is explicit and monotonic:

- after a non-sensitive candidate, the cursor resumes at the end of its
  separator;
- after a closed quoted sensitive value, it resumes immediately after the
  closing quote;
- after an empty sensitive assignment, it resumes after the CR/LF when present
  or terminates at end of string;
- after an unterminated quoted or unquoted sensitive value, it resumes after
  the CR/LF that bounded the replacement or terminates at end of string; and
- every iteration advances at least past a separator or line boundary, and
  scanning continues until no candidate remains.

This permits multiple quoted sensitive assignments on one line and sensitive
assignments on later lines to be redacted independently. An unquoted sensitive
assignment intentionally consumes the rest of its own line, so later values on
that line are removed as part of the same fail-closed replacement rather than
individually parsed.

After the assignment scanner builds its output, standalone transformations run
against that output in this exact order: URL userinfo, independent Bearer, then
the case-sensitive credential families from rule 9. They do not participate in
the assignment span set. Consequently a standalone-shaped token already
removed as part of a labeled value cannot create an overlapping replacement;
subsequent passes only see `***REDACTED***`.

For quoted assignment values, surrounding label syntax, separators, and quotes
are preserved; only the scalar contents become `***REDACTED***`. For unquoted
values, the label and separator are preserved and the remainder of the line is
replaced. Tests instantiate every sensitive provider label derived from
`CONFIG_TOML_CONTENT` and `DEFAULT_APP_TTS_CONFIG` in both a structured mapping
and representative quoted/unquoted assignment text. They also place a
non-sensitive assignment before a sensitive one on the same line. Therefore
the structured classifier and independent string scanner cannot drift or skip
a later secret while still satisfying AC #3.

Ambiguous provider keys are guaranteed to be redacted when accompanied by a
sensitive field name, environment-variable assignment, header, or structured
key. They are intentionally not guessed from arbitrary standalone alphanumeric
text. This replaces TASK-856's previous impossible requirement to recognize
every provider's opaque token without context.

The output must be idempotent: sanitizing an already sanitized string or
structure produces the same result.

### Consumer corrections

#### Ollama successful payloads

`_format_ollama_success_payload()` continues using `sanitize_dict()` before
rendering the bounded JSON shown in the current production UI. This is a true
credential-redaction boundary. The sanitizer does not make provider payloads
eligible for persistent logging; the output remains mounted-UI content only.

#### Ollama and Transformers model names

`_safe_ollama_model_names()` and `scan_transformers_local_models()` stop using
the log sanitizer. They use `Utils.input_validation.sanitize_string()` under an
explicitly named import, with the existing 256-character display bound, then
normalize retained tabs, carriage returns, newlines, and other whitespace to a
single display line.

This preserves legitimate names such as `claude-opus-4-20250514`. It also makes
the contract honest: these values are being validated for a RichLog display,
not prepared for a persistent diagnostic. Both paths continue to reject empty
or structurally invalid model entries and retain their existing result caps.

#### Subscription snapshot pruning

The snapshot-pruning diagnostic stops including the monitored URL. The message
retains the pruned count, subscription ID, and retention count, which are the
useful operational metadata. The dynamic log-sanitizer import is removed.

Using a sanitized full URL would still expose private paths and nonstandard
query values. Importing the private origin-only helper from
`Utils.egress` would create a cross-module dependency without adding useful
information to this message. Omission is the smallest boundary consistent with
ADR-029 and the precedent established by TASK-1722.

## Data flow

Credential-bearing renderable text follows:

```text
caller-approved string/structure
  -> structured log-field classification or assignment scanner/standalone rules
  -> complete values replaced with ***REDACTED***
  -> current UI or caller-owned logger callback
```

Model display text follows:

```text
provider/path model name
  -> bounded input validation
  -> single-line whitespace normalization
  -> current mounted RichLog
```

Subscription diagnostics follow:

```text
snapshot prune result
  -> retain count + subscription ID + retention count
  -> debug diagnostic (URL never enters the message)
```

## Error and safety behavior

- Redaction remains best-effort for strings but must not raise for ordinary
  supported input types.
- Sensitive structured fields fail closed by replacing the complete value,
  regardless of its type.
- The assignment scanner advances monotonically, builds output once, and parses
  quoted values without first searching the remaining line for a boundary.
  Long non-matching and dense quoted matched inputs deterministically exercise
  the intended single-pass bound; standalone regexes avoid nested ambiguous
  quantifiers.
- Redaction markers contain no original prefix or suffix from the secret.
- Formatting errors never fall back to interpolating raw arguments.
- Consumer display validation retains current invalid-result behavior rather
  than inventing replacement model names.
- No test or implementation logs live credentials; all values are conspicuous
  fixed sentinels or fake format-shaped strings.

## Verification strategy

Tests use direct production functions and the full production app only. No
reduced, fake, or test-only application is introduced.

### Sanitizer unit contract

Focused tests will prove:

- complete redaction of reproduced `sk-proj-*` and `sk-ant-api03-*` values;
- contextual redaction for opaque fake provider values;
- authorization, proxy-authorization, cookie, URL-userinfo, JSON-like,
  environment-style, and URL-query cases;
- unquoted multi-word/punctuation-bearing credentials, a non-sensitive
  assignment before a sensitive assignment, standalone `Bearer` boundaries,
  including hyphenated larger identifiers, and ordinary standalone `Basic`
  prose;
- multiple quoted sensitive assignments on one line, sensitive assignments on
  later lines, empty assignments followed by later secrets, and an unquoted
  sensitive assignment that fail-closes over the remainder of its line;
- a standalone-format token inside a labeled sensitive value, proving the
  assignment scan and subsequent standalone passes do not overlap and remain
  idempotent;
- no leakage of fixed sentinel values or credential fragments;
- preservation of `claude-*`, `max_tokens`, `api_key_env_var`, and ordinary
  identifiers;
- config-derived coverage by parsing `CONFIG_TOML_CONTENT` and
  `DEFAULT_APP_TTS_CONFIG`, constructing a structure from the actual sensitive
  key names, and asserting that every sentinel value is removed;
- log-specific fields not owned by the config predicate;
- nested dictionaries/lists, non-string keys, new-container identity,
  `deep=False`, non-string fallback, formatting fallback, and idempotence; and
- bounded behavior on long non-matching input; and
- exact redaction plus deterministic single-pass scan-work accounting on a long
  line containing many quoted sensitive assignments, without a wall-clock
  threshold.

Expected config key names come from the shipped configuration sources and the
canonical predicate, not from sanitizer constants. Tests also retain negative
sentinels for real environment-variable-name settings so a future change does
not redact or encrypt the name of an environment variable.

### Production consumer contract

Direct production helper tests prove that:

- Ollama payload rendering removes nested structured credentials;
- Ollama and Transformers display real `claude-*` model identifiers unchanged,
  bound length, and cannot inject a second display line;
- invalid model structures retain current failure behavior; and
- a real snapshot-pruning path emits its diagnostic without the monitored URL
  or any sentinel component from that URL.

The existing full-production-app LLM destination tests remain the integration
owner for mounted RichLog behavior. They are extended only where mounted
behavior adds coverage beyond the direct helper contract.

### Installed distribution

The existing installed-wheel probe imports the public log-sanitizer functions
from the isolated installed target and verifies both config-key delegation and
the `claude-*` false-positive regression. This catches a source-checkout-only
success or a missing packaged dependency without introducing another wheel
builder.

### Gates

Closeout runs the focused sanitizer, sensitive-key, LLM destination,
subscription-pruning, and installed-distribution tests; scoped Ruff and format
checks; `git diff --check`; and a broader relevant suite selected from the
final diff. Any unrelated full-suite baseline failures are reported separately
and never represented as TASK-856 passes.

The subscription diagnostic change necessarily changes the reviewed
`monitoring_engine.py` digest in
`Docs/security/production-diagnostic-inventory.json`.

Latest `dev` was rechecked at
`ceede62fe46d7aa090df4a36307077e097d8c044`. Its production changes did not
update the checked inventory: generated state has 467 owner files, 1,151
TASK-492 calls, 6,854 TASK-494 calls, and six sink files, while the committed
manifest records 466/1,144/6,851/6. The drift consists of one new owner file and
twenty changed existing owner entries; sink topology is unchanged. Semantic
call review confirms seven metadata-only robots.txt failure diagnostics, one
provider/model-only catalog diagnostic, one constant-message watchlist-star
diagnostic with exception context, and one reduced-motion card-name diagnostic.
The remaining owner digest changes are caused by moved or reformatted calls.
Persistent-sink topology is unchanged, and the existing admission tests remain
the proof that ordinary diagnostic records do not reach disk.

Implementation first patches only those reviewed pre-existing owner and
summary entries and commits that reconciliation separately, without running
the checker's blanket `--write` mode. The expected reconciled generated state
has non-monitoring SHA-256
`8fbd4266f14a51b9645626ba6f5ea624b00db65ac0baae0e4b98de1eaabc0fab`.
The checked and generated `monitoring_engine.py` entry remains TASK-494, 16
calls, digest `f9ccee6989b39da1333b` at that boundary.

TASK-856 then captures the reconciliation commit and generated base inventory,
and proves that every entry except `monitoring_engine.py` is identical at its
head. The monitoring entry must retain its owner, reason, and 16-call count;
only its digest may change. Closeout requires the checker and all three
architecture tests to be green.

After Tasks 1–3 were reviewed, `origin/dev` advanced to
`85a46bea8704d076fd6b544e56bead760fd3e9d9`. The rebase left every TASK-856
production/test file unchanged but exposed a second red upstream inventory
baseline: five metadata/constant-only STT executor diagnostics in `app.py`,
line-only digest changes in four related ingestion/library owners, and line-only
movement of three existing `app.py` sinks. The semantic diagnostic multiset and
sink-shape review found no additional payload/private-value sink. This drift is
reconciled in its own commit immediately before the Task 3 consumer commit.
The current boundary has `467/1151/6859/6`, monitoring digest
`f9ccee6989b39da1333b`, and non-monitoring fingerprint
`a927b4bc7a229d3c3328a5336054c410aabdedfe5fd40219ab1152a9880763eb`.
The Task 3 consumer commit retained that fingerprint and initially changed only
the monitoring digest to `3826b76482fd484ff194`. The subsequent scoped Ruff
format repair changed only the source-sensitive monitoring digest again, to the
current `911bf9d65817bf259923`; the owner, reason, 16-call count, sink topology,
and non-monitoring fingerprint remain unchanged.

After closeout, the branch rebased once more onto
`b030b0b73f217b955b298a45fce3a0256403447c`. The upstream Console rail
changes left every TASK-856 file and diagnostic call multiset unchanged but
moved calls in `chat_screen.py`, `settings_screen.py`, and `config.py`, plus the
existing private append sink line in `config.py`. A third reviewed inventory
reconciliation is placed immediately before the Task 3 consumer commit. Counts
remain `467/1151/6859/6`; the current non-monitoring fingerprint on both sides
of the TASK-856 monitoring change is
`5ce06a13eb48f8007eddfa92a0616b41e5122b89e6b2b7d494d4c81fb48723ac`.
Monitoring still changes only from `f9ccee6989b39da1333b` at that boundary to
`911bf9d65817bf259923` at the head.

## Architecture decision record

ADR required: yes

ADR path: `backlog/decisions/029-local-private-data-boundary.md`

Reason: TASK-856 changes behavior at a credential/privacy logging boundary, so
an ADR check is mandatory. ADR-029 already establishes that credentials and
private values are excluded from persistent diagnostics and that payload
redaction does not authorize payload logging. This task directly implements
that accepted boundary and does not introduce a new decision, so no duplicate
ADR is created.

## Alternatives considered

### Delete the sanitizer

Rejected because current `dev` has an active structured-payload redaction
consumer. Deletion would either expose credential values in the Ollama result
view or require an inline duplicate.

### Install a process-wide logging filter

Rejected because it is broader than the task, would interact with the accepted
persistent-sink policy, and could encourage callers to send private payloads to
logs under the assumption that the filter will repair them. ADR-029 requires
privacy at the producer boundary.

### Treat every provider-looking opaque token as a standalone secret

Rejected because opaque credentials are not distinguishable from ordinary
model IDs, content hashes, or user identifiers. Contextual field classification
provides reliable protection without destructive false positives.

### Extend the config-key predicate with HTTP header concepts

Rejected because that predicate is also used for configuration encryption and
Settings privacy posture. Authorization and cookie headers are valid log/payload
concepts but are not shipped configuration fields. The sanitizer composes the
canonical config predicate with its own small protocol-specific set instead.
