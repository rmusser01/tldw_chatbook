# Active Credential Redaction and Display Validation (TASK-856)

**Status:** Approved in conversation; pending written-spec review

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

String redaction uses an ordered set of precompiled patterns without nested
ambiguous quantifiers. Each pattern replaces a complete value; partial
credential fragments must not be left behind.

The normative matching contract is:

1. A candidate scalar assignment has an unquoted label matching
   `[A-Za-z0-9_.-]+` or that same label surrounded by one matching pair of
   single or double quotes. Optional ASCII horizontal whitespace may surround
   a `:` or `=` separator. The label starts at the beginning of the string or
   after a character outside `[A-Za-z0-9_.-]`, so matching cannot begin halfway
   through a larger label. The label is unquoted and passed to the same private
   log-field classifier used by `sanitize_dict()`. Non-sensitive labels leave
   the complete match unchanged.
2. A quoted scalar value starts with `'` or `"` and consumes through the same
   unescaped quote. Backslash plus the following character is part of the value
   and cannot close it. The quotes are preserved and only their contents are
   replaced. If a sensitive label starts a quoted value without a closing
   quote, sanitization fails closed through the first CR, LF, mapping delimiter
   (`,`/`}`/`]`), query delimiter (`&`/`#`), or end of string.
3. An ordinary unquoted scalar value consumes at least one character and stops
   before ASCII whitespace, a quote, `,`, `;`, `}`, `]`, `&`, `#`, CR, LF, or
   end of string. These terminators are preserved. This covers environment
   assignments and URL query parameters without consuming the next field.
4. For the exact normalized protocol labels `authorization`,
   `proxy_authorization`, `cookie`, and `set_cookie`, an unquoted value may
   contain horizontal whitespace and semicolons. It stops before `,`, `}`, `]`,
   CR, LF, or end of string. This consumes a complete `Bearer value`,
   `Basic value`, or cookie header rather than redacting only its first word.
   Quoted mapping values still follow rule 2 and therefore do not consume an
   adjacent mapping entry.
5. Assignment matching is for scalar text only. Serialized nested containers
   are not parsed with regex; callers holding structured data use
   `sanitize_dict()`/`sanitize_list()` so a sensitive container value is
   replaced as a unit.
6. Independent Basic and Bearer scheme matches consume the scheme plus its
   following non-whitespace credential when those values appear without a
   label. The scheme may be retained, but the complete credential is replaced.
7. HTTP(S) URL userinfo matches from immediately after `://` through the last
   `@` before the next `/`, `?`, `#`, ASCII whitespace, CR, or LF. The userinfo
   is replaced as a unit; scheme and authority remain only when the caller is
   otherwise allowed to render the URL.
8. Standalone recognizable tokens use these exact families, evaluated from
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

For assignment rules, surrounding label syntax, separators, quotes, and the
terminating delimiter are preserved; only the scalar value becomes
`***REDACTED***`. Tests instantiate every sensitive provider label derived from
`CONFIG_TOML_CONTENT` and `DEFAULT_APP_TTS_CONFIG` in both a structured mapping
and representative quoted/unquoted assignment text. Therefore the structured
classifier and independent string parser cannot drift while still satisfying
AC #3.

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
  -> structured log-field classification or ordered string rules
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
- Regexes avoid nested ambiguous quantifiers and are exercised with long input
  to guard against pathological backtracking.
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
- no leakage of fixed sentinel values or credential fragments;
- preservation of `claude-*`, `max_tokens`, `api_key_env_var`, and ordinary
  identifiers;
- config-derived coverage by parsing `CONFIG_TOML_CONTENT` and
  `DEFAULT_APP_TTS_CONFIG`, constructing a structure from the actual sensitive
  key names, and asserting that every sentinel value is removed;
- log-specific fields not owned by the config predicate;
- nested dictionaries/lists, non-string keys, new-container identity,
  `deep=False`, non-string fallback, formatting fallback, and idempotence; and
- bounded behavior on long non-matching input.

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
checks; the persistent-diagnostic inventory gate if the diagnostic call shape
changes its reviewed digest; `git diff --check`; and a broader relevant suite
selected from the final diff. Any unrelated full-suite baseline failures are
reported separately and never represented as TASK-856 passes.

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
