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
  string values/items, matching the current contract;
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
`Utils.sensitive_config_keys.is_sensitive_config_key()` and add a deliberately
small set of log/protocol fields that are outside that predicate's config
contract:

- authorization and proxy-authorization headers;
- cookie and set-cookie headers;
- credential containers;
- database URLs, connection strings, and DSNs.

The log-only comparison is case-insensitive and treats hyphen/underscore header
spellings equivalently. This composition is intentional: the canonical
predicate owns shipped configuration names, while the sanitizer owns protocol
fields that may appear in response or header dictionaries. Extending the
config predicate with HTTP-only concepts would incorrectly change encryption
and Settings privacy-posture behavior.

When a structured key is sensitive, its complete value is replaced before any
recursive traversal. Otherwise nested dictionaries/lists follow the existing
`deep` contract and direct strings pass through `sanitize_string()`.

### String redaction

String redaction uses an ordered set of precompiled, linear-time patterns. Each
pattern replaces a complete value; partial credential fragments must not be
left behind.

The supported categories are:

1. Candidate key/value assignments in environment, mapping-repr, JSON-like,
   header-like, and URL-query text. A replacement callback normalizes the
   candidate key and applies the same private log-field classifier used by
   `sanitize_dict()`. Non-sensitive labels such as `max_tokens` and
   `api_key_env_var` remain unchanged. Value matching stops at the appropriate
   quote, whitespace, mapping, query, or fragment delimiter so adjacent text is
   not swallowed.
2. Authorization schemes and authentication/cookie header values, including
   opaque Basic or Bearer values.
3. HTTP(S) URL userinfo. The full userinfo portion before `@` is removed; the
   scheme and host remain available only when the caller is otherwise allowed
   to render the URL.
4. Standalone credential families with a high-confidence, distinctive syntax,
   including the reproduced hyphenated OpenAI/Anthropic forms and existing
   Google-style form. Matching is bounded so a rule consumes the full token and
   does not reclassify `claude-*` model IDs.

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
