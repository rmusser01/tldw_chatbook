# TASK-208: Active Ingest Source Admission Design

## Purpose

Library should prevent an accidental second import of a source that is already
queued or running without taking away deliberate re-ingestion. The first Start
press refuses the duplicate and explains why in the existing inline gate. A
second deliberate press queues the duplicate as an explicit one-shot override.

This is active-job admission control. It is not a historical uniqueness rule,
an existing-Library content match, an overwrite policy, or a resume mechanism.

## Existing foundation

The current implementation already has several adjacent but distinct concepts:

- `LibraryIngestJobRegistry` owns the UI-thread job projection and records each
  job's source, backend origin, and lifecycle state.
- Local and Server submissions share `submit_library_ingest_job`, which expands
  folders before choosing the backend and routes accepted work to the local
  queue or remote service.
- background preflight forecasts some text files that already exist in the
  Library by content hash;
- the writer can return an `Already in Library` terminal receipt;
- `overwrite_existing` controls historical persistence behavior;
- the Start action already uses an inline two-press confirmation with a 300 ms
  dead zone for preflight warnings.

None of these prevents the same source from producing multiple active jobs.
TASK-208 fills only that gap and reuses the established consent grammar.

## Approaches considered

### 1. App-boundary active admission with an explicit override — selected

A pure registry query finds active jobs with the same canonical source and
backend. The screen uses the result to render the first-press confirmation, and
the app repeats the check immediately before local queue creation or remote
submission. A one-shot `allow_active_duplicate` flag carries the second press.

This approach gives the UI enough information to be honest while retaining an
authoritative non-UI guard for every entry point. It adds no database schema,
hashing, network work, or durable state.

### 2. Make registry submission silently return the existing job — rejected

This would be small internally but would make a refused request look like a
successful new submission. The form could clear, the queue could scroll, and
resource ownership could transfer even though no job was created. Callers also
could not offer an intentional override without reverse-engineering the result.

### 3. Add persistent idempotency keys or all-history content hashes — rejected

Persistent uniqueness would conflate active-job safety with historical media
identity, overlap the existing content-match/overwrite flow, require schema and
cleanup policy, and block legitimate re-ingestion of a changed source. Content
hashing also introduces filesystem/database work at an admission seam that
must remain cheap and deterministic.

## Binding product behavior

### Admission scope

Only `QUEUED`, `PARSING`, and `WRITING` jobs participate. `DONE`, `FAILED`,
`SKIPPED`, and `CANCELLED` jobs never block a new submission. Superseded or
dismissed rows cannot be active and remain excluded.

The key is `(backend origin, canonical source)`. Local and Server are separate
scopes: importing a source locally must not prevent sending the same source to
Server, and the reverse is also true.

The guard is default-on and has no persistent preference or checkbox. Its only
override is the one confirmed submission produced by the second Start press.

### Canonical source identity

Canonicalization is lexical and side-effect free:

- trim surrounding whitespace after the existing form validation;
- for filesystem paths, expand `~`, make the path absolute against the current
  application working directory, normalize separators and dot segments, and
  apply the platform's case normalization (`normcase` makes Windows matching
  case-insensitive while preserving case-sensitive behavior elsewhere);
- do not resolve symlinks, stat files, read content, query SQLite, or access the
  network;
- for HTTP(S), lowercase the scheme and host, omit a default `:80`/`:443`, treat
  an empty path as `/`, and remove the fragment because it is not part of the
  fetched resource;
- preserve URL path bytes and query ordering exactly. Do not sort query
  parameters, decode/re-encode percent escapes, strip tracking parameters, or
  infer redirects.

The public normalizer returns an immutable key suitable only for active
admission comparison. It is not a media identity or storage key.

### Folder submissions

Folder submission stays one user action. The existing bounded collector expands
the folder before admission. All members are compared against the active
registry before any member is appended or sent. If one or more members match,
the first press queues none and the inline copy states how many selected files
are already active. The confirmed second press admits the original batch using
its existing grouping and per-file behavior.

This all-or-nothing admission rule avoids a half-submitted folder whose form and
receipt cannot explain which members were silently omitted.

## Component design

### Pure source-key and registry query

`tldw_chatbook/Library/library_ingest_jobs.py` owns:

- a small immutable active-source key;
- the lexical normalizer;
- the single definition of active admission states; and
- a registry query that receives one or more source strings plus an origin and
  returns copies of matching active jobs in registry order.

The query does not mutate, notify, persist, allocate job IDs, or call the
filesystem. Keeping it beside the lifecycle model prevents the active-state
definition from drifting between screen and app.

### Authoritative app admission

`LibraryIngestQueueMixin` owns source expansion and backend routing, so it owns
the authoritative admission operation:

1. resolve the target backend;
2. expand a directory with the existing bounded collector;
3. query active matches using that backend origin;
4. if matches exist and the one-shot override is false, raise a typed expected
   refusal containing safe job copies;
5. otherwise continue through the unchanged local or Server submission path.

The check occurs before the first job append and before any remote service call.
The typed refusal is control flow, not a product failure: it creates no failed
job, no generic error receipt, and no error-level diagnostic.

The screen's read-only preview never repeats directory I/O on the Textual thread.
For a single source it queries the registry directly; for a folder it compares
the candidate paths already captured by the current background preflight result.
If that snapshot is missing or stale, preview may conservatively find no match,
but it cannot authorize the submission: the app's binding check still runs after
the existing bounded submit-time expansion. A typed refusal from that check is
converted into the same inline consent state. Future callers, keyboard entry
points, and a changed registry therefore cannot bypass the guard.

### Screen consent state

The existing Start confirmation becomes a union of consent reasons instead of
two serial confirmations. Its fingerprint contains:

- resolved source;
- backend origin;
- the form/options snapshot relevant to submission;
- preflight-warning identity; and
- the matching active job IDs and states.

On the first press, any owed consent arms the gate and returns without calling
the submit path. Duplicate-only copy is:

`This source is already being imported. Press Start again to queue another copy.`

Folder copy names the count:

`2 selected files are already being imported. Press Start again to queue another copy.`

When duplicate and tooling-warning reasons coexist, one combined inline sentence
names both risks and one second press accepts both. The user never owes a third
press.

The same 300 ms dead zone rejects double-clicks and key repeat. Button, Enter in
the path field, and any existing Start accelerator continue through the same
method. Arming changes only the gate line in place, preserving focus and scroll.

The consent disarms when its fingerprint can no longer describe the request:
source or option edits, backend changes, preflight replacement/invalidation,
active-match changes, canvas reset/exit, or Escape. The override is not stored in
form/config state and is consumed by one call only.

### External resource ownership

Admission preview occurs before external Parakeet preparation whenever the
current source can be checked immediately. The authoritative app guard still
runs after preparation. If that late guard refuses admission, the existing
scope-release path treats it as “no job created,” releases the retained scope,
preserves the form, and arms or refreshes duplicate consent instead of showing a
generic failure.

No retained model scope transfers to a duplicate job until admission succeeds.

## Error and race handling

- Malformed input continues to fail existing path/URL validation before
  canonicalization.
- Canonicalization failure returns no match during preview but cannot authorize
  submission: the authoritative submit path still performs existing validation
  and its own guarded comparison.
- Expected active-duplicate refusal never clears the form, persists option
  defaults, updates last-submission state, scrolls to a new receipt, or produces
  a failed job.
- If an active job becomes terminal after the first press, the changed match
  fingerprint disarms consent; the next press submits normally without requiring
  an obsolete override.
- If a matching active job appears after preview, the authoritative guard refuses
  and arms the current request rather than creating a duplicate silently.
- Registry mutation remains UI-thread-only, so the comparison and first append
  are atomic with respect to other registry mutations within one submit call.

## Testing strategy

### Pure contract tests

- Windows-equivalent path spellings match under a patched platform normalizer;
- case-sensitive platforms retain path case;
- relative/absolute and dot-segment spellings normalize consistently;
- conservative URL equivalences match while distinct paths, queries, or
  non-default ports remain distinct;
- fragments do not affect active identity;
- only QUEUED/PARSING/WRITING jobs match;
- backend origin partitions matches; and
- returned jobs cannot mutate registry-owned state.

### App/coordinator tests

- single-file Local and Server duplicate attempts are refused before side
  effects;
- terminal history permits re-ingestion;
- folder matching is all-or-nothing;
- the override admits the unchanged original batch once;
- direct/non-screen callers cannot bypass the guard;
- no local job ID, remote call, failed receipt, or queue runner start occurs on
  refusal; and
- a refused external-model submission releases its retained scope.

### Screen tests

- first press preserves every form field and arms the inline gate;
- second press outside the dead zone submits exactly once with the override;
- double-click/key repeat does not submit;
- Enter follows button semantics;
- simultaneous tooling and duplicate warnings need two presses total;
- source/options/backend/active-match changes and canvas exit disarm consent;
- duplicate and folder copy is exact, markup-safe, and non-modal; and
- focus, cursor, and scroll survive the in-place gate update.

Focused modules will run with repository-local pytest temp roots. Any established
Windows Proactor/network-guard or symlink-privilege limitation will be separated
from product assertions and reported with a scoped equivalent, never counted as
a product pass.

## Documentation and architecture

ADR-065 records the long-lived admission scope, identity rules, batch atomicity,
and one-shot override policy. ADR-014 remains authoritative for Library ingest
service ownership and recovery; ADR-065 narrows one submission-admission rule
without changing lifecycle authority or persistence.

The Library user guide will explain the first-press refusal and intentional
second-press override. No schema migration, runtime dependency, new setting, or
historical cleanup is part of TASK-208.

## Acceptance boundary

TASK-208 is complete when active duplicate submission is safely blocked and
explicitly overridable across Local, Server, single-file, folder, keyboard, and
external-resource paths. Historical content deduplication, content-hash fallback,
server-side idempotency keys, redirect-aware URL identity, and a durable “never
import twice” preference remain out of scope.
