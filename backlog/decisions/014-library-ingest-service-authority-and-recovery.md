# ADR-014: Library Ingest Service Authority and Recovery

Status: Accepted
Date: 2026-07-09
Related Task: N/A (design decision recorded before implementation task selection)
Supersedes: ADR-013

## Decision

Library remains the owner of user-facing media import, but
`media_reading_scope_service` is the authority for submitted Local and Server
job lifecycles. The app owns authority-bound drafts and a read-only job
projection, heterogeneous confirmations are partitioned before submission, and
Local interruption plus Server receipt recovery are explicit contracts.

## Context

ADR-013 established Library ownership, a staged-source canvas, app-owned job
coordination, and Settings ownership for durable ingestion-source
administration. A second implementation-risk review found that the repository
already has a stronger service boundary than ADR-013 assumed:

- `media_reading_scope_service` exposes Local and Server submit, detail, list,
  observe, and cancel operations;
- Local submission persists `local_ingestion_jobs` but executes synchronously;
- Server submission returns remote batch/job identities and continues
  independently;
- server file upload is capability-gated through existing multipart seams;
- one backend submission carries one media type and one compatible processing
  profile.

Making an app registry authoritative would duplicate existing Local persistence
and Server authority. Treating a heterogeneous user confirmation as one backend
batch would also misrepresent the service contract. File-system and duplicate
preflight on the UI path could race the Start action or block the event loop.

The first design also promised Server rehydration without identifying durable
batch receipts. The current Server list API requires a known batch ID, and an
unscoped recent-event stream is not a durable history contract. Local rows can
survive process exit while their synchronous execution cannot, leaving stale
`queued` or `running` records unless recovery is explicit.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Keep ADR-013's app registry as submitted-job authority | It would create two sources of truth for Local rows and Server status, cancellation, and completion. |
| Submit every ready source as one backend batch | Mixed media types and per-source processing profiles cannot share one backend request safely. |
| Perform path, capability, and duplicate checks synchronously when adding | Network drives, DNS, hashing, and service/database access can block the Textual event loop and race confirmation. |
| Retarget all drafts when Library authority changes | It could silently move Server-bound URLs to Local or change an entire mixed draft set to recover one unsupported source. |
| Trust unscoped Server events for restart recovery | Event retention is not guaranteed and the list/detail APIs need stable identities. |
| Add a new job-history database | Existing Local rows and Server state already own submitted lifecycle data; another store adds reconciliation and migration risk. |

## Consequences

The app owns stable draft identity, authority, defaults, overrides, preflight
state, selection, and mappings to returned backend batch/job IDs. It also owns a
read-only display projection for Library and Home. That projection reconciles
from `media_reading_scope_service` and is never authoritative for submitted
status or cancellation.

New drafts bind to the active Library authority. Local and Server drafts remain
separate. Changing Library authority changes the visible draft set; it does not
retarget drafts or active jobs. Moving a draft is explicit and restarts
preflight under the new authority.

Adding a source performs string parsing and normalization only. Background
preflight owns file-system access, URL network policy, service capability,
media-type resolution, existing-Library lookup, and hashing. Sources remain
`checking` until preflight produces `ready`, `needs input`, or `needs review`.
Only `ready` sources contribute to confirmation counts.

One confirmation is partitioned by authority, effective media type, and
compatible frozen processing options. Each visible source retains its own
returned batch/job identity even when one user confirmation creates several
backend batches.

Local submission groups execute one at a time in workers through the Local
service. Existing Local service rows remain authoritative. On startup or Local
Library entry, stale Local `queued` or `running` rows left by an interrupted
process reconcile to `interrupted` with explicit Retry rather than remaining
falsely active.

Server submission releases the submission path after stable identity capture.
A deduplicated app-owned observer reconciles events or polling, with one owner
per job or batch and bounded retry/backoff. Automatic restart recovery uses a
terminal-history-bounded ledger under `library.ingest.server_receipts`.
One receipt represents a submitted backend batch and contains its job/batch
IDs, authority, a safe display label, and submission timestamp. All nonterminal
receipts are retained; terminal history is capped at the 20 most recent
receipts. Receipts contain no credentials, raw URL query values, or job
payloads. Unscoped recent events may supplement but cannot replace receipt
reconciliation through job detail/list APIs.

Home consumes the read-only job projection for Running and Needs Attention. It
cannot mutate, cancel, retry, or become another lifecycle authority.

The UI redesign must be delivered in mergeable phases: contracts and safety,
Local canvas, Server/Home integration, then administration migration and
legacy retirement. Runtime coordination must not be added directly to the
already-large `library_screen.py`.

## Links

- [Library Import Media redesign](../../Docs/superpowers/specs/2026-07-09-library-ingest-upload-redesign-design.md)
- [Superseded ADR-013](013-library-ingest-ownership-and-job-lifecycle.md)
- [ADR-011: Chatbook Workbench UI System](011-chatbook-workbench-ui-system.md)
- [Media reading scope service](../../tldw_chatbook/Media/media_reading_scope_service.py)
- [Local media reading service](../../tldw_chatbook/Media/local_media_reading_service.py)
- [Server media reading service](../../tldw_chatbook/Media/server_media_reading_service.py)
