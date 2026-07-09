# ADR-013: Library Ingest Ownership and Job Lifecycle

Status: Superseded by ADR-014
Date: 2026-07-09
Related Task: N/A (design decision recorded before implementation task selection)
Supersedes: N/A

## Decision

Library owns user-facing media import through a Library-native staged-source
canvas, while app-owned coordination and runtime adapters manage local and
server job lifecycles; Settings owns durable ingestion-source administration.

## Context

The legacy Ingest destination combines local file selection, server source
configuration, server job submission and observation, Web Clipper enrichment,
and a results log under four top-level tabs. The approved product direction
defines Library as the owner of local and server-backed source material, and
the route inventory already maps the legacy `ingest` route to Library.

The replacement needs to support mixed files and URLs without duplicating a
destination, preserve current batch capabilities, make runtime authority
honest, and keep jobs observable when the Library canvas is not mounted. A
widget-owned worker would lose ownership on route changes. A single serial
worker that waited for server completion would allow one remote transcription
to block unrelated submissions. The URL path also crosses a network security
boundary that cannot rely on syntax validation alone.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Keep Ingest as a standalone top-level destination | Import is part of the source-to-Library lifecycle and a separate owner would preserve destination sprawl and duplicate navigation. |
| Keep every legacy tab inside the Library canvas | Source configuration and enrichment administration would continue to obscure the primary add-to-Library task. |
| Use one widget-owned queue worker | Route changes and recomposition could cancel work, duplicate workers, or hide active jobs from Home. |
| Use one app-owned worker that waits for every job to finish | Long server jobs would block later local and remote submissions even though server execution is independently authoritative. |
| Persist all local job history immediately | Durable local job storage adds schema, cleanup, crash-recovery, and migration work that is not required to deliver a recoverable v1 flow. |
| Depend on the existing basic URL validator | Syntax validation does not enforce redirect, resolved-address, private-network, size, timeout, or credential policy. |

## Consequences

The canonical user route is `Library > Ingest > Import media`. Global commands
and the legacy `ingest` alias deep-link to the same canvas. The legacy route is
retired only after capability inventory, route migration, responsiveness, and
screenshot approval gates pass under ADR-011.

The import canvas uses unified single and multi-source entry, explicit staging,
batch defaults, per-source overrides, a lifecycle list, and a selected-source
inspector. Settings owns durable server-source definitions, sync policies,
archive-source configuration, and Web Clipper defaults. Library Details exposes
read-only readiness and exact Settings recovery links.

An app-owned registry provides UI snapshots. A FIFO submission coordinator
serializes claims and local execution, but releases server jobs after submission
and stable server-ID capture. A deduplicated app-owned observer reconciles
server events or polling without being tied to the canvas. Its retry frequency
and backoff are bounded while its observation duration follows the
authoritative job lifecycle. Home consumes read-only snapshots for Running and
Needs Attention and cannot mutate jobs.

Local job history is session-scoped and active local work ends with the
application. Server jobs remain server-authoritative and rehydrate from
recent/active job queries when Server Library is revisited. Server observation
must use stable identity, bounded retry/backoff, and one observer per job or
batch.

The active Library authority determines destination. Local Library supports
files and URLs through local services. Server Library supports URLs and only
supports local files when a server upload adapter exists. Unsupported
combinations remain staged with a visible recovery path.

Queued settings are immutable snapshots. Batch defaults remain live only for
staged non-overridden fields. Exact staged duplicates are rejected immediately;
existing Library duplicates are detected in background preflight and surfaced
as explicit choices.

Every fetch adapter enforces URL and path policy at its boundary. URL policy
includes scheme, embedded-credential, resolved-address, redirect, private
network, timeout, response-size, and redaction rules. Local private-network
fetches require an explicit advanced override; Server policy remains
server-authoritative and cannot be weakened by the client.

## Links

- [Superseding ADR-014: Library Ingest Service Authority and Recovery](014-library-ingest-service-authority-and-recovery.md)
- [Library Import Media redesign](../../Docs/superpowers/specs/2026-07-09-library-ingest-upload-redesign-design.md)
- [Library L2b + L3 design](../../Docs/superpowers/specs/2026-07-07-library-l2b-l3-design.md)
- [ADR-011: Chatbook Workbench UI System](011-chatbook-workbench-ui-system.md)
- [Destination route inventory](../../tldw_chatbook/UI/Workbench/route_inventory.py)
