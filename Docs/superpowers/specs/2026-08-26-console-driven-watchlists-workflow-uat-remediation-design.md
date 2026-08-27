# Console-Driven Watchlists Workflow UAT Remediation Design

Date: 2026-08-26
Status: Approved; reconciled to latest `origin/dev` before planning
Scope: Latest `origin/dev` baseline (`c6218918d1e70c1938f7e11df592d0c70ca60383`)

Related decisions:

- [ADR-032 — Local Agent Tool Permission Boundary](../../../backlog/decisions/032-local-agent-tool-permission-boundary.md)
- [Watchlists Agent Search Tools Design](2026-08-14-watchlists-agent-search-tools-design.md)

ADR required: yes

ADR path: `backlog/decisions/032-local-agent-tool-permission-boundary.md`
(new addendum)

Reason: this design expands the private Watchlists agent boundary from bounded
reads to Console-approved domain mutations and network work. It also introduces
an explicit Console-versus-external-MCP exposure contract for local tool
descriptors. ADR-032 already owns the synthetic local principal, permission
store, approval semantics, definition-hash checks, kill switch, and external
MCP behavior, so an addendum is preferable to a parallel decision.

## Summary

Make the Console the primary orchestration surface for the complete threat-intel
Watchlists workflow, using the already-remediated First Run flow as its setup
prerequisite:

1. inspect and finish first-run provider/model setup;
2. register several news feeds in one agent-directed operation;
3. create a Watchlist collection from those sources;
4. check the sources and inspect durable run receipts;
5. generate and schedule a briefing every 24 hours;
6. let the user's agent list, retrieve, and reason over generated briefings;
7. import a real Codex skill, or receive an accurate classification and recovery
   path when a GitHub repository is a framework rather than an installable
   `SKILL.md` bundle.

The Watchlists, Library, First Run, Settings, and Scheduling surfaces remain the
human inspection and recovery interfaces. They do not become prerequisites for
normal orchestration after initial setup.

Threat-hunt creation is explicitly outside this product scope. Chatbook will
not add an ATHF-specific integration, hunt UI, hunt schema, hunt tools, or a
briefing-to-hunt handoff.

## Latest-dev reconciliation

The approved design was originally drafted against `6bed8d6f596c0e7bb1a494a0200fe5e97c720830`
and was rebased before task filing. Two UAT findings are already resolved on the
current baseline and must not be reimplemented:

- TASK-21142 through TASK-21149 plus TASK-22281 already cover the First Run
  keyboard/focus model, provider trust states, atomic provider/model commit and
  readback, blocked Console handoff, responsive layout, and copy. First Run is
  therefore a regression checkpoint in final UAT, not an implementation slice.
- Feed, URL-family, and API execution already send the product-identifying
  `tldw-chatbook/1.0 (+https://github.com/tldw/chatbook)` User-Agent. The
  remaining feed slice adds stable failure classification/recovery and retains
  that header as a no-regression contract.

## UAT problem statement

The underlying pieces exist, but the workflow feels like several unrelated
features rather than one dependable system:

- a Console agent can search Watchlists evidence but cannot create the sources,
  collection, schedule, or briefing it needs;
- the agent cannot enumerate or consume completed briefings;
- repeated manual source creation and membership assignment are slow and
  error-prone;
- cadence controls do not give a sufficiently strong operational receipt;
- source-check compatibility failures are opaque and difficult to recover from;
- a completed briefing can exist in storage while the Artifacts surface is
  blank;
- the original UAT found ambiguous First Run selection/commit state, now fixed
  on the reconciled latest-dev baseline and retained as a regression checkpoint;
- skill import treats a non-skill GitHub framework as though an expected skill
  file were merely missing, and a repeated submission can obscure an import
  that still completes in the background;
- dense controls, narrow layouts, tooltip-only meanings, and undiscoverable
  shortcuts create avoidable first-time and power-user friction.

## Product principles

### Console-first, not Console-only

The user should be able to say what outcome they want in Console and approve
the resulting tool calls. Dedicated screens remain valuable for reviewing,
editing, troubleshooting, and visual browsing.

### Domain operations, not database access

Agent tools express user intent such as “create these sources,” “set this
Watchlist to every 24 hours,” or “read this briefing.” They do not expose tables,
arbitrary field updates, SQL, or widget actions.

### Durable receipts over optimistic copy

Every network or long-running action has an inspectable persisted receipt.
Success means the receipt exists and reports the resulting state, not merely
that a click or coroutine was started.

### Progressive disclosure

First-time users see the next required action and its state. Power users keep
keyboard access, bulk operations, canonical IDs, filters, and concise status
views without being forced through a wizard.

### Honest boundaries

The application distinguishes an installable skill from a generic framework,
an inactive cadence from an active schedule, a queued operation from a
completed one, and generated prose from source evidence.

## Goals

- Complete the core workflow from Console through permissioned tool calls.
- Add bulk source registration and bulk collection membership.
- Expose bounded source, collection, operation, and briefing-receipt metadata
  to the Console agent and approved external MCP clients.
- Keep Watchlists article search/detail evidence in Console rather than
  externally publishing snippets or bodies.
- Keep complete briefing Markdown and its durable item/source provenance
  Console-only.
- Expose Watchlists mutations and network operations only to the interactive
  Console composition.
- Make recurring briefing state operationally legible: enabled, queued, next
  eligible time, last attempt, last success, and attention state.
- Let the Console agent consume a briefing with ordered, durable item/source
  provenance intact.
- Keep results parseable, bounded, redacted, and explicit about untrusted data.
- Fix the remaining Artifacts refresh, feed recovery, and skill-import issues
  observed in UAT while retaining the landed First Run behavior.
- Preserve direct UI workflows and current local/server runtime ownership.

## Non-goals

- Any threat-hunt or ATHF-specific product feature.
- Automatic conversion of a briefing into a hunt hypothesis or document.
- Installing a generic GitHub repository as though it were a Codex skill.
- Generic database CRUD or arbitrary network-fetch tools under Watchlists names.
- Server-backed Watchlists mutations until the server contract explicitly
  supports the same operations.
- Background daemon guarantees while Chatbook is closed. Recurring briefing
  copy continues to state that the in-app scheduler runs while the app is open.
- Unbounded briefing bodies, article corpora, run histories, or source lists.
- Claiming that untrusted-content labels guarantee model obedience.

## Architecture decision

### Selected approach: shared domain command/query facade

Extend the existing Watchlists agent-query seam and add a Console command seam.
Both call application-owned domain services; neither reads or manipulates
Textual widget state.

The components are:

1. `WatchlistsToolService`: bounded, synchronous, read-only queries for source,
   collection, item, operation, and briefing data.
2. `WatchlistsCommandService`: Console-only adapters for validated domain
   mutations. Short mutations complete synchronously. Long work is accepted
   only after its existing durable run/briefing receipt exists, then proceeds
   asynchronously on the application's event loop.
3. `LocalToolSpec` exposure and approval-effect metadata: one code-derived
   declaration states whether a tool is available to Console and external MCP
   or Console only, plus whether it reads private data, mutates local state,
   contacts the network, or can incur model spend. Exposure/effect copy remains
   separate from the risk tags that drive authorization.
4. Existing domain owners: `LocalWatchlistsService`,
   `WatchlistBundleService`, `SubscriptionsDB`, briefing generation, and the
   scheduler remain authoritative. Tool services orchestrate them rather than
   reproduce their SQL or business rules.

The Console injects the already-initialized database, runtime-source loader,
running event loop, scheduler reload callback, and generation/check
coordinator. External MCP retains the existing lazy read-only SQLite resolver
and receives only descriptors explicitly marked as externally exposable.

### Rejected alternatives

| Alternative | Reason rejected |
| --- | --- |
| Drive Textual controls from tools | Requires the Watchlists screen to be mounted, couples contracts to focus/widget IDs, and cannot serve headless clients reliably. |
| Expose generic Watchlists CRUD | Leaks storage structure, makes approval copy vague, and permits mutations broader than the user's stated intent. |
| Build a separate Watchlists MCP server | Duplicates the permission/audit/catalog boundary already established by ADR-032 and risks contract drift between Console and MCP. |

## Tool catalog and exposure

### Descriptor contract

`LocalToolSpec` gains a required explicit exposure value with two states:

- `console_and_external_mcp`: may be registered in both compositions;
- `console_only`: omitted from the external MCP catalog entirely.

The external composition filters by this descriptor before publishing its
catalog. It must not maintain a second hard-coded list of excluded tool names.
Tests derive expected exposure from the same descriptors, following the
repository's code-derived inventory requirement.

Exposure fails closed. There is no permissive default: every incumbent and new
descriptor is classified explicitly, and an unclassified descriptor is
rejected during provider construction rather than exposed accidentally.

Exposure is separate from authorization:

- every local Watchlists tool remains under `local:__local__`;
- fresh or missing permission state remains `ask`;
- the Console can present the existing approval card;
- external MCP cannot satisfy `ask` and still requires an operator-recorded
  tool-level Allow for exposed read tools;
- `console_only` Watchlists tools are absent externally even if a persisted
  permission record says Allow;
- mutation tools carry `("mutates",)`;
- code-derived approval-effect metadata drives visible copy for “Read private
  Watchlists data,” “Modify local Watchlists,” “Contact network,” and “May use
  an LLM”; it does not pretend to be an authorization risk tag;
- approval cards show the entity count, sanitized destination hosts/paths when
  known, once/session/persistent scope, and where a persistent permission can
  be reviewed or revoked;
- source checks state network activity in both their description and approval
  presentation, without inventing a risk tag that the resolver does not
  enforce.

### Read-tool partition

Source, collection, operation, and briefing-receipt metadata tools are
available to Console and approved external MCP clients. Item evidence and full
briefing content are Console-only.

#### `watchlists_list_sources`

Lists bounded local source metadata using stable
`casefolded_name_prefix_asc_name_prefix_asc_id_asc` ordering with cursor
pagination: the first 96 Unicode characters of the casefolded name, then the
first 96 Unicode characters of the raw name, then ID. Optional filters cover
name/type, active/paused state, and collection. Each row
uses canonical `local:subscription:<id>` identity and includes sanitized URL,
type, collection memberships, check frequency, last check, last successful
check, failure count, and attention state. Authentication, headers, extraction
secrets, and raw errors are never returned.

#### `watchlists_list_collections`

Lists bounded Watchlist collections using canonical `local:watchlist:<id>`
identity. Each row includes source count, selection mode, default preset,
stored cadence, effective scheduler state, next eligible time, last briefing
attempt/success, and attention state. It distinguishes:

- never scheduled;
- stored cadence but app-level scheduling disabled;
- scheduled and waiting;
- due/queued;
- generation in progress;
- last attempt failed;
- completed or empty.

#### `watchlists_search_items`

Retains its approved literal full-text, scoping, cursor, redaction, and bounded
untrusted-evidence contract without semantic or hunting behavior. Its exposure
is narrowed to `console_only`; external MCP does not receive result snippets.

#### `watchlists_get_item`

Retains its approved canonical-ID, bounded-detail, redaction, and untrusted
evidence contract. Its exposure is `console_only`; external MCP does not
receive article bodies.

#### `watchlists_list_briefings`

Lists bounded briefing receipts, optionally scoped to one collection and
filtered by status/date. Results are newest first with cursor continuation and
include:

- canonical `local:briefing:<id>` identity;
- collection canonical ID and name;
- `generating`, `complete`, `empty`, or `failed` status;
- created and updated timestamps;
- selection mode, preset identity/name, and model used;
- coverage window and watermark;
- item, featured, and overflow counts;
- `body_available` and body byte count without any briefing excerpt;
- retryability/attention state without raw exception details.

This is a receipt listing, not a bulk briefing-body endpoint.

External MCP stops at this receipt metadata. It does not receive briefing
Markdown, selected/cited item arrays, or the briefing's durable provenance
snapshot.

### Console-only private briefing read

#### `watchlists_get_briefing`

Retrieves one canonical `local:briefing:<id>`. Bare integers and foreign IDs
are rejected rather than guessed. This tool is `console_only`, has no mutation
tag, and is labeled as reading private generated briefing content. The response
includes:

- bounded Markdown with Unicode-safe truncation;
- all receipt metadata from the list tool;
- `content_is_generated: true` and `content_is_untrusted: true`;
- `content_truncated` and byte/count metadata;
- cited and selected Watchlists items from the migrated durable provenance
  junction, ordered by briefing and citation position;
- canonical item IDs, featured state, title, effective/published dates, and
  sanitized item URLs;
- canonical source IDs, source names/types, and sanitized source URLs;
- a distinction between items selected for the prompt and citation IDs parsed
  from the final Markdown;
- explicit legacy/missing-reference markers when an older stored citation has
  incomplete snapshot data.

The agent can use this result directly as briefing context. It does not need a
handoff object or a hunt-specific transformation.

The serialized response uses the established 30 KiB internal ceiling, leaving
headroom under `LocalToolProvider`'s 32 KiB boundary. Metadata is retained
before body text, but selected/cited provenance arrays also have hard item and
byte ceilings plus continuation. A fixed minimum body budget is reserved before
provenance rows are packed, so metadata cannot crowd all readable prose out of
the response. The body is Unicode-safely truncated to its remaining budget.
Successful responses always remain valid JSON.

“Read latest briefing” has one deterministic meaning: list the newest
`complete` briefing receipt for the selected collection and retrieve that ID.
A newer `generating`, `empty`, or `failed` receipt is shown as operational
context but does not replace the newest readable completion. The Console
renders a briefing result card with collection, completion/freshness,
generated/untrusted labels, readable body, truncation state, a collapsed
source/citation section, and “Open in Artifacts.” Assistant copy distinguishes
generated briefing prose from source evidence and discloses truncation,
legacy provenance, and missing references.

### Shared operation read tools

#### `watchlists_get_operations_status`

Returns a bounded operational snapshot, optionally scoped by source and/or
collection:

- current local/server runtime source;
- app-level watchlist-check and briefing-schedule gates;
- whether the scheduler is running;
- stored cadence and computed next eligible time;
- latest queue-reload request state when available;
- latest source-check run receipts;
- latest briefing receipts;
- current in-process generation/check state;
- one normalized `ok`, `waiting`, `running`, `needs_attention`, `disabled`, or
  `unsupported` state per operation.

It never infers that a stored cadence is active when the scheduler gate or loop
is off.

#### `watchlists_get_operation_status`

Retrieves one exact canonical operation receipt returned by an accepted source
check or briefing generation. It accepts only
`local:watchlist_run:<id>` or `local:briefing:<id>` and returns the operation's
current state, timestamps, bounded result/error category, owning source or
collection, retry/cancel capability, and owning Runs/Artifacts destination.
This is the deterministic follow path for an accepted operation; the overview
tool is not used to guess which receipt completed.

### Console-only command tools

#### `watchlists_create_sources`

Accepts 1–50 source definitions in one call. Each definition contains a URL,
optional name, `rss`/`atom`/`url` type, optional tags, active state, and check
frequency. Validation is per row and bounded before any write.

The operation uses exact configured-source identity after trimming only outer
whitespace; it does not lowercase paths, reorder/drop query parameters, or
otherwise guess URL equivalence. Duplicates within the request and exact
pre-existing identities return `existing`. A dedicated database-owner batch
seam acquires SQLite write intent before its lookup/insert sequence, so Console,
UI, and OPML creation paths cannot race the same exact identity through
separate check-before-insert calls.

The result preserves input order and reports one of `created`, `existing`, or
`invalid` per row with a canonical source ID where available. One bad row does
not discard valid independent rows, but `partial_success` sets
`follow_on_confirmation_required: true`; Console does not create a collection
from the reduced set unless the user explicitly chooses “Continue with N valid
sources” or fixes the failed rows. URL userinfo is rejected. Query-bearing
URLs may be stored because some legitimate feeds require them, but the entire
query and fragment are treated as sensitive and never echoed in receipts,
approvals, or logs.

#### `watchlists_create_collection`

Creates one collection and optionally assigns up to 100 canonical source IDs.
`if_exists` is required by the core contract and defaults to `conflict`; the
other allowed values are `return_existing` and `auto_suffix`. Returning an
existing collection never changes its membership or settings. Updating an
existing collection requires its canonical ID through a separate update tool.

All source IDs and collision behavior are validated before the transaction
begins. Creation and memberships then commit in one owner-level transaction or
all roll back. The response reports the effective collision policy,
created/existing outcome, membership counts, and canonical collection ID. It
does not change briefing settings, schedule, run a source check, or generate a
briefing implicitly; those remain separate, separately approved intents.

#### `watchlists_update_collection_sources`

Adds and/or removes bounded sets of canonical source IDs from one canonical
collection. The tool rejects the same source appearing in both sets. Missing
or ambiguous entities produce structured outcomes; the tool does not silently
choose by partial name. Every ID is validated before mutation; all membership
changes commit in one transaction or none do.

#### `watchlists_check_sources`

Accepts up to 50 canonical source IDs or one canonical collection containing no
more than 50 sources. A larger collection returns `invalid_argument` with a
bounded instruction to list and submit explicit batches; it is never silently
truncated. After validation, the tool atomically creates or resolves durable
active run receipts before returning. The coordinator executes no more than
four source checks concurrently and retains the existing per-source network
and rate-limit protections. The response contains canonical run/source IDs.

The tool does not hold the model call open for the entire fetch/parse cycle.
Repeated calls identify already queued/running work instead of starting
duplicate concurrent checks for the same source. Every accepted row includes
`poll_tool: "watchlists_get_operation_status"`, exact `poll_arguments`, a
bounded suggested poll delay/backoff, and the terminal state set.

#### `watchlists_generate_briefing`

Accepts one canonical collection ID and an optional preset override. It obtains
the existing per-collection generation claim and creates a durable `briefings`
row before acknowledging acceptance. Generation then continues asynchronously.
The response includes canonical briefing ID and initial status. A duplicate
request while generation is active returns the existing in-flight receipt
rather than creating an untracked second operation. The accepted response also
includes the exact operation-status poll contract and terminal states.

#### `watchlists_set_briefing_schedule`

Sets one collection's interval using a constrained vocabulary
(`every_12_hours`, `every_24_hours`, `every_7_days`, `off`) plus an advanced
seconds form bounded from 3,600 seconds through 2,678,400 seconds (31 days).
The UI label is “Every 24 hours,” not “Daily.” `every_24_hours` stores exactly
86,400 seconds and `every_7_days` stores exactly 604,800 seconds. The tool may
also update selection mode and preset when supplied.

This is the existing interval-based scheduling data model, not an LLM/model
selection. “Use the existing model” means each scheduled run uses the
collection's stored briefing preset provider/model; when no preset supplies
one, the existing briefing pipeline falls back to the app's configured default
provider and that provider's default model. It never silently adopts the
current Console conversation model, and the schedule stores no second model
setting. A never-attempted schedule is immediately eligible. Later runs are
anchored to the latest attempt plus the stored interval, whether that attempt
succeeded or failed. Overdue work becomes eligible when the app and scheduler
are running again. Display timestamps use the user's timezone, while
storage/comparison remains UTC. `off` clears only the stored cadence and
preserves prior briefing receipts, preset, and selection mode.

After a successful write, the command requests an immediate scheduler queue
reload through the application's existing callback seam. The response is an
operational receipt containing:

- stored cadence;
- app-level scheduling gate;
- scheduler running state;
- reload-requested state and, only when a new acknowledgement token has
  actually completed, reload-acknowledged state;
- computed next eligible time;
- last attempt and last success;
- honest recovery copy when the schedule is stored but inactive.

The existing `request_reload()` flag can truthfully report only
`reload_requested`. Reporting `reload_acknowledged` requires a new bounded
token/future completed after the queue reload. A stopped loop, timeout, or
callback failure leaves the cadence stored, returns acknowledged false with a
recovery message, and relies on normal startup/periodic reload later.

## Long-running operation coordination

The Console provider protocol remains synchronous because the agent runtime
invokes providers from worker threads. Long Watchlists commands use an
app-owned coordinator that schedules coroutines on the already-running
application event loop. The adapter must not call `asyncio.run()` against
application-owned services and must not touch Textual widgets.

The domain owners gain explicit accept/execute seams. Acceptance validates the
request, obtains the database-enforced active claim, creates or resolves the
durable receipt, and returns its canonical ID. Execution transitions that same
receipt and performs the network/model work. The public tool never reaches
into briefing generation's private helper functions.

Acceptance occurs only after a durable existing-domain receipt is committed:

- `local_watchlist_runs` for source checks;
- `briefings` for briefing generation.

The coordinator keeps strong references to every accepted task, consumes every
terminal exception, bounds source-check concurrency, and removes tasks only
after their durable receipt reaches a terminal state. Console navigation and
tool timeouts do not cancel accepted work. Explicit Cancel is offered only
where the domain can persist a truthful cancelled/failed outcome; otherwise the
card offers Stop following rather than pretending execution stopped.

On app shutdown, the coordinator stops accepting work, requests cancellation,
waits for the existing bounded shutdown grace, and persists interrupted states
for unfinished receipts. Startup reconciliation marks stale queued/running or
generating receipts honestly and releases stale claims. The in-process registry
is an execution aid, never the source of truth. Unexpected failures write a
bounded domain failure state and return scrubbed public errors.

Every accepted response names its exact status tool/arguments, suggested first
poll delay, capped exponential backoff, and terminal states. Console receipt
cards refresh from durable storage without requiring the model to spend an
unbounded sequence of tool calls. Agent-driven polling stops at the run's
normal tool-call budget and reports that the operation continues in the app.

## Shared response rules

All Watchlists tools return JSON objects with a top-level `status`,
`retryable`, and bounded human-readable `message` when relevant. Expected
outcomes cross the provider as successful structured content:

- `ok`
- `accepted`
- `partial_success`
- `needs_disambiguation`
- `invalid_argument`
- `not_found`
- `conflict`
- `unsupported`
- `feature_unavailable`

`partial_success` is a stop point, not implicit permission to continue a
dependent mutation. The response identifies the valid and failed subset and
sets `follow_on_confirmation_required`; Console asks whether to continue with
the valid subset or fix the failures. An agent must not infer consent from the
original bulk request when the resulting scope changed.

Permission denial, timeout, kill-switch, and gate-resolution failures remain
provider errors with ADR-032's pinned copy. Unexpected implementation/storage
exceptions remain scrubbed tool errors.

All schemas set `additionalProperties: false`; core validation remains
authoritative. Booleans are rejected for integer fields. Arrays, strings,
pagination, and result bytes are bounded. Canonical IDs are returned for every
entity the agent may need to reference in a later call.

Canonical identities are `local:subscription:<id>`, `local:watchlist:<id>`,
`local:watchlist_item:<id>`, `local:watchlist_run:<id>`, and
`local:briefing:<id>`. Date inputs accept `YYYY-MM-DD` or RFC 3339, use inclusive
lower bounds, and normalize comparisons to UTC; responses use RFC 3339 UTC plus
separate user-timezone display copy where the human surface needs it. Every
paged list pins a stable ordering and filter-bound cursor rather than offset
continuation.

Feed titles, names, article bodies, URLs, and generated briefing prose are
untrusted facts, never instructions. Output shaping uses explicit field
allowlists. URLs remove userinfo, full queries, and fragments and only emit
absolute HTTP(S) values with hosts. Auth configuration, custom headers, raw
payloads, database paths, and unsanitized errors are excluded.

## Human interface changes

### Console workflow guidance

When provider/model readiness is missing, Console shows one primary action:
“Configure provider and model.” It preserves the user's draft message/intent,
routes to First Run or the canonical Settings recovery surface, reads persisted
readiness after return, and resumes only after the user sends or explicitly
confirms the preserved intent. A valid “Skip for now” leaves Console visibly
blocked; it does not imply that agent tools can run without a model.

Once ready, Console guidance is progressive rather than six equal choices. It
shows the next valid workflow action first and keeps the full capability list
under examples/help. The outcome vocabulary is:

- “Add sources”
- “Create a Watchlist”
- “Check for new items”
- “Generate a briefing”
- “Schedule briefing every 24 hours”
- “Read latest briefing”

These are prompt examples or suggested actions, not bespoke workflow buttons.
Tool-call cards show the affected canonical entities, read/modify/network/model
effects, sanitized destination scope, approval duration, and durable receipt.
Accepted source-check and briefing cards transition through
Queued/Running/Complete/Empty/Failed from durable storage, survive navigation,
and offer the exact owning Runs/Artifacts destination plus Retry/Cancel only
when supported.

### Watchlists source and collection authoring

- Add a multiline/bulk source entry path with row-level validation and
  duplicate feedback.
- Allow multi-selection of sources and “Create Watchlist from selected.”
- Preserve power-user keyboard selection and expose implemented shortcuts in
  the footer/command palette.
- Keep exact canonical IDs available in detail/inspection copy without making
  them the primary visual label.
- When horizontal space cannot support persistent filter labels at the tested
  floor, move secondary filters into a labeled disclosure rather than relying
  on indistinguishable bare values.

### Artifacts and schedules

- Empty and first-run states foreground Generate and Schedule.
- Serving, exporting, keeping, audio, and other downstream controls move under
  a selected-briefing contextual action group or `More` disclosure.
- The selected collection always shows a compact automation receipt:
  interval, app-open limitation, next eligible time, last attempt, last
  success, and current attention state.
- Saving an interval requests an immediate queue reload and refreshes the
  receipt. The primary label is “Every 24 hours,” never the ambiguous “Daily.”
- Loading or generating never destroys the last good table/body. The region
  uses stale-while-refreshing behavior with an inline progress state.
- A refresh failure leaves prior content visible and adds an error with Retry.
- A completed durable briefing that is missing from the pane triggers a
  recoverable reload path and a visible diagnostic rather than a blank region.

### Feed compatibility and recovery

Source checking retains the already-shipped product-identifying, non-secret
default User-Agent for HTTP feed, URL-family, and API requests unless the source
explicitly configures a safe override. Redirect, SSRF, authentication, and
header-safety policies remain unchanged.

HTTP failures are classified into stable categories such as access denied,
authentication required, rate limited, invalid feed, connection failure, and
temporary server error. The receipt includes status code/category and a safe
next action without echoing response bodies, auth headers, or signed URLs.

The CISA-shaped UAT case becomes a local regression fixture proving that a
standards-compliant endpoint which rejects an absent/default client identity
succeeds with Chatbook's existing product User-Agent. This is not a
site-specific bypass.

### First-run provider/model setup (baseline contract)

The current latest-dev implementation already distinguishes `detected`,
selected draft, persisted/configured readback, skipped, and attention states;
atomically commits the provider/model selection; restores forward keyboard
focus; and preserves the blocked Console handoff. This programme adds no First
Run production changes. Final UAT re-runs its focused tests and one disposable
fresh-profile path so later Console integration cannot regress that prerequisite.

### Skill import

The Library import classifier distinguishes:

- installable skill bundle (`SKILL.md` found at an accepted root);
- archive/repository containing several independently installable skills;
- valid GitHub repository that is not a Codex skill;
- malformed/unsupported URL;
- fetch or authentication failure.

A framework repository without an installable `SKILL.md` reports: “This is a
repository/framework, not an installable Codex skill.” Recovery offers generic
choices that already fit Chatbook's product boundaries: select a specific
skill subdirectory, attach it as project instructions where appropriate, use
its external CLI, or create a separate wrapper skill after trust review. No
ATHF-specific special case is added.

All file, folder, zip, and URL imports share an explicit `idle`, `importing`,
`trust_review`, `complete`, or `failed` state. Import controls are disabled
while one import is in flight. A second submit cannot cancel only the UI await
while allowing the first threaded install to land silently. This closes the
existing TASK-613 behavior consistently across every import shape.

Leaving Library does not pretend to cancel an accepted threaded import. The
service finishes its existing atomic install/trust-pending operation; on return,
Library refreshes from the authoritative skill store and reports the result.
An explicit Cancel is shown only before installation begins or when the service
can prove no files will land. App shutdown follows the service's existing
atomic-write/recovery guarantees rather than abandoning a half-installed
bundle.

The Library action vocabulary distinguishes “Import skill,” “Import document,”
and other ingestion meanings instead of presenting several unrelated actions
as bare “Import.”

## Accessibility and responsive behavior

- Every editable field and Select has a persistent visible label or belongs to
  a clearly labeled disclosed filter group.
- Tooltip text supplements rather than carries the only meaning of primary
  controls.
- Focus indicators remain visible in both active and disabled/recovery states.
- Status is never conveyed by color alone; state words and concise symbols are
  paired.
- Footer hints advertise only implemented bindings and do not shadow global or
  terminal-convention keys.
- The existing supported terminal-size matrix gains first-run, Sources bulk
  entry, collection multi-select, Artifacts loading/error, and schedule receipt
  cases.
- Power-user paths retain direct focus targets and avoid modal confirmation for
  read-only operations; destructive or network/mutating tool calls retain the
  existing approval discipline.

## Surface responsibility and state matrix

| Surface | Primary responsibility | Required states/recovery |
| --- | --- | --- |
| Console | Orchestrate, approve, follow durable receipts, consume briefing | setup blocked, ready, approval pending, accepted, running, terminal, partial-success decision, unsupported server mode |
| Watchlists Sources | Inspect/edit feeds and bulk membership | empty, filtered empty, bulk draft, row validation, partial result, loading, failed/retry |
| Watchlists Runs | Inspect exact source-check receipts | queued, running, complete, skipped, failed, cancelled when supported |
| Watchlists Artifacts | Inspect/generate briefings and per-collection interval | no briefing, stale/loading, generating, complete/empty/failed, stored-but-disabled schedule, retry/open Settings |
| Settings | Own app-level scheduler and local-tool gates | enabled, disabled, persistence failed, permission review/revoke |
| First Run (regression only) | Stage then atomically commit provider/model readiness | detected, selected draft, configured readback, skipped, needs attention |
| Library | Classify/import skills and conduct trust review | idle, inspecting/importing, not-a-skill, trust review, complete, failed/retry |

Console receipts deep-link to the owning inspection surface. Settings owns only
global gates; Artifacts owns each collection's interval. Unsupported local
Watchlists suggestions are suppressed in server runtime mode. Narrow layouts
show one primary action and one state sentence before Details.

The implementation plan must pin each affected surface's existing supported
terminal floor, then add regression cases at that floor and one normal size.
At minimum, the known UAT pressure points are the landed First Run at 100x24
(regression only) and the Watchlists Sources toolbar at 160x42. Focus order,
Escape behavior, bulk-entry
syntax, range/select-all semantics, filtered-selection behavior, local hotkeys,
and draft preservation are acceptance criteria rather than implementation
choices.

## Data and migration impact

A focused Subscriptions schema migration is required and is part of this
approved design. It covers only durable briefing provenance and atomic active
operation claims.

### Ordered durable briefing provenance

The migration rebuilds/extends `briefing_items` so a completed briefing keeps
its explanatory evidence after later source/item edits or deletion. New rows
store:

- `briefing_id`;
- original numeric item identity;
- nullable live-item link when the original row still exists;
- zero-based selection position;
- featured state;
- cited state and nullable citation position;
- snapshot title;
- sanitized snapshot item URL;
- snapshot effective and published dates;
- original numeric source identity;
- snapshot source name and type;
- sanitized snapshot source URL;
- provenance format version.

Snapshot URLs are display-safe values with userinfo, complete query, and
fragment removed; no auth configuration, custom headers, raw content, or raw
feed payload is copied into provenance. Feed-supplied names/titles remain
explicitly untrusted.

New provenance is written in selection order in the same transaction that
publishes a successful briefing, after citation IDs are known and before the
row becomes `complete`. Item/source deletion no longer erases the snapshot.
The tool may left-join the current live item for supplemental state, but the
snapshot remains authoritative for what the briefing originally referenced.

Legacy junction rows migrate without invented history: their existing item ID
and featured state are retained, live metadata is snapshotted when still
available, selection/citation order remains unknown, and the tool reports
`provenance_quality: "legacy_best_effort"`. New rows report
`provenance_quality: "ordered_snapshot"`.

### Atomic active-operation claims

The migration adds database-enforced partial uniqueness:

- at most one `local_watchlist_runs` row per source whose status is `queued` or
  `running`;
- at most one `briefings` row per collection whose status is `generating`.

Before creating each index, migration reconciliation deterministically retains
the newest active receipt for a scope and moves older duplicates to the
appropriate interrupted/skipped terminal state with fixed non-sensitive
recovery copy. Owner-level accept operations insert under the constraint;
losing a concurrent race resolves and returns the winning active receipt.
Status transition to a terminal state releases the claim. This makes duplicate
suppression cross-thread and cross-process rather than relying on an
in-process set or a race-prone preflight.

The remaining storage owners are reused:

- source-check receipts remain in `local_watchlist_runs`;
- briefing receipts remain in `briefings`;
- scheduling remains `watchlists.briefing_cadence_seconds` plus the existing
  interval projection;
- membership remains `watchlist_sources`;
- first-run and skill-import state remain with their existing persistence
  owners.

Any broader schema change—calendar-time scheduling, server synchronization of
these commands, archival article bodies, or a generic operation ledger—is
outside this design and requires a separately approved ADR/task update.

## Delivery boundaries

This scope is too large for one atomic task or PR. Implementation will be
planned as independent, testable slices in dependency order:

1. TASK-22859: ADR-032 addendum plus fail-closed tool
   exposure/approval-effect metadata.
2. TASK-22860: Subscriptions migration for ordered provenance and atomic
   source-check / briefing-generation active claims.
3. TASK-22861: bounded shared receipt/source/collection/operation reads and
   Console-only full briefing consumption.
4. TASK-22862: transactional bulk source and collection commands, including
   collision, partial-success, and membership semantics.
5. TASK-22863: app-owned async operation coordinator, exact receipt status,
   source-check and briefing-generation command tools, and live Console receipt
   cards.
6. TASK-22864: every-24-hours scheduling contract, bounded advanced intervals,
   immediate reload request/acknowledgement, and automation receipts.
7. TASK-22865: feed transport compatibility and classified recovery, including
   the product User-Agent regression.
8. TASK-22866: Watchlists bulk/multi-select UI, status labels/shortcut
   discovery, and Artifacts stale/loading/error/control-density remediation.
9. TASK-613 followed by TASK-22867: single-flight import behavior, skill-package
   classification, and Library vocabulary.
10. TASK-22868: cross-surface UAT, user documentation, and targeted regression
    closeout, including the already-landed First Run contract.

Each slice receives its own Backlog task, acceptance criteria, implementation
plan, targeted automated tests, and implementation notes. Existing TASK-613 is
used rather than duplicated for the import in-flight race; its acceptance
criteria may be clarified before work begins if needed.

Detailed implementation plans:

- [Agent boundary, provenance, and query tools](../plans/2026-08-27-watchlists-agent-boundary-and-provenance.md)
- [Console commands, durable operations, and schedules](../plans/2026-08-27-console-watchlists-commands-and-operations.md)
- [Feed recovery and Watchlists interface remediation](../plans/2026-08-27-watchlists-feed-and-interface-uat-remediation.md)
- [Library skill/framework classification](../plans/2026-08-27-library-skill-import-framework-classification.md)
- [Cross-surface workflow UAT closeout](../plans/2026-08-27-console-watchlists-workflow-uat-closeout.md)

## Verification strategy

### Tool and permission contracts

- Catalog tests pin every tool name, schema, canonical ID, result ceiling,
  risk tag, and exposure value.
- Console composition includes read and command tools when local tools are
  enabled.
- External MCP includes only explicitly exposable receipt/metadata reads and
  omits item snippets/bodies, full briefing Markdown, and every Console-only
  Watchlists command regardless of stored permission state.
- Fresh/missing permissions ask; deny, timeout, kill switch, gate error,
  session allow, and definition-hash changes retain ADR-032 behavior.
- Server Watchlists mode returns structured unsupported outcomes before
  resolving the local database or scheduling work.

### Domain behavior

- Bulk source creation covers all-created, all-existing, mixed-validity, exact
  duplicate, concurrent exact duplicate, secret-bearing URL, and upper-bound
  cases.
- Collection create/update covers idempotent membership, conflicting add/remove
  sets, missing IDs, and atomic membership outcomes.
- Source checks create receipts before background execution, enforce one
  database-backed active claim per source, and return the winning receipt to a
  concurrent loser.
- Briefing generation returns a canonical durable receipt, enforces one
  database-backed active claim per collection, records
  completion/empty/failure, and survives screen navigation.
- Every-24-hours scheduling stores exactly 86,400 seconds, requests queue
  reload, and reports requested/acknowledged/disabled/stopped states honestly.
- Briefing list/detail responses remain valid bounded JSON and preserve
  selected/cited item and source provenance under truncation.
- Migration tests cover legacy best-effort provenance, ordered new snapshots,
  deletion survival, duplicate-active reconciliation, and both partial unique
  constraints.

### Interface behavior

- Textual pilot tests drive first-time and power-user keyboard paths.
- Existing First Run tests and one fresh-profile regression verify highlight
  versus persisted selection, forward focus, readback, and Console return at
  narrow and normal sizes; no duplicate First Run implementation is added.
- Watchlists tests cover multiline source errors, multi-selection membership,
  operation receipts, loading/error/retry, and stale-while-refreshing content.
- Skill-import tests cover framework classification and the superseded-submit
  race across local file, folder, zip, and URL paths.
- Feed transport tests pin the default User-Agent and safe failure categories.

### End-to-end UAT

Using a fresh profile on the latest dev baseline, a user can ask the Console
agent to add several threat-intel feeds, create a collection, check the feeds,
generate a briefing, schedule it every 24 hours, inspect operational receipts, and read
the completed briefing with provenance. The user can independently verify the
same entities in Watchlists and Settings. A non-skill framework repository is
classified accurately, with generic recovery guidance and no hunt integration.

Only targeted suites touching each slice run by default. A full repository test
sweep requires explicit user opt-in under the repository testing policy.

## Documentation impact

Update:

- `Docs/User_Guide/console/agent-runs-and-tools.md` with the code-derived
  Watchlists tool surface, approval behavior, and example Console prompts;
- `Docs/User_Guide/watchlists.md` with bulk authoring, operation receipts,
  briefing consumption, every-24-hours scheduling, and app-open semantics;
- Library skill-import help with skill-versus-framework classification;
- ADR-032 with the Console mutation and external exposure addendum.

The documentation must not imply a briefing-to-hunt product feature.
