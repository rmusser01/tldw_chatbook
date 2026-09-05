# ADR-115: Local versioned Canvas artifacts and browser sandbox

Status: Accepted
Date: 2026-09-03
Related Tasks: TASK-31226, TASK-31227, TASK-31228, TASK-31229, TASK-31230, TASK-31003
Related: TASK-31003, ADR-069, ADR-032

## Decision

Chatbook will add **Canvas** as a conversation-owned, local-first artifact
system. A Canvas is a stable identity whose immutable revisions contain one
complete, self-contained HTML document. Revisions form a parent-linked graph,
are anchored to the assistant message/turn that produced them, and are
resolved against the Console conversation's active message branch. Multiple
named Canvases may belong to a conversation; each live Chatbook session has
one selected Canvas/revision at a time. Each revision records a
Chatbook-selected runtime profile so unsupported content is never executed
under a guessed or weaker compatibility contract.

The assistant receives four narrow, conversation-scoped tools:

- `canvas_list()` returns bounded metadata for Canvases reachable on the
  active branch.
- `canvas_read(canvas_id)` returns the selected reachable revision's complete
  HTML and identity.
- `canvas_create(title, html)` stages a new Canvas under the current run.
- `canvas_update(canvas_id, expected_parent_revision_id, html)` stages one
  complete replacement and fails on a stale selected parent.

Conversation and browser identities are injected by Chatbook, never supplied
by the model. Canvas mutations are local and reversible and do not enter the
ordinary tool-approval UI. The runtime must project Canvas tool calls
differently for each consumer: full HTML may enter volatile model history and
the Canvas service, while run logs, diagnostics, transcript tool cards, cycle
detection, and ordinary persisted tool records receive only safe metadata
(opaque IDs, title where safe, byte count, digest, status, and redacted
errors). `canvas_read` receives the same output-projection treatment.

Tool revisions remain staged until the originating assistant turn commits.
The assistant message, transcript Canvas card, and every staged revision for
that turn commit atomically. A failed or cancelled turn discards its staged
revisions and restores the last committed preview. Temporary-chat Canvases
stay in memory, carry a **Temporary** badge, and join the existing atomic
temporary-conversation promotion transaction. Ending an unsaved session
destroys their host-side state and revokes its capabilities.

Canvas titles are revisioned state, not mutable document-global metadata. A
direct user rename creates a lightweight revision with the same HTML, a new
title, a user actor marker, and the same optimistic-parent rule. Selecting an
older revision is not itself a mutation; the next update or rename branches
from that selection.

### Browser delivery boundary

The Chatbook application process remains the authority for conversation
scope, branch selection, staging, persistence, and composer insertion. A
Canvas Gateway is a delivery/session broker only.

- In native terminal mode, Chatbook lazily starts a loopback-only gateway and
  opens the trusted shell through Textual's driver URL-opening API. Failure to
  open a browser does not roll back the Canvas; Chatbook exposes a copyable
  URL and retry action.
- In `--serve` mode, Canvas routes share the existing Chatbook server origin.
  The textual-serve parent owns HTTP/browser delivery while each browser's
  Chatbook application runs in a child process. Each child therefore connects
  to the parent through a private, authenticated, loopback control channel.
  A per-AppService secret and browser-session identity are injected at spawn.
  The channel carries bounded control events and revision bytes; it never
  grants the parent authority to enumerate the database or choose a branch.
  Loss of the channel disables Canvas for that session without terminating
  the TUI.

The served UI uses a Chatbook-owned, versioned browser shell rather than
adding further string patches to textual-serve's bundled JavaScript. That
shell owns the split pane, toolbar, confirmation surfaces, and Canvas event
channel. A live session explicitly authorizes only the conversation it has
opened; no Canvas route provides a global conversation or Canvas listing.

The model-authored document is compiled into a closed render plan; raw source
is never handed to a browser parser as active markup. Inline generated scripts
execute only in a version-pinned ECMAScript engine compiled to WebAssembly,
behind a `CanvasScriptRuntime` adapter. A mirrored DOM and capability-limited
facade turn script mutations into bounded typed patches, each revalidated by
the native renderer. Generated code receives no native browser global,
navigation, network, storage, filesystem, clipboard, worker, or Chatbook API.

The renderer iframe runs only Chatbook-owned bootstrap code with
`sandbox="allow-scripts"` and without same-origin, forms, popups, downloads,
navigation, storage, or modal privileges. A response CSP permits only fixed
integrity-pinned runtime assets and the one trusted worker, while denying
connections, untrusted/native scripts, child frames, objects, forms, and
external resources. The renderer route accepts a fresh, single-use, session +
conversation + Canvas + revision capability for iframe navigation, consumes
it on load, and refuses top-level navigation. Generated code receives no
reusable browser-session credential.

### WebAssembly engine dependency and security addendum

The reviewed V1 candidate is accepted for the worker implementation and real-
browser qualification gate. Chatbook vendors the exact runtime closure
`quickjs-emscripten-core@0.32.0`,
`@jitl/quickjs-singlefile-browser-release-sync@0.32.0`, and
`@jitl/quickjs-ffi-types@0.32.0`. The convenience `quickjs-emscripten` package
is deliberately absent because the bundle entry point does not import it. The
variant contains Bellard QuickJS revision `2025-09-13+f1139494` and its
published metadata targets browser/worker ES modules in release/synchronous
mode with `FILESYSTEM=0` and `SINGLE_FILE=1`.

All three runtime packages are MIT-licensed and identify the same upstream
`justjake/quickjs-emscripten` repository; the variant's notice also carries the
Bellard/QuickJS MIT terms. Their coordinated `0.32.0` release, explicit source
revision, documented browser/worker export, and complete generated variant are
an acceptable maintenance posture for a pinned candidate. They do not prove a
security response SLA or continuing maintenance. Every upgrade therefore
requires a new license, integrity, dependency-closure, browser, and
reproducibility review.

Registry inputs are pinned to these exact HTTPS tarballs and SHA-512 SRI values:

- `quickjs-emscripten-core-0.32.0.tgz` —
  `sha512-QFnPfjFey8EqknSrSxe1hZrf1/8z7/6s1QzGOmKo6++02r7QRRX7ZoyNaZh7JuVjWsVW87KnQrbZqnHkOAzUyg==`
- `quickjs-singlefile-browser-release-sync-0.32.0.tgz` —
  `sha512-Hfdl7rh8dzxNWFRiYAYNbhn0RMF1/tO6SMH2mUW0aTibqwaAtqPRbi4WkwaIDlhNz8Z4dksJi1Zjl1R54Jsc/Q==`
- `quickjs-ffi-types-0.32.0.tgz` —
  `sha512-v9T+GQpmk43VDJ7d72sf0Nexhk+ArvtUihW27dy7lqAl0zBObFKtSBBIm5RBjwIhE8VwsPPm9PNuvPvNqLWUEg==`

Regeneration uses only the pinned MIT-licensed
`esbuild-wasm@0.25.9` tarball with SRI
`sha512-Jpv5tCSwQg18aCqCRD3oHIX/prBhXMDapIoG//A+6+dV0e7KQMGFg85ihJ5T1EeMjbZjON3TqFy0VrGAnIHLDA==`.
The Python vendor command downloads no other URL, rejects redirects, verifies
SRI before opening archives, rejects path traversal, links, non-files,
duplicates, undeclared members, and undeclared runtime dependencies, then calls
the extracted esbuild executable directly. It never invokes a package manager.
The output manifest records every extracted-file SHA-256 and generated-output
SHA-256. Node and network are build-time inputs only; installed applications
load committed resources through Python package data.

The synchronous single-file variant is intentional: Canvas event dispatch has
a synchronous host/worker control path, while embedding the identical WASM
bytes removes a runtime `.wasm` fetch, URL resolver, and second integrity/CSP
surface. This does not authorize synchronous execution on the browser main
thread; Task 1.4 must keep the engine in a terminable worker. That task must use
`setMemoryLimit`, `setMaxStackSize`, `setInterruptHandler`, and bounded
`executePendingJobs`/`hasPendingJob`, configure no module loader, and expose no
ambient browser or host capabilities.

Neither QuickJS, quickjs-emscripten, the package's use of words such as
"safe", nor this dependency review is a security audit or a complete sandbox.
The package remains unaudited for Chatbook's threat model. Acceptance here only
permits implementation behind the disabled-by-default runtime seam. Generated
JavaScript cannot ship enabled until the Task 1.4 adversarial real-browser
suite proves zero generated-code egress, no native-realm execution, enforced
resource termination, and benign browser behavior. Failure of that gate keeps
scripts disabled; it must never fall back to native JavaScript execution.

V1's DOM/CSS API is an explicit compatibility subset. It supports structural
HTML, local form controls, tables, SVG, selectors, event listeners, and common
text/class/style/node mutations. It excludes external links, active embeds,
native custom elements, native markup sinks such as `innerHTML`, and arbitrary
browser APIs. Data assets are opaque validated handles. A QuickJS-family WASM
engine is the reference candidate; implementation begins with a license,
bundle, browser, resource-control, and security spike. Failure to satisfy the
zero-egress suite disables generated JavaScript rather than falling back to
native execution.

### Task 1.4 browser qualification addendum

The reference runtime uses one fresh QuickJS-WASM module, runtime, and context
per Canvas load in a dedicated module worker. The worker configures a 32 MiB
heap ceiling, 512 KiB stack, 250 ms startup and 50 ms event interrupts, and
bounded jobs, timers, listeners, event queues, patches, and mutation rates
before evaluating generated code. Typed bridge requests are additionally capped
at 16 per operation and 32 per second so bounded values cannot amplify into a
trusted-renderer message storm. It configures no module loader. Generated
scripts see only the documented virtual DOM/timer/console/SVG/typed-bridge
facade; native host capabilities are absent. Every failure causes the trusted
renderer to terminate and discard the worker, with scripts disabled and no
native execution fallback.

The worker retains install, operation, dispatch, drain, and timer functions only
as private QuickJS handles held by the native wrapper. Generated `globalThis`
cannot name or recover them. Transport serialization uses captured intrinsics
and null-prototype records; private timer enumeration uses captured
`Map.forEach`, not mutable iterator dispatch. The native wrapper independently
bounds timer count, uniqueness, identity, and delay. The renderer likewise uses
two-phase
validation: plans remain detached until every DOM/CSS/asset record is valid,
and transactions mutate a detached shadow tree and validate every bridge before
producing an immutable mutation journal. The validated journal is replayed
synchronously against stable live node maps, so ordinary event transactions do
not replace controls or lose their mutable native value/focus/selection state.
Nested descendants inside a CSS style rule are unsupported and rejected before
stylesheet adoption. A rejected transaction leaves the inert committed DOM,
stylesheet, and passive assets unchanged; those asset URLs remain document-
owned until renderer exit.

Passive-image handling is also fail closed at this boundary. The renderer
accepts at most 64 static PNG/JPEG/GIF/WebP assets, independently parses their
container dimensions/frame structure, caps each dimension at 4,096 and each
image at 4,194,304 pixels (16,777,216 aggregate), and requires native decode to
succeed within one second per image and three seconds total before it creates a
renderer-owned blob URL. Encoded bytes remain capped at 1 MiB each and 4 MiB
aggregate. Animated GIF/APNG/WebP and metadata/decode dimension disagreement
are rejected before generated execution.

Chromium's opaque-origin sandbox does not permit a direct HTTP module-worker
constructor even when the worker module is packaged beside the renderer. The
accepted V1 mechanism is therefore a fixed renderer-authored `data:` module
bootstrap which imports the exact packaged worker URL. No generated bytes or
URL enter this bootstrap. The renderer CSP permits `worker-src data:` for this
single trusted mechanism and `script-src 'self' 'wasm-unsafe-eval'` so the
packaged embedded QuickJS WASM can compile; it does not permit `unsafe-eval`,
inline native script, network connections, forms, frames, external images,
fonts, media, objects, or navigation privileges. Runtime requests complete
before the explicit generated-execution acknowledgement. The mandatory
Chromium harness now makes that statement executable: it hard-fails when
Playwright or Chromium is absent and withholds the acknowledgement unless the
exact successful trusted HTTP requests/responses and completed-request events,
renderer navigations, fixed bootstrap worker, and plan-derived local blob-image
loads are present with no foreign observation or independent egress receipt.

The owned numeric-loopback Playwright harness passed the mandatory Chromium
gate against computed/literal URLs, redirect and resource surfaces, beacons,
media/fonts/CSS, popups, workers, DOM clobbering, prototype pollution, active
SVG, blob/data navigation, native-download attempts, bridge spoofing, resource
exhaustion, syntax failures, and event storms. It observed no HTTP(S),
WebSocket, navigation, popup, download, or worker activity after generated
execution began, and the egress listener received no requests. Native shell,
renderer, parent, and worker sentinels remained unchanged. A separate native
CSP probe observed a blocked image request in browser instrumentation while
the target server received nothing, and confirmed opaque parent/storage access
and inline native script execution were blocked. Firefox and WebKit were not
installed and are recorded as skips. Product integration remains disabled
pending the independent security-focused review required by TASK-31226.

The following bridge behavior is the required future product-gateway contract;
Task 1.4 implements request emission and validation only, not confirmation or
host effects. The only model-page bridge in V1 is:

- `canvas.submit(value)`: bounded text/JSON is validated by the trusted shell,
  shown for user confirmation, revalidated by the Chatbook process, and
  inserted into the exact matching chat composer as an unsent draft.
- `canvas.download(request)`: bounded generated bytes/text are validated and
  shown with filename, MIME type, and size before the trusted shell initiates
  a browser download. Active/executable formats are rejected.

The shell separately offers trusted source inspection, copy, and source
download. Exact source defaults to an inert text attachment; downloading it as
runnable HTML requires an explicit warning that execution outside Chatbook
bypasses the Canvas security boundary. Every bridge message is checked against
the exact iframe window,
per-load nonce, schema, size/depth limits, rate limit, current conversation,
selected Canvas/revision, and browser session. At most one confirmation may be
pending. Nothing is submitted or downloaded automatically.

### Authentication and portability

Binding `--serve` beyond loopback requires a configured Chatbook web access
token and origin-wide authentication for the terminal shell, terminal
websocket, Canvas shell/events/renderer, and downloads. The bootstrap token is
exchanged for a short-lived, host-only, HttpOnly, SameSite browser session and
removed from the visible URL. Host, Origin, websocket, and CSRF checks apply.
Authentication is not encryption: non-loopback plain HTTP fails closed unless
the user explicitly enables an insecure-network override; TLS or a trusted
reverse proxy is the supported recommendation. Per-browser Canvas
capabilities cannot list or open another browser session's conversations.

### Served-mode implementation record

TASK-31230 implements this boundary against pinned `textual-serve` 1.1.3.
Chatbook extends the child-spawn environment only through
`AppService._build_environment(width, height)` and owns the WebSocket
`AppService` factory in its `Server.handle_websocket` override. The child
command is unchanged, and no new mutation of textual-serve's minified client
bundle is used. A Chatbook-owned responsive HTML/CSS/JavaScript shell embeds
the terminal and Canvas as sibling regions on the authenticated origin.

The private child-control protocol is version 1. Frames use a four-byte
big-endian length followed by strict JSON. The ceiling is derived from the
10 MiB generated-download limit plus bounded base64 and envelope overhead.
Only typed scope-snapshot, list/read, selection, authoritative event,
bridge-preparation/decision, health, shutdown, cancellation, authentication,
and bounded error messages are admitted. The broker allows at most 32 pending
requests and 64 queued events per child. Request IDs, deadlines, cancellation,
late-response tombstones, backpressure, and response-type ordering fail
closed. Each AppService receives a random one-use launch secret through the
supported spawn environment, connects only to the parent's numeric-loopback
listener, and loses that capability on disconnect, stop, or restart.

Browser admission uses only the dedicated Chatbook web credential, resolved
in the order `TLDW_CHATBOOK_WEB_ACCESS_TOKEN`, `[web_server].access_token`,
then OS keyring service `tldw_chatbook_web` / account `access_token`. Remote
binds fail without it. Remote plaintext fails unless the warned
`allow_insecure_remote_http` development override is explicit. Direct TLS or
an exact `public_url` behind a TLS-terminating proxy is required; forwarded
scheme, authority, and client data are honored only from literal addresses in
`trusted_proxy_addresses`. Browser login exchanges either the configured
credential or a one-time 60-second bootstrap nonce for a host-only opaque
session cookie with `HttpOnly`, `SameSite=Strict`, and `Secure` on HTTPS.
Host/Origin, CSRF, and WebSocket-subprotocol checks cover every
authority-bearing route. Sessions expire after 30 minutes idle or eight hours
absolute by default, revoke live channels, and live in bounded in-memory
stores (512 sessions, 64 bootstraps, and bounded per-subject login attempts).

The parent maps one authenticated browser session to one AppService child and
accepts only that child's authority-issued Canvas scope. Mounted shell,
event, source, renderer, submit, and download routes return the same not-found
shape for foreign, copied, or guessed capabilities. Bridge effects use an
exact-load reservation plus a per-preparation nonce and idempotent child
receipt: parent admission must still be current before the child may insert a
draft or authorize a download. A dropped control channel clears only that
browser's Canvas region and never substitutes another conversation; the
terminal WebSocket remains alive.

The release checkpoint exercised the production server behind an allowlisted
numeric-loopback TLS reverse proxy with an ephemeral dedicated credential,
two independent Chromium profiles, two real AppService child processes, and
terminal-driven `canvas_create` tool calls. It observed one-time bootstrap
replay rejection, secure cookies and WSS, distinct rendered documents,
indistinguishable denial of copied/guessed shell and event/source/action
capabilities, a confirmed canonical-JSON unsent draft, an exact passive
download, zero browser egress, and continued terminal input/output after one
Canvas child channel was revoked. All credentials, profiles, certificates,
downloads, processes, and disposable data were removed after the run.

Durable Canvas records live only on the Chatbook host and are included in
conversation/Chatbook export-import. A Canvas-bearing archive uses Chatbook
format 3.0, keeps revision files inert during validation/import, enforces
declared and actual uncompressed limits, and remaps conversation, message,
Canvas, and revision identities as one graph when importing as new. A
same-identity restore is digest-idempotent. Runtime profiles are preserved;
unknown or retired profiles never fall back to native execution. Archives
without Canvas content may remain format 2.0 for compatibility.

Canvas records are deliberately excluded from server synchronization until an
explicit contract is designed and approved under TASK-31003. Adding network,
filesystem, cookies, persistent page storage, external connectors, sharing,
or collaboration requires a later security/ownership decision and must not be
enabled by relaxing the V1 sandbox.

### Durable revision implementation record

TASK-31227 implements the durable portion of this decision in schema migration
65 to 66. The migration adds `canvas_documents`, `canvas_revisions`, and the
local-only `canvas_conversation_hints` table. Documents have immutable
conversation ownership. Revisions have immutable identity and payload, a
same-Canvas parent, a unique per-Canvas sequence, revisioned title/runtime
profile, UTF-8 source bytes with validated byte count and SHA-256 digest, and
an origin message/turn. Foreign keys, ownership triggers, payload-validation
checks, and no-update/no-delete triggers make invalid lineage or mutation fail
closed even when repository validation is bypassed. Soft deletion is recorded
on the document, while an authorized owning-conversation hard purge traverses
message and Canvas children before removing the conversation. Canvas writes do
not create sync-log records.

The initial durable limits are 10 Canvases per conversation, 100 revisions per
Canvas, 50 MiB of source per conversation, 512 KiB per revision, 4 KiB per
title, 256 bytes per origin turn, and 4,096 messages in an active path.
Temporary sessions additionally have an 8 MiB in-memory staged-source ceiling.
These are explicit failure limits; committed revisions are never silently
pruned.

Temporary histories are owned by a session-incarnation token rather than a
reusable session ID. Promotion takes an exclusive lease, resolves an exact
native-to-durable message-ID map, contributes all Canvas rows to the existing
conversation transaction, and publishes or retires only that exact staged
snapshot after commit. Rollback releases the lease without changing the staged
graph, allowing a deterministic retry. Close, restore, same-ID recreation, and
runtime teardown cannot race an active promotion; ending an unsaved session
retires its owner and destroys its staged source.

The active-path queries were reviewed with SQLite `EXPLAIN QUERY PLAN` on a
fresh schema-66 database. Conversation and message scope checks use their
primary-key indexes; Canvas listing uses
`idx_canvas_documents_conversation` and
`idx_canvas_revisions_canvas_sequence`; exact revision reads use the revision
primary key and `uq_canvas_documents_id_conversation`. There are no table
scans. SQLite uses a temporary B-tree only for the final bounded list ordering,
whose input is capped at 100 revisions per Canvas.

### Archive portability implementation record

TASK-31231 defines Chatbook format 3.0 as a conditional Canvas extension.
Exports containing no Canvas records keep the existing 2.0 format; an export
with Canvas uses 3.0 and writes each immutable revision only as the inert entry
`canvas/<canvas-id>/<revision-id>.html.txt`. The source-free manifest records
document ownership and deletion, the complete revision graph and sequence,
revisioned title/runtime profile, exact UTF-8 size and SHA-256 digest, actor,
origin message/turn, timestamps, and non-authoritative reopen hints.

Archive-model ceilings are 1,000 Canvas documents, 100 revisions per Canvas,
100,000 revisions total, 10,000 reopen hints, 512 KiB source per revision,
and 512 MiB Canvas source per archive. Existing durable per-conversation
ceilings still apply. The container boundary separately caps 10,000 members,
128 MiB per member, 512 MiB compressed and uncompressed totals, a 1,000:1
compression ratio, 1,024-byte paths, 255-byte components, and 32 path
components. Validation rejects duplicate raw or normalized paths,
file/directory prefix collisions, special entries, traversal, ambiguous
ownership, malformed UTF-8, count/size/digest mismatches, duplicate identities,
cycles, invalid parents/sequences, foreign or absent origins, invalid hints,
and malformed runtime-profile identifiers before extraction or mutation.
Validation and extraction share one opened descriptor, and physical container
size is checked on that descriptor to close replacement and metadata races.

Export streams authoritative repository source and recomputes size/digest;
the staged Canvas origins must exist in the exact conversation graph being
written before the archive is finalized. Import-as-new precomputes maps for
conversation, message, Canvas, revision, parent, origin, and hint identities,
validates the remapped graph, then writes messages and Canvas rows under one
`BEGIN IMMEDIATE` transaction. Same-identity restore revalidates the entire
existing graph inside that transaction: exact identity and metadata are
idempotent, while any content, order, ownership, or lineage conflict aborts
without overwrite. Historical origin messages may be soft-deleted but must
still exist in the owning conversation.

Schema 67 replaces the original schema-66 `runtime_profile = 'canvas-v1'`
storage check with a bounded safe-identifier check. This permits a well-formed
future profile to round-trip as inert local data, while the compiler and
renderer still execute only explicitly supported profiles and never guess or
downgrade one. The v66→v67 migration rebuilds the immutable revision table,
preserves all constraints/triggers/indexes, and has genuine-v66 rollback and
fresh-schema parity coverage. Canvas rows remain excluded from synchronization
logs and services.

The checkpoint archive contained two documents, four revisions including a
sibling branch and title change, one soft-deleted document, a deleted
historical origin, a reopen hint, and an inert unknown runtime profile. Manual
ZIP/manifest inspection confirmed deterministic 1980 timestamps and sorted
entries, exact graph relationships and digests, four `.html.txt` sources, and
no runnable `.html` entry. Whole-graph import-as-new and exact-identity restore
both completed atomically; injected validation, streaming, message, Canvas,
and commit failures left no partial graph.

### Settings and revocation implementation record

Canonical F9 Privacy & Security owns `canvas.enabled` and
`canvas.auto_open_on_create`, both true by default. Malformed execution
preferences fail closed; quotas are read-only and configuration cannot
override them. The Settings card describes configured served posture using
the existing credential resolver and validated web-auth policy, while served
startup supplies its actual effective policy. Execution-policy polling does
not resolve credentials or read keyring.

Disabling Canvas latches execution off for that process until restart. The
application runtime owns this state, including disabled startup, deferred
tool providers, cached catalog entries, HTML-block actions, and explicit
opens. An accepted Settings disable latches synchronously before asynchronous
cleanup, so a later config writer cannot cancel that process's accepted
revocation. Native and served watchers also observe external config changes.
Native watcher activation occurs in the actual Textual mount lifecycle,
independently of preview creation, and disposal cannot restart it.

Revocation closes browser/control capabilities and stops generated execution:
the native shell closes its renderer channel, navigates the preview to a
blank document, cancels bridge state, and ends polling after disconnect;
the served host closes its broker/gateway and clears sibling browser-child
bindings while the terminal remains available. Stored revisions, temporary
history ownership, and archive import/export remain with their existing
lifecycle. Re-enabling preferences requires restarting the affected process.

Task 7.1 review regressions cover active-script browser shutdown, stable
keyboard focus after disconnect, deferred provider authority, save-generation
races, real synchronous CLI construction followed by Textual startup, and
validated proxy/keyring posture. Independent review approved commits
`69405e7c8d`, `e69141538f`, and `ff8a45fb65`. Quota measurements and complete
rollout verification remain in TASK-31232's subsequent steps.

### Runtime quota qualification record

Task 7.2 lowered V1 ceilings to 1,800 DOM nodes, 900 CSS rules, and 500
patches per operation; the 256 KiB script ceiling and other limits were
retained. Python, renderer, worker, and virtual facade enforce the same values,
and packaged asset hashes were regenerated. The reproducible probe uses
explicitly labeled synthetic assistant-authored documents, not sampled live
provider responses or user content. The design's measured-results section
records the single macOS/Python 3.12/Chromium qualification environment.

Direct trusted-engine checks distinguish a successful 16 MiB allocation from
an exact guest out-of-memory refusal for a 32 MiB allocation under the
configured 32 MiB heap. Recursion has a positive accepted control and an
observed exact native engine stack trap; that trap is containment evidence
under the configured 512 KiB stack setting, not independent proof that this
setting caused refusal. A trapped probe runtime is never reused and its
owning browser context is closed. Unexpected host/API/disposal errors fail
qualification with bounded errors rather than counting as limit refusals.
No probe diagnostics are exposed to generated code.

Browser memory measurements are gross summed process-tree RSS, potentially
double-counting shared macOS pages, not Canvas-only memory or QuickJS heap
measurements. Trusted startup, generated execution, and end-to-end event
clocks are reported separately. Compiler measurements at the final combined
ceilings ranged from 83.857 ms median/97.181 ms maximum in the initial run to
107.189/124.874 ms in the review-fix run. Thus lowered quotas do not guarantee
sub-100-ms main-loop work: Task 7.2a is a required scheduling gate before
rollout, preserving existing mutation ownership and lifecycle fencing.

### Interactive compilation ownership

Pure compilation and HTML title parsing run outside the application/server
event loop and outside the controller lock. The existing controller owns one
two-slot non-queueing admission helper shared by native preview, import, and
tools; the existing served proxy owns one for browser delivery. Async work
uses the existing default executor, while tool work already on a worker uses
the same bounded synchronous admission. No extra executor, source cache,
database owner, or revision writer is introduced.

Admission is released only by the real worker's completion, not cancellation
of its waiter. Late worker failures are observed without logging arbitrary
exception representations. A completed prepared plan remains operation-local
and must match the exact source bytes/digest and supported runtime profile.
Service compatibility failures retain bounded, source-free errors.

Tools repeat owner/state/replay/expected-parent checks under the controller
lock after unlocked compilation. HTML imports additionally capture and
revalidate session/conversation/path, selection, temporary-session incarnation,
view generation, and mounted runtime ownership before applying through the
existing controller. Already-imported parsed blocks replay before compilation.
Served delivery compares source-free child scope before and after compilation;
the gateway separately checks load identity so even a same-revision reload
rejects the old plan response.

Single-host near-limit measurements used five samples per path. Maximum
observed loop gaps were 51.002 ms native, 59.746 ms served, and 52.096 ms HTML
import, with compilation wall times reaching 154.804 ms. These qualify the
scheduling boundary, not a universal latency guarantee: Python GIL contention
and existing source reads/authority checks/mutations still use their existing
owners. Task 7.2a fulfills the scheduling gate identified above; engine quotas
and the stack-trap evidence limitation are unchanged.

## Context

Chatbook's Console already persists a branching message tree, records one
local active-leaf hint, and promotes temporary conversations atomically. Its
`--serve` mode is not the same process as the Textual app: aiohttp and browser
websockets live in the textual-serve parent, which launches a Chatbook child
per browser session. Canvas must work in both modes without exposing an extra
localhost port to a remote browser or letting the parent server become a
second conversation authority.

Generic agent runtime behavior is also unsuitable for large/sensitive HTML:
tool arguments and full tool results are currently written to run logs and
step state. Canvas therefore needs a formal projection seam rather than
special-case redaction scattered across UI widgets.

The product direction follows useful lessons from Claude Code/Claude
Artifacts and ChatGPT Visualizations: use an artifact only when terminal/chat
text is the wrong medium, keep a stable browser view with explicit versions,
make errors easy to return to chat, preserve useful non-JavaScript content,
and treat generated visualizations as reviewable snapshots rather than live
authoritative dashboards.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Run a separate public Canvas server/port | Remote browsers could not safely reach a second localhost listener, configuration would diverge between native and served modes, and it would expand the attack surface. |
| Let the textual-serve parent read/write Canvas tables directly | It cannot authoritatively know the child session's live conversation branch or composer and would create two mutation owners. |
| Put all Canvas HTTP serving inside the Textual child | Works locally but is not reachable through the same served origin without a proxy/control contract, and one port per browser child is operationally fragile. |
| Reuse only textual-serve's `open_url` and always open a new tab | It gives native mode a useful fallback but cannot implement the approved served split pane or trusted return-to-composer flow. |
| Continue patching textual-serve's minified browser bundle | Brittle across dependency upgrades and provides no stable application-owned protocol for Canvas state. |
| Store one mutable HTML document per Canvas | Loses undo, branch history, origin attribution, stale-write protection, and exact transcript-card reopening. |
| Store patch/diff updates | More token-efficient but nondeterministic to validate/apply across providers and inconsistent with V1's single-file model. Complete replacement documents were chosen explicitly. |
| Keep titles mutable outside revisions | Renames would leak across sibling conversation branches and could not be undone consistently. |
| Persist temporary Canvases in staging tables | Survives crashes unexpectedly and weakens the promise that unsaved session artifacts are destroyed. In-memory authority plus bounded broker cache is sufficient. |
| Put Canvas HTML in message metadata or generic attachments | Conflates transcript and artifact lifecycles, duplicates large snapshots, and makes branch/import semantics implicit. |
| Allow external CDNs in V1 | Makes rendering depend on network state and broadens CSP, privacy, integrity, and export guarantees. A bundled-library catalog is deferred to V2. |
| Execute generated JavaScript natively in a sandboxed iframe | CSP and sandbox flags block ordinary APIs and resources but do not prove that a hostile script cannot initiate iframe self-navigation carrying data. This violates strict zero egress. |
| Render the full browser platform in a controlled remote Chromium stream | Provides request interception but adds a large process, streaming, input, accessibility, and cross-platform subsystem. A capability-limited in-browser virtual engine keeps rendering local and inspectable. |
| Ship HTML/CSS only | Meets the boundary with less machinery but fails the approved interactive JavaScript goal. It remains the fail-closed behavior if no virtual engine passes the release gates. |
| Give the iframe direct Chatbook endpoints | Any generated script would inherit excessive authority. Confirmed, typed bridge actions preserve least privilege. |
| Protect only Canvas routes on remote `--serve` | The unauthenticated terminal websocket would still grant control of Chatbook, making Canvas-only authentication meaningless. |
| Treat access tokens over HTTP as sufficient remote security | Tokens can be observed in transit. Authentication and transport confidentiality are separate requirements. |
| Depend on browser process isolation for runaway JavaScript | Browser process assignment is not a reliable application boundary. Virtual execution uses engine interrupt/memory limits and a terminable worker instead. |

## Consequences

- A new Canvas domain/repository and schema migration are required, including
  document identity, immutable revision lineage, origin bindings, revisioned
  titles, and local reopen hints.
- The Console turn-commit and temporary-promotion transactions gain a Canvas
  participant so message and artifact history cannot partially commit.
- The agent runtime gains a general, testable sensitive tool-call/result
  projection seam. Canvas HTML is still necessarily visible to the selected
  model/provider while it creates or reads the document; the guarantee is
  that Chatbook does not create unnecessary local log/display copies.
- `--serve` gains origin-wide authentication, a Chatbook-owned shell, and a
  private parent/child Canvas control channel. This is a larger boundary than
  adding a few aiohttp routes, but it is required for correct browser-session
  isolation.
- Canvas creation/update can succeed even when browser delivery fails; the
  durable artifact and its preview availability have separate statuses.
- Full snapshots consume storage and model tokens. Strict per-document,
  decoded-asset, generated-download, submit, revision-count, Canvas-count, and
  per-conversation quotas fail explicitly; committed history is never silently
  pruned.
- Soft-deleting a conversation hides its Canvases and denies new
  capabilities. Existing hard-purge lifecycle cascades through Canvas rows.
  Host-side deletion cannot retract bytes a browser has already rendered;
  capability revocation and shell blanking are best effort for delivered
  content.
- Virtual-engine interrupt/memory limits and a terminable worker contain
  ordinary runaway generated code. The shell still exposes reload, previous
  revision, scripts-disabled reopening, source inspection, and a bounded "Ask
  assistant to fix" draft for engine/compiler/browser failures.
- Strict zero egress costs browser compatibility: generated pages target a
  documented Canvas DOM/CSS subset rather than the ambient web platform.
- The WebAssembly engine and render plan are derived runtime machinery, not
  artifact truth. Immutable revisions retain original HTML and a runtime
  profile; a security update may refuse unsafe legacy content without silently
  changing its source.
- No Canvas sync, publication, sharing gallery, browser automation, backend,
  multi-route app, or multi-file virtual filesystem is included in V1.

## Links

- [Approved design spec](../../Docs/superpowers/specs/2026-09-03-chatbook-canvas-design.md)
- [Canvas user workflow and recovery](../../Docs/User_Guide/console/canvas.md)
- [Canvas V1 runtime compatibility and security boundary](../../Docs/Canvas/V1_RUNTIME_COMPATIBILITY.md)
- [Served-mode authentication and operations](../../tldw_chatbook/Web_Server/README.md)
- [TASK-31003: Define server synchronization contract for Canvas artifacts](../tasks/task-31003%20-%20Define-server-synchronization-contract-for-Canvas-artifacts.md)
- [ADR-069: Console project-instruction local state and preflight](069-console-project-instruction-local-state-and-preflight.md)
- [ADR-032: Local agent tool permission boundary](032-local-agent-tool-permission-boundary.md)
- [Claude Code artifacts](https://code.claude.com/docs/en/artifacts)
- [Claude Code checkpointing](https://code.claude.com/docs/en/checkpointing)
- [Claude Artifacts](https://support.claude.com/en/articles/9487310-what-are-artifacts-and-how-do-i-use-them)
- [ChatGPT Visualizations](https://learn.chatgpt.com/docs/visualizations)
