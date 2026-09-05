# Chatbook Canvas — Design Specification

- **Date:** 2026-09-03
- **Status:** Approved
- **ADR required:** yes
- **ADR path:** `backlog/decisions/115-local-versioned-canvas-artifacts-and-browser-sandbox.md`
- **Reason:** Canvas introduces durable schema, branch-aware artifact ownership,
  browser sandbox and authentication boundaries, a parent/child process
  protocol, export/import changes, and future capability constraints.
- **Deferred synchronization:** `TASK-31003`

## 1. Purpose

Canvas gives an existing Chatbook conversation a browser-rendered companion
for substantial, self-contained interactive content. The assistant can create
and revise a Canvas through explicit tools; a user can also open an assistant
HTML code block in Canvas. The primary workflow is preview-first: interact
with the rendered page, return a bounded result to the chat when useful, and
ask the assistant to revise the complete document.

Canvas works from both execution surfaces:

- **Native terminal:** Chatbook opens a trusted Canvas shell in the user's
  system browser.
- **`--serve`:** the same Chatbook server origin presents Canvas beside the
  served terminal in a split pane. A remote browser never needs access to an
  additional localhost port.

V1 is deliberately a local artifact system, not a hosted application
platform. Each revision is one complete HTML document with inline CSS and
JavaScript plus size-capped `data:` assets. Generated JavaScript executes in a
capability-free virtual runtime rather than the browser's native JavaScript
realm. There is no external network,
filesystem, cookie/storage access, Chatbook API access, parent DOM access,
backend, CDN import, module fetch, or multi-file virtual filesystem.

## 2. Success Criteria

The design succeeds when:

1. A Canvas can be created, automatically opened, interacted with, revised,
   inspected, copied, and downloaded in native and served modes.
2. Every committed update is immutable, attributable to its originating
   assistant message/turn, recoverable, and correctly resolved against the
   active conversation branch.
3. Multiple named Canvases can belong to one conversation without leaking
   sibling-branch or other browser-session content.
4. Generated JavaScript has no ambient Chatbook, origin, filesystem, network,
   cookie, persistent-storage, popup, form, download, or parent-page authority.
5. `canvas.submit()` and `canvas.download()` cross the boundary only through
   bounded, typed, user-confirmed requests.
6. Temporary-chat Canvas history stays in memory and joins chat persistence
   atomically if the chat is saved.
7. Durable Canvas history round-trips through conversation/Chatbook export and
   import without executing its HTML.
8. Binding beyond loopback fails closed without origin-wide authentication;
   browser capabilities remain short-lived and session-scoped.
9. Canvas content is not duplicated into ordinary run logs, diagnostics, tool
   cards, or generic tool-history storage.

## 3. Product Principles

### 3.1 Use Canvas when the medium helps

The assistant should create a Canvas only when the result is substantial,
self-contained, likely to be reused or iterated, and materially easier to
understand or operate visually than as chat text. A normal response, table,
diagram, or code block remains preferred for smaller work. The user's **Open
in Canvas** action is an explicit override.

### 3.2 Preview first

Users operate the rendered result and request changes through conversation.
V1 has source inspection, copy, and download but no built-in code editor.

### 3.3 Snapshot, not live dashboard

Every preview is labeled with its revision and originating message. V1 never
implies that displayed information remains synchronized with an external
source.

### 3.4 Reversible, not approval-heavy

Canvas creation, content replacement, and rename are local, scoped, and
reversible, so assistant mutations do not require tool approval. Irreversible
or externally authoritative actions remain outside the sandbox. User
confirmation is still required when a generated page requests a chat draft or
download.

### 3.5 Trusted chrome around untrusted content

Canvas selection, revision controls, status, confirmations, source actions,
and failure recovery belong to the trusted shell. Model-authored HTML can
never cover, restyle, navigate, or impersonate those controls.

## 4. Scope by Release

### 4.1 V1

- One complete HTML document per revision.
- Inline CSS and JavaScript.
- Bounded passive `data:` assets; no external resources.
- Explicit assistant tools and **Open in Canvas** for assistant HTML blocks.
- Multiple named Canvases per conversation; one selected per live session.
- Immutable revisions, optimistic parent checks, branch-aware projection,
  exact historical transcript links, rename revisions, undo, and hot reload.
- Trusted source view/copy/download.
- Confirmed `canvas.submit()` and `canvas.download()` bridge actions.
- Local durable storage plus Chatbook archive export/import.
- Native browser view and served split-pane view.

### 4.2 V2

A small Chatbook-bundled, version-pinned catalog such as Canvas-compatible
builds/adapters for React, D3, and Mermaid. A library is admitted only if it
runs inside the virtual runtime and uses the mediated DOM contract; it never
receives the native browser global. Library versions, compatibility profiles,
and integrity become part of revision/export compatibility.

### 4.3 V3

A multi-file project (`index.html`, CSS, JavaScript, and assets) backed by an
in-memory virtual filesystem. V3 requires a separate design for file identity,
imports, bundling, quotas, revision graphs, and export compatibility.

### 4.4 Explicitly deferred

- Server synchronization (`TASK-31003`).
- Network, filesystem, cookies, page-managed persistent storage, external
  connectors, and Chatbook APIs.
- Public publishing, gallery/discovery, sharing, collaboration, and comments.
- Browser automation or assistant access to the preview DOM/console.
- Multi-route applications or an application backend.
- Built-in source editing.

These capabilities cannot be added by loosening V1 flags. Each requires an
explicit capability, ownership, and security contract; a separate renderer
origin is likely for networked versions.

## 5. Core Concepts and Identities

| Concept | Meaning |
| --- | --- |
| Canvas | Stable conversation-owned artifact identity. |
| Revision | Immutable title + complete HTML snapshot and provenance. |
| Revision parent | The exact Canvas revision from which a mutation branched. |
| Origin message | Persisted assistant message that caused a tool revision, or active-path message anchoring a direct user rename/import. |
| Selected Canvas/revision | Per-live-session browser/TUI state used as the next mutation base. |
| Resolved head | Newest eligible revision for a Canvas on the current conversation path when no historical revision is explicitly selected. |
| Following URL | Opens the resolved head and follows completed updates on the active branch. |
| Pinned URL | Opens one exact historical revision and does not silently substitute the head. |
| Browser session | Exact trusted shell connected to one Textual/Chatbook session. |
| Frame capability | Single-use renderer authorization scoped to browser session, conversation, Canvas, and revision. |

Opaque, non-sequential IDs identify conversations, messages, Canvases,
revisions, runs, browser sessions, and capabilities. Model tool arguments never
include conversation or browser-session identity.

## 6. Architecture

### 6.1 Components

1. **Canvas domain service (Chatbook app process)**
   - Owns scope, selected branch/revision, staging, validation, persistence,
     optimistic concurrency, export projections, and bridge revalidation.
2. **Canvas repository (Chatbook data layer)**
   - Reads/writes committed documents, revisions, and local reopen hints.
   - Participates in existing `CharactersRAGDB` transactions.
3. **In-memory staging store (Chatbook app process)**
   - Holds uncommitted tool revisions and all temporary-chat Canvas history.
4. **Canvas Gateway (browser delivery boundary)**
   - Serves the trusted shell and sandboxed revisions, issues/consumes browser
     capabilities, brokers live update events, and holds only bounded
     per-session delivery cache.
5. **Trusted Canvas shell (browser)**
   - Owns toolbar, split/full layout, frame lifecycle, confirmation dialogs,
     recovery, and event channel.
6. **Isolated renderer iframe (browser)**
   - Runs only Chatbook-owned bootstrap code, reconstructs a validated render
     plan, and displays one untrusted revision.
   - Hosts a bounded worker containing the virtual ECMAScript runtime; model
     JavaScript never executes in the native browser realm.
7. **Canvas control channel (`--serve`)**
   - Private bidirectional channel joining the authoritative child process to
     the parent broker that owns the public origin.

### 6.2 Native terminal flow

1. The first create/open action lazily starts the gateway on an OS-assigned
   loopback port.
2. The app creates a short-lived shell bootstrap scoped to its live session.
3. `App.open_url()` delegates browser opening through Textual's driver.
4. The shell exchanges the bootstrap for its browser session and opens the
   selected revision.
5. Subsequent creates focus/reuse the named session window when the browser
   permits; duplicate windows share selection through the shell event channel.
6. Browser-launch failure leaves the Canvas intact and shows a copyable URL,
   Retry, and source-download fallback in Chatbook.

The gateway binds loopback only. It stops with Chatbook after revoking all
capabilities and asking connected shells to enter disconnected state.

### 6.3 `--serve` parent/child flow

The current web server owns aiohttp in a parent process and launches one
Chatbook app child per browser websocket. Canvas uses that split explicitly:

1. The parent creates a per-AppService random secret and internal session ID.
2. It injects a private loopback control endpoint, secret, and public-origin
   facts into the child environment at spawn.
3. The child initiates an authenticated control connection to the parent.
4. The parent binds the existing terminal websocket/browser to that exact
   AppService and issues a separate short-lived Canvas browser session.
5. The child sends only bounded events and requested revision bytes. The
   parent cannot enumerate Chatbook's database or choose conversation scope.
6. Browser actions that return to chat travel shell → gateway → authenticated
   child channel → child domain service. The child revalidates everything and
   alone may update the composer.
7. Disconnect revokes that browser's capabilities and removes cached content.
   The Textual session continues if Canvas alone fails.

The internal control listener binds loopback on an OS-assigned port and
requires the per-AppService secret. Secrets are never placed in public URLs or
logs. Messages are versioned, length-prefixed/structured, size-capped, and
fail closed on unknown types or versions.

### 6.4 Served browser shell ownership

Chatbook owns a versioned top-level template and Canvas client module. It may
consume documented textual-serve assets/protocols, but Canvas behavior must
not depend on new string substitutions inside the minified textual-serve
bundle. Dependency upgrades receive a compatibility test for the owned shell.

The shell renders the Textual terminal and Canvas as sibling regions. On
Canvas activation it opens or expands the pane; Close collapses it without
destroying the artifact. Responsive layouts may use an overlay or full-width
Canvas rather than two unusably narrow panes.

### 6.5 Strict-zero-egress rendering pipeline

Chatbook stores the original complete document but never hands it to a browser
parser as active markup. For each revision load:

1. The host-side Canvas compiler parses HTML and CSS, extracts inline scripts
   in source order, validates assets, and produces a versioned render plan.
2. The renderer reconstructs the allowed element/attribute/style tree through
   DOM APIs; it never assigns untrusted source to `innerHTML`, `srcdoc`,
   `document.write`, or another native code/markup evaluation sink.
3. A Chatbook-owned worker starts a version-pinned ECMAScript engine compiled
   to WebAssembly. The initial safe DOM is mirrored inside that engine.
4. Generated scripts execute only in the virtual engine. A Canvas DOM facade
   converts their mutations into bounded typed patches.
5. The renderer validates every patch again before applying it to the native
   DOM. Browser events are reduced to bounded inert event records and
   dispatched back into the virtual DOM.
6. `canvas.submit()` and `canvas.download()` become explicit typed requests
   from the virtual runtime to the trusted confirmation flow. No other
   generated-runtime message can reach Chatbook or the network.

The outer shell alone owns the Chatbook event channel. The inner renderer has
no browser-session credential, conversation API, or arbitrary request path.
Its trusted runtime assets are loaded by fixed host code before untrusted
execution and are not addressable through the Canvas DOM facade.

The render plan is derived data keyed by source digest, runtime profile, and
compiler security version. It may be cached within existing quotas, but it is
never the durable source of truth or exported in place of the original HTML.
Security fixes may refuse an old document or produce a stricter plan; they do
not mutate the immutable source revision.

The reference engine family is QuickJS compiled to WebAssembly, behind a
`CanvasScriptRuntime` adapter. Implementation planning begins with a bounded
dependency spike covering license, reproducible bundling, browser support,
binary size, memory/interrupt controls, and CSP behavior. If no candidate
passes the strict-egress tests, JavaScript remains disabled rather than falling
back to native execution.

## 7. Persistence and Revision Graph

### 7.1 Durable conceptual schema

`canvas_documents`:

- `id` — stable Canvas ID.
- `conversation_id` — owning conversation, foreign key with hard-delete
  cascade.
- `created_at`, `deleted`, `deleted_at` — lifecycle metadata.

`canvas_revisions`:

- `id`, `canvas_id`, `parent_revision_id`.
- `sequence` — monotonically increasing within one Canvas; unique with
  `canvas_id`.
- `title` — revisioned title snapshot.
- `html` — complete UTF-8 document snapshot.
- `content_sha256`, `html_bytes`.
- `runtime_profile` — versioned Canvas DOM/CSS/JavaScript compatibility
  contract selected by Chatbook, not by the model.
- `actor_kind` — `assistant`, `user_rename`, or `user_import`.
- `origin_message_id` and `origin_turn_id`.
- `created_at`.

`canvas_conversation_hints`:

- `conversation_id` and last-used `canvas_id` only.
- Local reopen hint, excluded from synchronization.

Committed revisions always have a valid origin message and parent belonging to
the same Canvas. The repository enforces parent/canvas ownership and sequence
uniqueness transactionally; callers cannot supply SQL identifiers.

### 7.2 Why the title is revisioned

The Canvas title appears in branch-sensitive selection and must be reversible.
A direct rename creates a new revision with identical HTML and a changed title.
It is anchored to the active conversation message and branches from the exact
selected revision. The assistant sets the initial title at create time; V1
does not expose an assistant rename tool.

### 7.3 Branch reachability and default resolution

The child obtains the active message path from the Console store, not by
trusting the database's last-write-wins active-leaf hint. A committed revision
is eligible when its origin message lies on that path and its Canvas belongs
to the conversation.

For each Canvas, the default resolved head is the eligible revision whose
origin message is latest on the active path, then whose Canvas sequence is
latest at that anchor. An explicitly selected reachable historical revision
overrides the default for that live session. Switching conversation branches
clears an unreachable selection and chooses:

1. the resolved head of the last-used Canvas if reachable;
2. otherwise the most recently touched reachable Canvas head;
3. otherwise no active Canvas.

Sibling-branch Canvas metadata is absent from `canvas_list`, shell selectors,
and browser routes.

### 7.4 Optimistic updates

`canvas_update` requires `expected_parent_revision_id`. It must equal the
live session's selected/resolved revision at dispatch. A mismatch makes no
mutation and returns bounded current metadata so the assistant can
`canvas_read` and retry.

The run captures conversation ID, active message-leaf/path identity, Canvas
selection, and expected parent before dispatch. If the user switches chats,
branches, Canvases, or revisions while generation is in progress, completion
remains attached to the original run/branch and does not hot-reload the newly
selected view. If its parent is no longer the captured base, it fails stale.

A provider batch may mutate different Canvases, but multiple mutations for the
same Canvas in one parallel batch are refused as ambiguous. Sequential updates
within one assistant turn are allowed because each receives the preceding
revision ID.

### 7.5 Turn staging and commit

Tool success stages a revision under `(session_id, run_id, tool_call_id)` and
may preview it with **Previewing uncommitted update**. Staging is idempotent by
tool-call identity.

When the turn completes:

1. Persist or finalize the originating assistant message. A Canvas-only turn
   still produces an assistant transcript card/message anchor.
2. Bind all staged revisions to that persisted message and turn.
3. Insert documents/revisions and Canvas-card metadata.
4. Commit the message and Canvas writes in one database transaction.
5. Publish the committed event; the shell changes the badge to **Updated**.

If any write fails, the transaction rolls back, the stage remains eligible for
bounded retry during the run finalizer, and the shell continues showing the
last committed revision. Cancellation or terminal run failure discards the
stage and displays **Draft update discarded**.

### 7.6 Temporary chats

Temporary Canvas documents and revisions stay in the child staging store;
the parent broker holds only the bounded copy needed by the exact browser
session. They display a **Temporary** badge.

Saving the chat extends `ConsoleChatStore.promote_ephemeral_session`'s existing
outer database transaction with a Canvas commit participant. It writes the
conversation, complete message tree, Canvas documents, revision graph, origin
ID mappings, and reopen hint atomically. An error restores all session and
Canvas state to temporary.

Normal session end destroys the child store, revokes capabilities, removes
broker cache, and asks the shell to blank its iframe. Abnormal process loss
also drops the in-memory authority. Chatbook cannot retract source bytes a
browser has already received; host-side destruction and future access denial
are the guarantee.

### 7.7 Deletion

- Soft-deleting a conversation hides all child Canvases and prevents new
  capabilities or tool access.
- Restoring the conversation restores its Canvas visibility.
- Existing hard-purge policy cascades to Canvas documents and revisions.
- Selecting an earlier revision is not deletion.
- V1 never silently prunes committed revisions to satisfy quota.

## 8. Assistant Tool Contract

### 8.1 `canvas_list()`

No model-supplied scope arguments. Returns reachable Canvas IDs, revisioned
titles, selected/resolved revision IDs, byte counts, digests, and compact
origin metadata. It never returns HTML or sibling-branch records.

### 8.2 `canvas_read(canvas_id)`

Returns the exact live-session selected/resolved reachable revision: Canvas
ID, revision ID, parent ID, title, SHA-256 digest, byte count, and complete
HTML. It refuses foreign, deleted, or unreachable IDs. The full result reaches
the model but uses a metadata-only display/log projection.

### 8.3 `canvas_create(title, html)`

Validates title and a complete/recoverably wrappable V1 document, compiles a
strict-runtime compatibility report, creates a stable Canvas ID, stages
revision 1 under the current run, selects it, and requests automatic opening.
Conversation, branch, run, and browser scope are injected.

### 8.4 `canvas_update(canvas_id, expected_parent_revision_id, html)`

Validates scope, selected parent, complete replacement HTML, strict-runtime
compatibility, assets, and quota; then stages one immutable child revision. It
never applies a patch or modifies the parent.

### 8.5 Tool projection and retention

Canvas tool definitions declare sensitive argument/result fields. The agent
runtime provides distinct projections for:

- model history — complete content required for the tool loop;
- invocation — complete validated values;
- user display — IDs, title, bytes, digest, and status;
- run log/diagnostics — content-free metadata;
- cycle detection — stable digest rather than serialized HTML;
- continuation/resume — enough identity to reconcile committed state without
  persisting the HTML in generic agent records.

Tool results never echo HTML after create/update. They return only the created
identity, revision identity, digest, status, and bounded error/retry metadata.
This does not hide Canvas content from the selected LLM provider; it prevents
unnecessary additional local copies.

## 9. User-Initiated Creation and Selection

Assistant fenced code blocks labeled `html` expose **Open in Canvas**. The
identity `(conversation_id, message_id, code_block_index)` makes the action
idempotent: repeated activation reopens the same imported Canvas. **Open as
new** intentionally creates another identity.

- A full document is preserved after validation.
- A fragment may be wrapped deterministically in Chatbook's minimal document
  shell; the source view marks it as wrapped.
- Forbidden external dependencies or oversize/active data assets are not
  silently removed. Chatbook offers source-only inspection and an unsent
  **Ask assistant to make this self-contained** draft.
- Unsupported native-browser APIs or markup sinks are likewise not emulated
  unsafely. Chatbook reports the compatibility failure and offers an unsent
  **Ask assistant to adapt this to the Canvas runtime** draft.
- The imported revision is anchored to the source assistant message with
  actor `user_import`.

The shell's Canvas and Revision selectors operate only over the active
session's reachable set. A following view tracks the resolved head. A
transcript Canvas card opens an exact pinned revision and displays **Viewing
rN · Go to current**. Opening a historical revision makes it the selected base
for the next explicit update or rename, which creates a branch.

## 10. Browser UX

### 10.1 Trusted toolbar

Every Canvas shell consistently exposes:

- editable revisioned title;
- Canvas selector;
- revision selector with current/historical/temporary/draft state;
- Source, Copy source, Download source, Reload, and Close;
- Snapshot provenance (origin message/turn and revision);
- connection state and concise update/recovery notices.

Creation opens Canvas automatically. A completed staged update hot-reloads
only the exact matching following view. The previous committed revision stays
available. After commit, show **Updated · Undo / View previous**. Undo selects
the parent Canvas revision; it does not rewind chat history.

### 10.2 Generated-page submission

The virtual runtime exposes only a tiny bridge facade.
`canvas.submit(value)` accepts a JSON value or text. The renderer and shell
check:

- exact virtual-worker and iframe endpoints;
- per-load random nonce and render-plan identity;
- exact message schema and allowed action;
- serialized byte limit, JSON depth/shape limits, and rate limit;
- one pending confirmation maximum.

The trusted confirmation renders the payload as inert text/tree, identifies
the target conversation, and offers Confirm, Cancel, and Copy. On Confirm the
child domain service rechecks browser session, conversation, active branch,
Canvas, revision, and composer availability. It inserts an unsent draft only.
If context changed, it inserts nothing and offers Switch, Retry, or Copy.

### 10.3 Generated-page download

`canvas.download({filename, mime_type, data})` requests a browser download;
it does not access a host path. The shell validates and sanitizes the filename,
allows only passive V1 MIME types, checks encoded and decoded size, renders
filename / type / size in a confirmation, and creates a trusted one-shot
browser Blob only after confirmation. Executable/active formats—including
HTML, JavaScript, SVG, XML, browser extensions, and platform executables—are
rejected from the generated bridge.

Source download is a separate shell-owned action and does not use the
generated bridge. It defaults to an inert `.canvas.html.txt` attachment that
preserves the exact source bytes. An explicit **Download as runnable HTML**
action may use `.html` only after warning that opening it outside Chatbook
bypasses the Canvas runtime and its zero-egress guarantee. Copy source carries
the same concise warning.

### 10.4 Failure recovery

Blank output, compiler refusal, runtime exception, CSP refusal, resource-budget
exhaustion, oversize output, and connection loss receive distinct states.
The shell offers:

- Reload;
- View previous revision;
- Reopen with scripts disabled;
- Inspect source;
- Ask assistant to fix.

**Ask assistant to fix** creates an unsent draft with Canvas/revision identity,
failure class, and bounded sanitized error excerpt. Raw HTML and arbitrary
console logs are not attached automatically.

A tight loop executes inside the virtual engine's worker. The engine interrupt
budget stops ordinary runaway execution; the shell may terminate and recreate
an unresponsive worker without losing committed state. DOM patch, event,
timer, microtask, memory, and render-rate budgets prevent generated work from
becoming an unbounded native-browser queue. Browser/engine defects remain out
of scope, but generated loops do not run on the shell's UI thread.

### 10.5 Accessibility and graceful output

Generation guidance and advisory checks encourage semantic controls, keyboard
operation, visible focus, sufficient contrast, reduced motion, labeled axes
and units, non-color-only meaning, a concise text summary/data table where
appropriate, and a useful static or `<noscript>` explanation. These checks can
warn and offer an assistant repair draft; they are not represented as proof of
accessibility and are not the security boundary.

## 11. Sandbox and Content Validation

### 11.1 Runtime authority

Generated JavaScript runs in a WebAssembly-hosted ECMAScript engine with an
explicit allowlist of standard language primitives, bounded console, virtual
timers, the Canvas DOM facade, and the two confirmed bridge calls. It receives
no native `window`, `document`, `location`, `navigator`, `fetch`, XHR,
WebSocket, EventSource, sendBeacon, Worker, importScripts, dynamic module
loader, cookie/storage, clipboard, filesystem, or URL-navigation object.

The renderer iframe uses `sandbox="allow-scripts"` only because its
Chatbook-owned bootstrap must run. It omits `allow-same-origin`, forms, popups,
top navigation, downloads, modals, pointer lock, presentation, and storage.
Untrusted scripts are data consumed by the virtual engine, never native
`<script>` nodes, event-handler attributes, or JavaScript URLs. The shell and
renderer never share DOM authority.

### 11.2 Response policy

The renderer response supplies defense-in-depth headers equivalent to:

- `Content-Security-Policy`: default deny; only integrity-pinned Chatbook
  bootstrap/runtime resources; bounded presentation assets from the render
  plan; `connect-src 'none'`; no untrusted inline/native scripts, child frames,
  objects, forms, base URL, manifests, or external sources; only the one
  Chatbook-owned runtime worker; CSP sandbox with scripts only; frame ancestors
  limited to the trusted shell origin.
- `Referrer-Policy: no-referrer`.
- `X-Content-Type-Options: nosniff`.
- `Cache-Control: no-store`.

The final directive set is covered by real-browser tests because support and
inheritance behavior vary. Static HTML scanning is a usability/admission
check, never the isolation boundary.

### 11.3 Frame capabilities

The shell requests a new high-entropy capability for each load. It is scoped
to browser session, conversation, Canvas, revision, renderer purpose, and a
short expiry; it is consumed by the first valid iframe navigation. The server
checks fetch destination/site headers as defense in depth and refuses a
top-level renderer response. Reload obtains a new capability. Capability
values are never logged, persisted, exported, or passed to generated bridge
code as reusable credentials.

### 11.4 V1 document admission

Admission validates at least:

- valid UTF-8 and bounded source bytes;
- document/fragment normalization without silent semantic stripping;
- no external URL-bearing resources, module fetches, imports, refreshes, or
  disallowed schemes;
- no native-execution sinks, inline event attributes, JavaScript URLs, active
  embeds, or navigation targets in the compiled render plan;
- `data:` MIME allowlist, aggregate decoded bytes, and per-asset bytes;
- raster pixel dimensions and animated frame limits where supported;
- rejection of active embedded MIME types such as HTML/XML/SVG unless a later
  sanitizer contract explicitly admits them;
- conversation Canvas/revision/byte quota before staging.

Generated source is stored and exported inertly. Chatbook never evaluates it
during validation, persistence, search, export, import, logging, or branch
resolution.

### 11.5 DOM and CSS capability contract

The Canvas DOM is a documented compatibility subset, not the full browser
platform. V1 supports ordinary structural elements, forms as local controls,
tables, SVG, local-fragment focus/navigation, text/class/style mutation,
selectors, event listeners, and deterministic creation/removal of allowed
nodes. Form submission has no default network action.

External links, native custom elements, iframes, embeds, media capture,
autoplay, `innerHTML`, `document.write`, `insertAdjacentHTML`, native dialogs,
and arbitrary browser APIs are unsupported. String-to-markup operations may be
added only through the same compiler and patch validator; they cannot become a
native evaluation shortcut. CSS is parsed through an allowlist that rejects
imports, external or dynamically constructed URLs, active content, visited
link probing, and unsupported escape mechanisms. Data assets become opaque
render-plan handles; generated code never receives a network-capable URL.

### 11.6 Zero-egress enforcement and verification

The security claim is that generated HTML/CSS/JavaScript cannot directly or
without explicit user confirmation cause an outbound browser request,
including blind navigation or URL-based exfiltration. It is established by
construction: generated scripts have no native browser realm, generated
markup is never parsed as active markup, CSS and attributes are compiled to a
closed allowlist, DOM mutations pass the same validator, and the iframe CSP
independently denies connections and external resources.

Every native request observed from the renderer must match a fixed
Chatbook-owned runtime-resource allowlist before generated execution starts.
Real-browser tests attach request observers and fail on any later HTTP(S), WS,
DNS-triggering resource/navigation, beacon, worker, or download attempt. The
test corpus includes computed/encoded URLs, DOM mutation, CSS escapes, SVG,
error handlers, redirects, forms, base tags, and iframe self-navigation. A
runtime/compiler feature that cannot satisfy this suite is excluded from V1.

## 12. Authentication and Session Isolation

### 12.1 Bind policy

Loopback (`127.0.0.1`, `::1`, or validated localhost resolution) may run
without the configured remote access token. Any wider bind refuses startup
unless a valid Chatbook web access token resolves through environment/keyring
precedence. Provider API keys and the legacy server API token are not reused.

The auth middleware covers every authority-bearing route: terminal shell,
terminal websocket, Canvas shell, Canvas events, frame capability issuance,
renderer, bridge actions, and downloads. Static immutable assets may be
public only when they contain no session/config data.

### 12.2 Browser login

A system-browser bootstrap URL contains a one-time, rapidly expiring login
nonce—not the configured long-lived token. A manual remote login accepts the
configured token through a form. Successful authentication creates a random,
short-lived, host-only, HttpOnly, SameSite=Strict session cookie and redirects
to a clean URL. Use `Secure` under HTTPS. Comparisons are constant-time;
failures are rate-limited and content-free in logs.

Every websocket and state-changing request validates Host, Origin, browser
session, and CSRF proof. Proxy headers are trusted only from configured proxy
addresses. Session revocation closes terminal and Canvas event channels.

### 12.3 Transport confidentiality

An access token controls admission but does not encrypt terminal, Canvas, or
chat content. Non-loopback HTTP therefore fails by default unless an explicit
`allow_insecure_remote_http` override is enabled with a prominent warning.
Supported remote guidance is HTTPS termination in Chatbook or a configured
trusted reverse proxy. This design does not claim secure internet exposure
from token authentication alone.

### 12.4 Browser-session authorization

Global authentication proves access to this Chatbook host; it does not grant a
Canvas gallery. The child explicitly registers the exact conversation active
in its session. The parent broker issues capabilities only for that registered
scope and cannot query other conversations. Knowing another Canvas/revision
ID or stale URL is insufficient without the matching live browser capability.

## 13. Quotas and Resource Governance

V1 defines configurable hard ceilings with conservative documented defaults
for:

- HTML UTF-8 bytes per revision;
- compiled elements, attributes, CSS rules, script bytes, and asset count;
- decoded bytes per data asset and in aggregate;
- raster pixels and animation frames;
- Canvases per conversation;
- committed revisions per Canvas;
- total uncompressed committed Canvas bytes per conversation;
- staged bytes per run/session;
- `canvas.submit()` bytes, JSON depth, and calls per interval;
- `canvas.download()` decoded bytes and calls per interval;
- browser delivery cache and frame-capability count;
- virtual-engine memory/stack, execution steps or wall-time interrupt budget,
  timers, listeners, queued jobs, DOM patches, and mutations per interval.

Exact defaults must be fixed in the implementation plan after a provider
tool-call/output and browser-memory probe, then locked by tests and settings
copy. Quotas are computed on uncompressed/decoded data so compression or
base64 cannot bypass them. A refusal reports the exceeded category and safe
current/maximum values. Chatbook never silently deletes committed history.

Repeated identical full documents may be deduplicated later by digest without
changing revision semantics; V1 does not require content-addressed storage.

### 13.1 Measured V1 defaults

Task 7.2 froze the initial runtime defaults using
`scripts/canvas_runtime_quota_probe.py`. The fixtures are deterministic,
agent-authored synthetic pages; no live provider output, user content, source,
runtime messages, credentials, or tokens are retained. This is one-host
qualification, not a multi-platform benchmark: macOS 26.5.2 arm64, Python
3.12.11, Chromium 145.0.7632.6, Playwright 1.58.0, Textual 8.2.8,
html5lib 1.1, and tinycss2 1.5.1.

The 15-sample compiler arm produced the following content-free measurements.
Plan expansion is serialized typed-plan bytes divided by UTF-8 source bytes.

| Synthetic fixture | Source / plan | Expansion | Nodes / CSS / script | Compile median / p95 / max |
| --- | ---: | ---: | ---: | ---: |
| Representative cards (small) | 2,353 B / 11,248 B | 4.78x | 127 / 2 / 171 B | 1.503 / 2.125 / 2.125 ms |
| Representative cards (large) | 10,341 B / 52,741 B | 5.10x | 607 / 2 / 171 B | 7.222 / 7.692 / 7.692 ms |
| Combined adversarial ceiling | 328,883 B / 454,811 B | 1.38x | 1,800 / 900 / 256 KiB | 83.857 / 97.181 / 97.181 ms |
| DOM ceiling | 43,857 B / 167,115 B | 3.81x | 1,800 / 0 / 0 B | 41.197 / 53.878 / 53.878 ms |
| CSS ceiling | 22,919 B / 25,971 B | 1.13x | 3 / 900 / 0 B | 14.545 / 25.032 / 25.032 ms |
| Script ceiling | 262,215 B / 262,569 B | 1.00x | 3 / 0 / 256 KiB | 9.333 / 9.540 / 9.540 ms |

Exact one-over fixtures returned `dom-limit`, `css-rule-limit`, and
`script-limit`. Earlier candidate measurements rejected 5,000 nodes (207.840
ms median), the combined 5,000-node/2,000-rule/256-KiB ceiling (370.544 ms),
and even a 2,000-node/1,000-rule/256-KiB candidate (98.299 ms median, 101.556
ms maximum in 12 fresh interpreters). The frozen defaults are therefore 1,800
DOM nodes and 900 CSS rules, retaining the 256-KiB script ceiling. Compilation
still belongs off the UI/event loop; the measured ceiling is a conservative
budget, not permission to block an interactive caller.

Five real-Chromium samples used the production renderer and pinned QuickJS
worker. Median/p95 were 1.6/2.0 ms for representative generated startup,
251.4/251.4 ms for a 250-ms runaway-startup interrupt, 55.0/57.4 ms end to end
for a 50-ms runaway-event interrupt, and 17.2/18.8 ms to validate and commit
500 patches. A 501-patch event failed with `patch-limit`; 500 patches completed
at a measured median 29,070 patches/s. The combined near-limit plan's trusted
WASM/plan preparation was 76.3/78.2 ms and generated startup was 4.4/4.7 ms.
These clocks are separate from the 10-second trusted worker preparation and
native image-decode deadlines.

A trusted direct-engine arm, which is not exposed to generated code, measured
86,214 bytes of QuickJS memory before allocation and 16,863,782 bytes after an
accepted 16-MiB typed-array allocation under the retained 32-MiB heap ceiling;
a single 32-MiB allocation was refused. Under the retained 512-KiB stack
ceiling, recursion depth 1,819 completed and 1,820 was refused in a 10.6-ms
binary-search probe. The production-facade heap-pressure fixture failed closed
as `runtime-error` (142.4/144.8 ms median/p95); that status is termination
evidence, not a claim that the public failure code distinguishes OOM.

Comparable warmed Chromium process-tree RSS was 700.469 MiB for a blank page,
941.000 MiB with the trusted runtime, 971.859 MiB for the large representative
page, and 1,021.156 MiB for the combined near-limit page. These are sums across
owned Chromium processes on macOS, where shared pages may be counted more than
once. They are not Canvas-only resident memory and do not prove the QuickJS
heap ceiling. No browser-memory security ceiling was raised from these results.

The resulting reduced defaults are 1,800 DOM nodes, 900 CSS rules, and 500
patches per operation. All other fixed V1 runtime ceilings are retained. Python,
the worker, the private virtual facade, and the renderer carry matching values,
and runtime-asset integrity metadata is regenerated and checked after changes.

## 14. Export and Import

Canvas-bearing exports use Chatbook archive format 3.0. Archives without
Canvas data may remain 2.0 so older Chatbook releases can consume them.

Each conversation package includes inert Canvas manifests plus one `.html`
file per revision. Manifests record stable IDs, parent graph, revisioned title,
origin message/turn, actor, sequence, digest, byte size, runtime profile, and
deletion metadata. The root manifest identifies the Canvas extension/version
and aggregate uncompressed size.

Export verifies stored byte counts/digests and writes through existing atomic
archive handling. Import performs path traversal, entry count, declared size,
actual uncompressed size, duplicate identity, digest, UTF-8, graph cycle,
foreign parent, and origin-message validation before any database mutation.
It never renders or parses with a browser engine.

- **Restore same identity:** identical digest is idempotent; a conflicting
  identity/digest requires an explicit conflict outcome and cannot overwrite
  silently.
- **Import as new:** remap conversation, message, Canvas, revision, parent,
  origin, and reopen-hint IDs together, preserving both message and Canvas
  graphs.
- **Older archives:** continue through existing V1/V2 paths.
- **Newer/unknown Canvas extension:** fail clearly or preserve only under a
  separately designed opaque-forward-compatibility rule; never partially
  invent ancestry.
- **Unsupported runtime profile:** retain/import inert source only when the
  archive contract allows it, never execute under a guessed or weaker profile,
  and clearly require a user-approved assistant adaptation to a supported
  profile.

Canvas remains excluded from sync logs, outbound sync payloads, inbound sync
apply, and sync conflict resolution. Export/import is the only V1 portability
contract.

## 15. Observability and Privacy

Operational events may record event kind, opaque IDs, revision sequence,
byte/count buckets, digest prefix where useful, timing, validation/failure
class, and capability issue/consume/revoke outcome.

They do not record by default:

- HTML source or data assets;
- submit/download payload contents;
- configured access tokens, login nonces, browser sessions, frame
  capabilities, internal channel secrets, or CSRF values;
- composer drafts;
- full browser exceptions or console output containing generated content.

The virtual console is bounded and user-visible only for the active preview;
it is not copied into ordinary Chatbook logs or support exports by default.

User-visible diagnostics and support exports are sanitized metadata unless the
user explicitly chooses to attach Canvas source. Errors crossing the control
channel use stable codes and bounded safe messages.

## 16. Failure Semantics

| Failure | Required result |
| --- | --- |
| Browser cannot open | Canvas mutation remains valid; show copyable URL and Retry. |
| Gateway cannot start | Tool fails before staging with actionable local error; chat remains usable. |
| Served child control channel unavailable | Canvas disabled for that session; TUI continues. |
| Tool validation/quota failure | No stage; bounded category-specific error. |
| Stale expected parent | No mutation; return current safe metadata. |
| Turn commit fails | No message/revision partial commit; retain last committed preview. |
| Turn cancelled/failed after preview | Discard stage, revert preview, show Draft update discarded. |
| User changed chat/branch during run | Do not redirect update or hot-reload current view. |
| Renderer capability expired/reused | Deny; shell requests a fresh capability if still authorized. |
| Renderer exception/CSP violation | Preserve committed source; recovery actions and fix draft. |
| Runaway virtual script | Interrupt or terminate the worker, preserve committed state, and offer retry/scripts-disabled/fix actions. |
| Submit/download scope changed | No action; offer Switch/Retry/Copy as appropriate. |
| Import graph/digest/limit invalid | Reject before mutation; never render. |
| Soft-deleted conversation | Deny tool/browser access until restored. |

## 17. Verification Strategy

Only targeted tests are part of feature implementation unless the user later
requests a full repository sweep.

### 17.1 Pure/domain tests

- Document validation, fragment wrapping, render-plan compilation, external
  URL rejection, DOM/CSS capability enforcement, asset MIME / byte / pixel /
  frame limits, digest calculation, and quota categories.
- Virtual ECMAScript global allowlist, DOM mirror/patch semantics, event
  reduction, timers, interruption, memory limits, and scripts-disabled mode.
- Runtime-profile selection and fail-closed behavior for unknown, retired, or
  security-incompatible profiles.
- Revision graph ownership, same-Canvas parents, sequences, title revisions,
  stale-parent behavior, default head resolution, historical selection, and
  branch switching.
- Property-based message-tree × revision-tree reachability tests, including
  sibling branches and malformed cycles.
- Run-scope capture and chat/branch/selection races.
- Same-Canvas multi-call batch refusal and sequential same-turn updates.
- Tool model/display/log/cycle/continuation projections proving HTML absence
  outside the intended model/service path.

### 17.2 Persistence tests

- Schema migration from the previous head and rollback behavior.
- Assistant message + Canvas revision/card atomic commit.
- Multi-Canvas/multi-update turn commit and cancellation cleanup.
- Temporary chat promotion success and injected failure at every write phase,
  proving complete rollback to temporary state.
- Soft delete, restore, hard purge, and local reopen hint behavior.
- Quota refusal with no silent pruning.

### 17.3 Gateway and authentication tests

- Native loopback startup, random port, shutdown, and browser-open fallback.
- Non-loopback startup refusal without token.
- Plain remote HTTP refusal and explicit insecure override warning.
- Login nonce single use, cookie attributes, clean redirect, expiry,
  revocation, constant-time comparison seam, and auth rate limiting.
- Host/Origin/CSRF/websocket rejection matrices.
- Two AppService children and two browser contexts proving session,
  conversation, Canvas, revision, and capability isolation.
- Private control handshake/version/size/error cases; channel loss leaves TUI
  alive.
- Single-use frame capability expiry/replay/top-level denial.

### 17.4 Real-browser tests

- Assert iframe sandbox attributes and effective CSP behavior for network,
  storage, parent DOM, forms, popups, navigation, workers, frames, and direct
  downloads.
- Observe renderer requests and prove that computed URLs, CSS/SVG escape
  attempts, DOM patches, native-API probes, redirects, beacons, and iframe
  navigation produce zero post-startup egress.
- Prove generated scripts never execute in a native browser realm and cannot
  address the trusted runtime worker, shell event channel, or runtime assets.
- Hot reload only after a complete tool update and only in the matching view.
- Following versus pinned URLs, exact transcript-card revision, and branch
  switches.
- `postMessage` source/nonce/schema/size/depth/rate/one-pending controls.
- Submit confirmation inserts an unsent draft in the correct composer.
- Generated download confirmation and source download.
- Scripts-disabled recovery, source inspection, connection loss, responsive
  served layout, keyboard toolbar, and visible focus.
- Infinite-loop fixture documents actual reload/reconnect behavior without
  claiming process isolation.

### 17.5 Archive tests

- V2 export when no Canvas is included and V3 when Canvas is included.
- V3 exact restore, digest-idempotent retry, import-as-new whole-graph remap,
  and multiple conversation branches.
- Corrupt digest, cycles, missing parents/origins, duplicate IDs, path
  traversal, oversized entries, decompression bombs, and interrupted import.
- Proof that import/export never initializes or invokes a browser renderer.

### 17.6 Live verification

Targeted live evidence must cover both modes:

1. Native terminal → create → system browser → update → submit draft → undo.
2. Loopback `--serve` → split pane → hot reload → branch switch → historical
   transcript card.
3. Non-loopback bind refusal without auth.
4. Authenticated HTTPS/trusted-proxy served session with two isolated browser
   profiles.
5. Temporary Canvas save and unsaved-session destruction.
6. Canvas-bearing Chatbook export/import round trip.

Screenshots alone are insufficient for sandbox, persistence, or isolation
claims; capture command/test output and database assertions as appropriate.

## 18. Configuration Surface

The canonical Settings screen owns user-facing Canvas configuration. The
conceptual section is:

```toml
[canvas]
enabled = true
auto_open = true
# Hard quota defaults are fixed during implementation planning/probing.

[web_server]
# access_token is resolved from environment/keyring rather than emitted here.
allow_insecure_remote_http = false
trusted_proxy_hosts = []
```

There is one kill switch, not separate native/served implementations. When
Canvas is disabled, tools are not advertised, HTML blocks omit the action, and
Canvas routes/control connections fail closed. Non-loopback authentication
requirements cannot be disabled by `canvas.enabled` or by a project file.

## 19. Rollout

1. Land the ADR and complete the virtual-engine dependency/security spike.
2. Land schema/domain contracts, compiler/runtime boundary, projection seam,
   and zero-egress tests.
3. Add native loopback gateway and trusted shell behind `[canvas] enabled`.
4. Add assistant tools, staging/atomic commit, transcript cards, and manual
   HTML import.
5. Add served parent/child control protocol, owned split-pane shell, and
   origin-wide authentication.
6. Add archive format 3.0 export/import.
7. Complete the targeted security, browser, temporary-session, and two-browser
   isolation matrix before enabling by default.

The kill switch remains available after default enablement. No server sync is
added during rollout.

## 20. Reference Lessons Incorporated

- Claude Code Artifacts demonstrates that terminal work can open a stable
  browser page, update it in place, retain explicit versions, and remain
  constrained to a self-contained page.
- Claude/Claude Artifacts demonstrates multiple artifact selection,
  source/copy/download affordances, branch-derived artifact versions, and an
  error-to-chat repair path.
- Claude Code checkpointing demonstrates the importance of independently
  reversible artifact/work state and conversation state.
- ChatGPT Visualizations emphasizes using the smallest useful visual medium,
  preserving readable non-JavaScript content, reviewing generated revisions,
  labeling snapshots rather than live data, accessibility, and smaller
  fallback formats when generation fails.
- Claude Code browser integration reinforces that authenticated browser
  control is a different, more privileged capability. Chatbook V1 explicitly
  does not grant it.

References:

- https://code.claude.com/docs/en/artifacts
- https://code.claude.com/docs/en/checkpointing
- https://code.claude.com/docs/en/chrome
- https://support.claude.com/en/articles/9487310-what-are-artifacts-and-how-do-i-use-them
- https://learn.chatgpt.com/docs/visualizations

## 21. Final Design Decisions

- **ADR required:** yes; ADR-115 records the storage, runtime, security,
  process, authentication, and portability boundaries.
- **Chosen architecture:** authoritative Canvas domain service in the Chatbook
  app process plus a same-origin browser gateway/session broker; private
  authenticated control channel in served mode.
- **Update representation:** complete replacement HTML plus
  `expected_parent_revision_id`.
- **Durability:** immutable branch-aware revisions attached to conversations;
  temporary history remains in memory and promotes atomically.
- **Security:** generated JavaScript in a capability-free virtual engine,
  compiled/mediated DOM and CSS, sandboxed opaque-origin renderer,
  default-deny CSP, single-use capabilities, origin-wide remote auth, and a
  confirmed narrow bridge.
- **Portability:** local storage and Chatbook archive format 3.0; synchronization
  deferred to TASK-31003.
- **Evolution:** bundled libraries in V2; multi-file in-memory VFS in V3;
  privileged external capabilities require a new contract.
