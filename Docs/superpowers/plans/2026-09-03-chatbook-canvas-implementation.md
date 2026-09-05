# Chatbook Canvas V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a preview-first Canvas for native terminal and `--serve` sessions that stores branch-aware HTML artifacts, runs their interactive JavaScript with strict zero egress, and exposes only confirmed, bounded return actions to Chatbook.

**Architecture:** Treat Canvas as a local conversation artifact domain, not as an iframe containing arbitrary web content. Chatbook compiles one complete HTML document into a closed render plan; generated JavaScript runs only in a bounded QuickJS-family WebAssembly worker over a mediated virtual DOM, while a trusted renderer applies revalidated patches. Tool mutations are staged with an assistant run and committed atomically with its message. Native mode exposes a loopback shell; served mode mounts the same shell on the existing authenticated Chatbook origin through a private parent/child control protocol.

**Tech Stack:** Python 3.11, Textual 8.x, SQLite/FTS5, aiohttp, html5lib, tinycss2, QuickJS WebAssembly, vanilla trusted browser JavaScript, pytest, Hypothesis, and Playwright.

**Spec:** `Docs/superpowers/specs/2026-09-03-chatbook-canvas-design.md`

## Global Constraints

- ADR required: yes
- ADR path: `backlog/decisions/121-local-versioned-canvas-artifacts-and-browser-sandbox.md`
- Reason: Canvas establishes new storage, runtime, authentication, process, archive, and long-lived UX boundaries.
- Execute against a fresh branch based on current `origin/dev`. At plan-writing time, `origin/dev` is schema version 65 while this planning branch is version 42. Re-read `_CURRENT_SCHEMA_VERSION` immediately before creating the migration and allocate exactly the next version; do not assume that v66 will still be available.
- Start each Backlog task by changing it to In Progress and adding its task-specific Implementation Plan. Finish it only after its acceptance criteria, Implementation Notes, targeted checks, documentation, self-review, and ADR links satisfy `AGENTS.md`.
- Do not run the full repository suite without explicit user approval. Every delivery below specifies targeted evidence.
- Preserve Canvas source as UTF-8 bytes and identify it with SHA-256. Never log source, bridge payloads, bearer tokens, frame capabilities, or browser session secrets.
- Never pass generated HTML to `innerHTML`, `srcdoc`, `document.write`, `eval`, `Function`, a module loader, or a native browser script element. There is no compatibility fallback that weakens this rule.
- All quotas are enforced at every relevant boundary. The initial constants below are conservative hypotheses; TASK-31232 may lower them from recorded measurements, but raising them requires documented security and memory evidence.
- Existing provider approval, continuation, chat branching, deletion, restore, archive, and failure behavior must remain byte-for-byte compatible when Canvas is disabled.
- Use real in-memory SQLite for repository tests and actual browser/network observation for zero-egress claims. Mocks alone are not completion evidence.

## Proposed File Structure

```text
tldw_chatbook/Canvas/
├── __init__.py                 # public Canvas domain exports
├── capabilities.py             # browser/frame capability mint, verify, revoke
├── compiler.py                 # HTML/CSS parser and closed render-plan compiler
├── control_protocol.py         # typed served-mode parent/child wire protocol
├── gateway.py                  # native loopback routes and shared shell handlers
├── limits.py                   # hard size, CPU, memory, and mutation quotas
├── models.py                   # IDs, revisions, render plans, bridge requests
├── repository.py               # transactional durable revision graph
├── runtime_assets.py           # packaged asset manifest and digest verification
├── service.py                  # scoped Canvas create/read/update/rename/list logic
├── staging.py                  # per-session/run temporary and pending revisions
├── web_auth.py                 # remote login, cookie, CSRF, origin, proxy policy
└── static/
    ├── canvas_shell.html
    ├── canvas_shell.css
    ├── canvas_shell.js
    ├── canvas_renderer.js
    ├── canvas_runtime_worker.js
    ├── quickjs-runtime.js
    ├── quickjs-runtime.wasm
    ├── runtime-manifest.json
    └── THIRD_PARTY_LICENSES.txt
tldw_chatbook/Agents/canvas_tool_provider.py
tldw_chatbook/Chat/console_canvas_controller.py
tldw_chatbook/Widgets/Console/console_canvas_card.py
scripts/vendor_canvas_runtime.py
Tests/Canvas/
├── test_capabilities.py
├── test_compiler.py
├── test_control_protocol.py
├── test_gateway.py
├── test_limits.py
├── test_repository.py
├── test_runtime_assets.py
├── test_service.py
├── test_staging.py
├── test_web_auth.py
└── browser/
    ├── test_canvas_accessibility.py
    ├── test_canvas_native_flow.py
    ├── test_canvas_served_flow.py
    └── test_canvas_zero_egress.py
```

The implementation may place tests beside an existing narrower suite when that better matches repository conventions, but it must preserve the ownership boundaries above. Do not create a second database connection owner, chat lifecycle, tool registry, or web server.

---

## Delivery 1 — Prove and package the strict zero-egress runtime (TASK-31226)

### Task 1.1: Freeze the runtime contract and limits

**Files:**

- Create: `tldw_chatbook/Canvas/__init__.py`
- Create: `tldw_chatbook/Canvas/models.py`
- Create: `tldw_chatbook/Canvas/limits.py`
- Create: `Tests/Canvas/test_limits.py`
- Modify: `pyproject.toml`

- [x] Add failing tests for UTF-8 byte counting, nested JSON depth, decoded `data:` asset sizes, aggregate asset sizes, DOM node counts, CSS rule counts, script bytes, and exact boundary acceptance/rejection.
- [x] Define immutable, slotted dataclasses for `CanvasCompatibilityIssue`, `RenderAsset`, `RenderNode`, `CanvasRenderPlan`, `CanvasBridgeRequest`, and `CanvasRuntimeFailure`. Use opaque strings for wire IDs and reject unknown fields while decoding browser messages.
- [x] Define `RuntimeProfile = Literal["canvas-v1"]` and a frozen `CanvasLimits` with these starting ceilings: 512 KiB HTML, 1 MiB per asset, 4 MiB aggregate assets, 5,000 nodes, 2,000 CSS rules, 256 KiB scripts, 32 MiB runtime memory, 512 KiB stack, 250 ms startup, 50 ms per event, 1,000 patches per event, 16 KiB submit payload, JSON depth 16, and 10 MiB download payload.
- [x] Centralize byte/depth/count validation in pure functions so compiler, tool provider, gateway, and archive paths cannot drift.
- [x] Add `html5lib>=1.1,<2` and `tinycss2>=1.4,<2` as core dependencies; Canvas is a default terminal feature, so its parser cannot live behind an unrelated extra.
- [x] Run `pytest Tests/Canvas/test_limits.py -q` and `python -m build --wheel` followed by wheel-content inspection for the new package.
- [x] Self-review malformed Unicode, integer overflow, duplicate identifiers, and all off-by-one boundaries.
- [x] Commit: `feat(canvas): define runtime contract and hard limits`

### Task 1.2: Compile HTML/CSS into a closed render plan

**Files:**

- Create: `tldw_chatbook/Canvas/compiler.py`
- Create: `Tests/Canvas/test_compiler.py`
- Create: `Tests/Canvas/fixtures/compiler/`

- [x] Write failing unit and Hypothesis tests for well-formed and malformed complete documents, inline styles, inline classic scripts, SVG, supported form controls, entity decoding, duplicate IDs, capped `data:` images, and stable SHA-256 output.
- [x] Write rejection tests for external URLs in every parsed URL-bearing HTML/SVG attribute, `srcset`, meta refresh, forms, CSS `url()`, `@import`, font sources, namespaces, event-handler attributes, module scripts, base URLs, navigation, embedded documents, and unsupported MIME types. Include whitespace, case, entity, escape, and computed-token variants.
- [x] Parse HTML with html5lib and CSS with tinycss2; do not use regex as the security parser. Normalize into immutable `RenderNode` and `RenderAsset` records with compiler-assigned IDs.
- [x] Emit only an allowlisted element/property/attribute/event vocabulary. Convert `data:` assets to separate render-plan entries and rewrite references to opaque asset IDs.
- [x] Preserve script source only as worker input. Strip all native event-handler attributes and represent script/event bindings through the virtual runtime protocol.
- [x] Return bounded, position-aware compatibility issues and fail the whole compile when any security-relevant construct is unsupported. Do not silently weaken or partially execute a document.
- [x] Run `pytest Tests/Canvas/test_compiler.py Tests/Canvas/test_limits.py -q`.
- [x] Self-review every browser fetch/navigation surface against MDN/HTML and CSS parser semantics, then add a regression fixture for each discovered gap.
- [x] Commit: `feat(canvas): compile documents into closed render plans`

### Task 1.3: Vendor and verify the WebAssembly engine reproducibly

**Files:**

- Create: `scripts/vendor_canvas_runtime.py`
- Create: `tldw_chatbook/Canvas/runtime_assets.py`
- Create: `tldw_chatbook/Canvas/static/runtime-manifest.json`
- Create: `tldw_chatbook/Canvas/static/THIRD_PARTY_LICENSES.txt`
- Create: `Tests/Canvas/test_runtime_assets.py`
- Modify: `pyproject.toml`

- [x] Record an ADR-121 addendum choosing or rejecting the candidate only after reviewing its license, published integrity, maintenance posture, browser support, and disclosure that the package is not itself a security audit. The current candidate is `quickjs-emscripten`/`quickjs-emscripten-core`/`@jitl/quickjs-singlefile-browser-release-sync` 0.32.0; reverify those facts at execution time.
- [x] Write a failing manifest test that requires exact package names, versions, source URLs, SHA-512 tarball integrity, extracted-file SHA-256 values, licenses, build tool version, runtime profile, and reproducible command.
- [x] Implement a vendoring script that downloads only pinned HTTPS package URLs, verifies SRI before extraction, rejects traversal/symlinks, extracts an allowlist into a temporary directory, builds the trusted bundle with an exact build-tool version, and atomically replaces only the declared generated assets. The generated JS/WASM and notices are committed; application startup never invokes Node or the network.
- [x] Add `runtime_assets.py` to load assets via `importlib.resources`, compare them to the committed manifest, and disable Canvas with a bounded diagnostic on any digest mismatch.
- [x] Include `tldw_chatbook.Canvas` and all static runtime assets in sdists and wheels.
- [x] Re-run the vendoring process twice in clean temporary directories and compare SHA-256 outputs; investigate any difference rather than blessing it.
- [x] Run `pytest Tests/Canvas/test_runtime_assets.py -q`, build sdist/wheel, install the wheel into a fresh venv, disconnect networking, and verify that the runtime assets load.
- [x] Commit: `build(canvas): vendor the pinned wasm runtime`

### Task 1.4: Implement the virtual DOM worker and adversarial proof

**Files:**

- Create: `tldw_chatbook/Canvas/static/canvas_renderer.js`
- Create: `tldw_chatbook/Canvas/static/canvas_runtime_worker.js`
- Create: `Tests/Canvas/browser/test_canvas_zero_egress.py`
- Create: `Tests/Canvas/browser/fixtures/`
- Modify: `Tests/Canvas/test_runtime_assets.py`

- [x] First build a browser harness whose trusted shell can load a render plan and whose test HTTP servers record all DNS-independent requests, redirects, websocket attempts, navigation, popup, beacon, form, media, font, CSS, and worker activity.
- [x] Add failing adversarial cases for literal and computed URLs, DOM clobbering, prototype pollution, encoded CSS URLs, SVG animation/links, timers, event storms, infinite loops, promise/job loops, deep recursion, oversized patches, listener leaks, blob/data navigation, native downloads, and bridge spoofing.
- [x] Implement a worker-only QuickJS runtime exposing a documented virtual subset of `document`, nodes, events, timers, JSON, console, SVG, `canvas.submit()`, and `canvas.download()`. Do not expose browser globals, import/module hooks, native fetch primitives, SharedArrayBuffer, storage, cookies, filesystem, WebAssembly compilation, workers, or arbitrary host callbacks.
- [x] Enforce QuickJS memory, stack, interrupt/time, pending-job, timer, listener, patch-count, and mutation-rate limits. Termination must discard the worker and leave the trusted shell responsive with scripts disabled.
- [x] Implement a trusted renderer that creates nodes only with `createElement`/`createTextNode`, applies an allowlisted property/attribute/style patch vocabulary, revalidates every patch, owns all object URLs, and never interprets generated strings as markup or code.
- [x] Use a renderer iframe with an opaque origin and a CSP that denies network, navigation, forms, plugins, frames, and native scripts except the packaged trusted renderer. Generated code remains data sent to the worker.
- [x] Assert from the harness that every adversarial case produced zero egress and zero top-level navigation while a benign interactive counter/form/SVG fixture still works.
- [x] Run `pytest Tests/Canvas/test_compiler.py Tests/Canvas/test_runtime_assets.py Tests/Canvas/browser/test_canvas_zero_egress.py -q` in Chromium and, if supported by the existing CI matrix, WebKit/Firefox.
- [x] Commit: `feat(canvas): execute scripts in a bounded virtual browser`

### Delivery 1 checkpoint

- [x] Update TASK-31226 acceptance criteria and Implementation Notes with package versions, measured limits, threat cases, commands, and evidence paths.
- [x] Request a security-focused code review before any Canvas tool or UI is enabled.
- [x] If zero egress is not demonstrated through the real browser harness, stop here: keep Canvas disabled and do not continue with product integration.

---

## Delivery 2 — Add durable branch-aware Canvas revisions (TASK-31227)

### Task 2.1: Add the schema migration and immutable repository

**Files:**

- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py`
- Create: `tldw_chatbook/Canvas/repository.py`
- Create: `Tests/Canvas/test_repository.py`
- Modify: migration fixtures under the existing database test suite

- [x] Rebase on current `origin/dev`, read the actual schema head, and name the migration from that head to the next integer. Update `_CURRENT_SCHEMA_VERSION` only in the same change.
- [x] Add failing migration tests from the immediately previous schema fixture plus current-schema create/open tests. Do not claim migration coverage using an already-current database.
- [x] Add tables for Canvas identity/ownership and immutable revisions. Store revision parent, sequence, title, runtime profile, UTF-8 source, SHA-256, source byte count, origin message/turn, created time, deleted time, and local reopen hints. Add foreign keys and indexes for conversation lookup, canvas ancestry, origin message, and sequence.
- [x] Keep titles revisioned so rename history follows branches. Enforce same-conversation/same-Canvas parentage, immutable rows, unique `(canvas_id, sequence)`, digest agreement, and quotas in one immediate transaction.
- [x] Implement typed repository methods for list, read revision, append revision, soft delete/restore, purge with owning conversation, and import batches. All SQL values are parameterized.
- [x] Add concurrent-writer tests with two real SQLite connections and injected rollback failures.
- [x] Run the focused DB migration and `Tests/Canvas/test_repository.py` suites.
- [x] Commit: `feat(canvas): persist immutable revision graphs`

### Task 2.2: Resolve revisions against the active chat branch

**Files:**

- Create: `tldw_chatbook/Canvas/service.py`
- Modify: `tldw_chatbook/Canvas/models.py`
- Create: `Tests/Canvas/test_service.py`
- Modify: `Tests/Chat/test_console_chat_store.py`

- [x] Add failing graph tests covering two message branches, two Canvas branches, historical selection, title-only revisions, sibling exclusion, exact revision reopen, and deterministic ties.
- [x] Define `CanvasScope(session_id, conversation_id, active_message_ids, selected_canvas_id, selected_revision_id, run_id)` and require it for every service operation. The service must never derive authority from a client-supplied conversation ID alone.
- [x] Implement `list_canvases`, `read_canvas`, `create_canvas`, `update_canvas`, and `rename_canvas`. `update_canvas` accepts the complete replacement document and required `expected_parent_revision_id`.
- [x] Select the newest revision whose origin message is reachable on the active path. If a user explicitly selected a historical revision, the next mutation branches from exactly that revision.
- [x] Return a structured optimistic-conflict result containing only current revision ID, digest, title, sequence, and origin—not source—and make no write.
- [x] Enforce per-conversation limits initially at 10 Canvases, 100 revisions per Canvas, and 50 MiB of durable Canvas source; keep constants centralized for later measured tuning.
- [x] Run `pytest Tests/Canvas/test_service.py Tests/Canvas/test_repository.py Tests/Chat/test_console_chat_store.py -q`.
- [x] Commit: `feat(canvas): resolve revisions on conversation branches`

### Task 2.3: Stage temporary history and join existing promotion

**Files:**

- Create: `tldw_chatbook/Canvas/staging.py`
- Modify: `tldw_chatbook/Chat/console_chat_store.py`
- Modify: `tldw_chatbook/Chat/console_chat_store.py` protocol declarations
- Create: `Tests/Canvas/test_staging.py`
- Modify: `Tests/Chat/test_console_chat_store.py`
- Modify: `Tests/Chat/test_chat_persistence_service.py`

- [x] Add failing tests for temporary create/update/rename chains, 8 MiB staged cap, idempotent `(session_id, run_id, tool_call_id)`, shutdown destruction, successful promotion, failure after each database write, and retry after rollback.
- [x] Implement an in-memory `CanvasStagingStore` keyed by session/run/tool call. It owns staged source and compile plans; transcript/tool records retain metadata only.
- [x] Add an explicit optional Canvas promotion participant to `ConsoleChatStore` rather than reaching through private attributes. The participant receives the existing transaction connection, new conversation ID, and message-ID mapping.
- [x] Extend `promote_ephemeral_session` so conversation, message tree, Canvas identities, revisions, and origin links commit in its existing transaction. Restore both chat and Canvas state completely on failure.
- [x] Destroy temporary Canvas state when its ephemeral session is discarded or the process exits. Never create orphan disk files for temporary history.
- [x] Assert that conversation soft delete/restore follows existing ownership and hard purge removes Canvas rows; no Canvas mutation enters current sync queues.
- [x] Run the focused Canvas staging, Console store, and chat persistence suites.
- [x] Commit: `feat(canvas): promote temporary histories atomically`

### Delivery 2 checkpoint

- [x] Update TASK-31227 and ADR-121 with the actual migration number, tables, constraints, quotas, and rollback evidence.
- [x] Review repository queries with `EXPLAIN QUERY PLAN` for active-path list/read and record the result in task notes.

---

## Delivery 3 — Integrate Canvas tools with atomic Console turns (TASK-31228)

### Task 3.1: Add safe generic tool-record projection

**Files:**

- Modify: `tldw_chatbook/Agents/agent_models.py`
- Modify: `tldw_chatbook/Agents/tool_catalog.py`
- Modify: the Agent runtime file that defines `LoopDeps` and persists `AgentStep`
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py`
- Create/modify: focused tests under `Tests/Agents/`

- [x] Locate every use of raw tool arguments/results in display, logging, cycle fingerprints, run records, continuations, diagnostics, and model history; encode this call-site inventory in parametrized failing tests.
- [x] Add `ToolProjectionAudience = Literal["display", "log", "cycle", "continuation"]` and an immutable `ToolRecordProjection`. Define an optional provider projection protocol and a registry dispatch method.
- [x] Keep raw values only for the immediate provider invocation and model tool-result history. Route every durable/non-model call site through its audience projection.
- [x] Default projection must preserve existing providers exactly. Add regression assertions across builtin, local, skill, and MCP providers.
- [x] Ensure a projection failure fails closed to tool name, call ID, success state, and bounded error category—never raw arguments/results.
- [x] Run the focused Agent runtime, tool catalog, run-store, continuation, and cycle-detection tests.
- [x] Commit: `refactor(agents): project sensitive tool records by audience`

### Task 3.2: Register scoped Canvas tools

**Files:**

- Create: `tldw_chatbook/Agents/canvas_tool_provider.py`
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py`
- Modify: the Console tool-registration seam
- Create: `Tests/Agents/test_canvas_tool_provider.py`

- [x] Add schema tests for `canvas_list`, `canvas_read`, `canvas_create`, and `canvas_update`; validate additional properties are rejected and all strings/counts use shared limits.
- [x] Inject a server-owned `CanvasScope` when resolving the provider. Do not accept session, conversation, branch, or run authority fields from model arguments.
- [x] Implement full-document `canvas_create(title, html)` and `canvas_update(canvas_id, expected_parent_revision_id, html)` against staging/service APIs. Return revision IDs, digests, titles, compatibility diagnostics, and conflict metadata.
- [x] Make Canvas mutations pre-authorized as reversible conversation-local operations through a narrowly named policy classification. Prove with tests that no other tool bypasses normal approval.
- [x] Return source only from explicit `canvas_read` to model history. For display/log/cycle/continuation projections, retain metadata/digests and omit all HTML.
- [x] Advertise tools only when Canvas is enabled and the session has a valid Canvas coordinator.
- [x] Run `pytest Tests/Agents/test_canvas_tool_provider.py` plus focused tool approval/catalog tests.
- [x] Commit: `feat(canvas): expose scoped assistant tools`

### Task 3.3: Commit staged mutations with the originating assistant turn

**Files:**

- Create: `tldw_chatbook/Chat/console_canvas_controller.py`
- Modify: `tldw_chatbook/Chat/console_chat_store.py`
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py`
- Modify: `tldw_chatbook/Widgets/Console/console_transcript.py`
- Create: `Tests/Chat/test_console_canvas_controller.py`
- Modify: focused Agent runtime and Console tests

- [x] Add failing tests for successful finalization, Canvas-only turns, cancellation, provider failure, app shutdown, duplicate callbacks, sequential same-turn updates, parallel ambiguous updates, message-write failure, revision-write failure, and continuation resume.
- [x] Coordinate one stage per assistant run. Serialize same-Canvas calls in invocation order; reject parallel calls whose ancestry cannot be proven.
- [x] On successful turn completion, ensure an assistant message/turn anchor exists, then commit its message, Canvas card metadata, and staged revisions within the existing persistence transaction. Mark the stage committed only after transaction success.
- [x] On cancellation or terminal failure, discard the run stage and render a non-reopenable bounded failure/status card. Retrying a tool call with the same identity must not duplicate a revision.
- [x] Persist transcript cards with Canvas/revision IDs, title, sequence, digest, status, and origin only. Reopen source through the Canvas service.
- [x] Prove serialized AgentStep records, logs, transcript widgets, cycle keys, and continuation payloads contain no unique sentinel from source HTML.
- [x] Run the focused controller, Console persistence, Agent runtime, continuation, cancellation, and transcript tests.
- [x] Commit: `feat(canvas): commit revisions with assistant turns`

### Delivery 3 checkpoint

- [x] Update TASK-31228 with the projection inventory, transaction boundary, cancellation semantics, and sentinel-leak evidence.
- [x] Request a review specifically for approval bypass scope and source leakage.

---

## Delivery 4 — Deliver native browser UX and confirmed bridge (TASK-31229)

### Task 4.1: Build the trusted native Canvas gateway

**Files:**

- Create: `tldw_chatbook/Canvas/gateway.py`
- Create: `tldw_chatbook/Canvas/capabilities.py`
- Create: `Tests/Canvas/test_gateway.py`
- Create: `Tests/Canvas/test_capabilities.py`
- Modify: `pyproject.toml`

- [x] Add `aiohttp>=3.9,<4` to core dependencies after checking existing constraints. A default terminal Canvas cannot depend on an unrelated optional extra.
- [x] Add tests proving lazy startup, OS-assigned port, loopback-only bind, one gateway per app, clean shutdown, unavailable-browser recovery, and no second conversation authority.
- [x] Implement typed routes for the trusted shell, packaged static assets, render plans, event stream, source actions, and bridge confirmation. Every route resolves server-owned session scope.
- [x] Mint cryptographically random, short-lived, single-use capabilities scoped to browser session, frame, conversation session, Canvas, revision, action, and expiry. Store only hashes, rotate on reload, and revoke on session/branch/change/close.
- [x] Reject capabilities in query parameters for top-level pages. Deliver frame capabilities through a trusted boot exchange so they do not enter history, referrers, logs, or screenshots.
- [x] Add Host, Origin, CSRF, MIME, no-store, frame-ancestor, and CSP headers even on loopback.
- [x] Run gateway/capability tests and package-wheel tests.
- [x] Commit: `feat(canvas): add the loopback browser gateway`

### Task 4.2: Build the preview-first shell and revision UX

**Files:**

- Create: `tldw_chatbook/Canvas/static/canvas_shell.html`
- Create: `tldw_chatbook/Canvas/static/canvas_shell.css`
- Create: `tldw_chatbook/Canvas/static/canvas_shell.js`
- Create: `tldw_chatbook/Widgets/Console/console_canvas_card.py`
- Modify: `tldw_chatbook/Widgets/Console/console_transcript.py`
- Modify: `tldw_chatbook/Chat/console_message_actions.py`
- Create: `Tests/Canvas/browser/test_canvas_native_flow.py`

- [x] Write a Playwright flow before UI implementation: tool create auto-opens, update hot-reloads, `Updated · Undo / View previous` appears, exact revision remains pinned, following URL tracks active branch, and branch switch changes the visible head.
- [x] Build a trusted, accessible shell with Canvas selector, editable title, Temporary badge, revision/provenance controls, follow/pin state, source inspect/copy/download, reload, close, connection status, compatibility notices, and scripts-disabled recovery.
- [x] Keep one selected Canvas per Console session. A following view updates only if session/conversation/Canvas/branch still match; otherwise show a new-version notice without redirecting the user.
- [x] Add transcript cards that reopen the exact originating revision and an explicit control to return to the branch-following head.
- [x] Detect assistant HTML fenced blocks through the parsed message model, not regex over rendered Markdown. Add idempotent `Open in Canvas` and `Open as new`; incompatible documents produce a prefilled repair request rather than partial execution.
- [x] Open the first native Canvas with Textual's supported URL-opening API. If the platform cannot open it, show/copy the loopback URL without blocking the terminal.
- [x] Run the Playwright native flow and focused transcript/message-action widget tests.
- [x] Commit: `feat(canvas): deliver preview-first native UX`

### Task 4.3: Add confirmed submit and download actions

**Files:**

- Modify: `tldw_chatbook/Canvas/models.py`
- Modify: `tldw_chatbook/Canvas/gateway.py`
- Modify: `tldw_chatbook/Canvas/static/canvas_shell.js`
- Modify: `tldw_chatbook/Canvas/static/canvas_runtime_worker.js`
- Modify: `tldw_chatbook/Chat/console_canvas_controller.py`
- Modify: `Tests/Canvas/browser/test_canvas_native_flow.py`

- [x] Add tests for text/JSON submit, JSON depth and byte limits, complete confirmation display, cancel, stale session, changed composer, exact-session routing, replay, expiry, and two simultaneous Canvas windows.
- [x] Implement `canvas.submit(value)` as a virtual-runtime request only. Serialize canonical JSON or text, show the complete bounded payload in trusted UI, and after confirmation insert it into the exact matching Console composer as an unsent draft. Never auto-send.
- [x] Add tests for allowlisted passive formats, filename sanitization, MIME enforcement, encoded and decoded byte caps, cancel, replay, and runnable HTML warning.
- [x] Implement `canvas.download({filename, mime_type, data})` as a confirmed request. Allow only literal UTF-8 text/CSV/JSON and signature-checked PNG/JPEG/GIF/WebP data URLs; reject SVG and every active, executable, archive, or ambiguous format. Trusted code owns the browser download and revokes object URLs.
- [x] Default source download to an inert `.canvas.html.txt`. Offer runnable `.html` only behind a clear warning that it executes outside Chatbook's sandbox and bypasses Canvas zero-egress protections.
- [x] Reuse single-use action-scoped capabilities and reject worker messages not tied to the current frame/revision.
- [x] Run gateway, capability, Console controller, package-integrity, zero-egress, and native Chromium tests.
- [x] Commit: `feat(canvas): confirm bridge and download requests` (plus independently reviewed hardening commits).

### Delivery 4 checkpoint

- [x] Update TASK-31229 with screenshots, keyboard/accessibility results, browser-open behavior, capability lifetime, and end-to-end commands.
- [x] Manually verify the outermost path in a real terminal: assistant creates Canvas, browser opens, user interacts, confirms submit, and text appears unsent in the correct composer.

---

## Delivery 5 — Add same-origin served Canvas and remote authentication (TASK-31230)

### Task 5.1: Define the private parent/child protocol

**Files:**

- Create: `tldw_chatbook/Canvas/control_protocol.py`
- Modify: `tldw_chatbook/Web_Server/serve.py`
- Modify: the app startup path in `tldw_chatbook/app.py`
- Create: `Tests/Canvas/test_control_protocol.py`
- Modify: focused `Tests/Web_Server/` tests

- [x] Inspect the installed/pinned textual-serve `Server`/`AppService` process-spawn API and capture a compatibility test before choosing the override point. Do not patch a shell command string or minified JavaScript.
- [x] Define a versioned length-bounded protocol with explicit message types for scope snapshot, list/read render metadata, selection, events, bridge request/decision, health, and shutdown. Reject unknown versions, types, fields, oversized frames, and out-of-order replies.
- [x] Give each AppService child a random secret through the supported spawn environment and listen only on a parent-owned loopback endpoint. Authenticate before transmitting any conversation metadata; rotate/revoke on child restart.
- [x] Implement request IDs, deadlines, backpressure, cancellation, and bounded errors. The parent is a transport/UI host; the child remains the only conversation and Canvas authority.
- [x] Prove two child processes cannot authenticate as or receive events for one another.
- [x] Run protocol and focused textual-serve lifecycle tests.
- [x] Commit: `feat(canvas): connect served parent and child securely`

### Task 5.2: Protect the complete remote origin

**Files:**

- Create: `tldw_chatbook/Canvas/web_auth.py`
- Modify: `tldw_chatbook/Web_Server/serve.py`
- Modify: `tldw_chatbook/config.py`
- Modify: the canonical config defaults/example
- Create: `Tests/Canvas/test_web_auth.py`
- Modify: focused Web Server tests

- [x] Add table-driven tests for IPv4/IPv6 loopback, wildcard/private/public binds, missing token, environment/config/keyring precedence, plaintext remote bind, trusted proxy allowlist, malformed forwarded headers, Host/Origin mismatch, CSRF, websocket upgrade, expiry, revocation, and rate limits.
- [x] Add a dedicated Chatbook web access token setting. Never reuse provider API keys, MCP credentials, or legacy server tokens; never log the configured value.
- [x] Refuse non-loopback bind without the token. Refuse non-loopback plaintext HTTP by default; permit only an explicit warned insecure-development override. Document direct TLS and trusted reverse-proxy deployment.
- [x] Implement one-time login bootstrap, opaque in-memory sessions, `HttpOnly`/`Secure`/`SameSite=Strict` cookies, CSRF tokens, Host/Origin validation, websocket checks, idle/absolute expiry, revocation, constant-time secret comparison, and bounded rate limiting.
- [x] Apply middleware to every authority-bearing route on the origin, including `/`, `/ws`, Canvas APIs/events, downloads, and static boot data. Static immutable assets may be public only if they contain no runtime state.
- [x] Trust forwarded scheme/host/client data only from explicitly configured proxy addresses.
- [x] Run web-auth and existing served-mode tests.
- [x] Commit: `feat(server): authenticate remote Chatbook sessions`

### Task 5.3: Mount the same-origin split-pane shell

**Files:**

- Modify: `tldw_chatbook/Web_Server/serve.py`
- Create/modify: owned served-shell template and styles under the Web Server or Canvas package
- Modify: `tldw_chatbook/Canvas/gateway.py` to share route handlers
- Create: `Tests/Canvas/browser/test_canvas_served_flow.py`

- [x] Add Playwright tests at narrow and wide viewports for terminal-only state, split view, close/reopen, hot reload, active branch switch, exact transcript reopen, connection loss, and terminal survival.
- [x] Serve terminal and Canvas as sibling regions from an owned responsive template on the same origin. Reuse trusted Canvas handlers; do not duplicate compiler/runtime or embed a second localhost URL.
- [x] Route each browser session only to its authenticated AppService child and child-issued Canvas scope. A URL or ID from another browser must return indistinguishable not-found/unauthorized behavior.
- [x] On control-channel loss, disable Canvas and show reconnection state while the Textual websocket remains usable. Never fall back to a global or most-recent conversation.
- [x] Test two browser profiles, two AppService children, guessed IDs, copied exact URLs, event streams, submits, and downloads for cross-session isolation.
- [x] Run served Playwright, control protocol, authentication, and existing textual-serve compatibility suites.
- [x] Commit: `feat(canvas): add the authenticated served split view`

### Delivery 5 checkpoint

- [x] Update TASK-31230 and ADR-121 with the actual textual-serve extension seam, protocol version, authentication flow, proxy policy, and isolation evidence.
- [x] Perform one real authenticated remote/proxy flow and one two-browser isolation flow through the user-visible server, not only handler tests.

---

## Delivery 6 — Round-trip Canvas through Chatbook archives (TASK-31231)

### Task 6.1: Specify Chatbook archive 3.0 Canvas records

**Files:**

- Modify: `tldw_chatbook/Chatbooks/chatbook_models.py`
- Create: `tldw_chatbook/Canvas/archive.py`
- Create/modify: focused Chatbook model tests
- Modify: archive format documentation

- [x] Add failing serialization/validation tests for Canvas identity, revisions, parent graph, titles, runtime profile, digest, byte count, origin message/turn, deletion metadata, and reopen hints.
- [x] Add `ChatbookVersion.V3` and typed Canvas manifest records. Keep V1/V2 decoding unchanged.
- [x] Define inert archive paths such as `canvas/<canvas-id>/<revision-id>.html.txt`; never use runnable `.html` entries or render while inspecting/importing.
- [x] Make archives with Canvas select 3.0; archives without Canvas may remain 2.0 for compatibility.
- [x] Document every new field, limit, ID-remapping rule, and unsupported-runtime behavior.
- [x] Run focused Chatbook model tests.
- [x] Commit: `feat(chatbooks): define Canvas archive format 3`

### Task 6.2: Export and import the complete graph atomically

**Files:**

- Modify: `tldw_chatbook/Chatbooks/chatbook_creator.py`
- Modify: `tldw_chatbook/Chatbooks/chatbook_importer.py`
- Modify: `tldw_chatbook/Chatbooks/local_chatbook_service.py`
- Modify: `tldw_chatbook/Canvas/archive.py`
- Modify: `tldw_chatbook/Canvas/repository.py`
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py`
- Create: `tldw_chatbook/DB/migrations/chachanotes_v66_to_v67_canvas_runtime_profiles.sql`
- Create/modify: focused Chatbook creator/importer tests

- [x] Add a whole-graph round-trip fixture with multiple Canvases, title changes, sibling branches, historical origins, deletions, and reopen hints. Assert exact source/digest/ancestry equality.
- [x] Export source from the repository in bounded chunks, recompute digest/size, and emit deterministic ordering and timestamps without executing or compiling source.
- [x] Before extraction/import, validate archive entry count, path containment, duplicate normalized paths, declared and streamed uncompressed sizes, compression ratio, aggregate limits, UTF-8, digests, duplicate IDs, graph cycles, parent ownership, origin-message existence, and runtime profile.
- [x] Implement digest-idempotent same-identity restore. Refuse same-ID/different-digest conflicts with no mutation.
- [x] Implement import-as-new by precomputing maps for conversation, messages, Canvas, revisions, parents, origins, and hints, validating the remapped graph, then committing all records in one transaction.
- [x] Inject failures at validation, file streaming, message import, Canvas import, and final commit; assert no partial imported graph remains.
- [x] Keep unsupported profiles inert and labeled; never compile them using the current profile.
- [x] Migrate schema 66 to 67 so well-formed bounded unknown runtime-profile identifiers can be stored inert, while execution remains restricted to explicitly supported profiles; prove genuine-v66 migration, rollback, and fresh-schema parity.
- [x] Verify V1/V2 golden archives still behave identically and no Canvas data enters synchronization services.
- [x] Run focused Chatbook, repository, decompression-bomb, property, and transaction tests.
- [x] Commit: `feat(chatbooks): round-trip Canvas histories`

### Delivery 6 checkpoint

- [x] Update TASK-31231 with archive examples, limits, atomicity evidence, and compatibility results.
- [x] Inspect a produced archive manually to confirm source is inert and all manifest relationships are understandable without executing content.

---

## Delivery 7 — Complete settings, documentation, and cross-mode verification (TASK-31232)

### Task 7.1: Add canonical settings and the kill switch

**Files:**

- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`
- Modify: `tldw_chatbook/config.py`
- Modify: canonical default/example config
- Modify: Console tool and message-action registration seams
- Modify: native and served route startup seams
- Create/modify: focused Settings and configuration tests

- [x] Add failing tests for defaults, environment/config precedence, invalid limits, live-disable behavior, and disabled startup in native/served modes.
- [x] Add settings only to the canonical F9 Settings surface: enabled, auto-open on create, remote access policy/status, and read-only effective hard quotas. Do not add new controls to deprecated settings widgets.
- [x] Implement one kill switch whose disabled state removes Canvas tool schemas, hides HTML-block actions, revokes browser/control capabilities, returns fail-closed route responses, and leaves stored artifacts/export intact.
- [x] Keep security ceilings non-increasable from ordinary UI. Any advanced lower-limit overrides must validate through `CanvasLimits` and apply consistently after restart.
- [x] Run focused config, Settings screen, tool registration, and route tests.
- [x] Commit: `feat(canvas): add settings and a global kill switch`

### Task 7.2: Measure and freeze conservative defaults

**Files:**

- Create: a reproducible Canvas benchmark/probe script under `scripts/`
- Modify: `tldw_chatbook/Canvas/limits.py`
- Modify: `Docs/superpowers/specs/2026-09-03-chatbook-canvas-design.md`
- Modify: Canvas user/security documentation

- [x] Build probes for representative synthetic assistant-authored pages and adversarial near-limit pages. Label fixture provenance explicitly; this qualification does not claim live-provider sampling. Record compiler latency, render-plan expansion, QuickJS heap/stack, startup/event interruption accuracy, patch throughput, and browser process memory.
- [x] Run probes on the repository's supported baseline environment and save summarized, non-source evidence. Do not include model outputs that may contain user data.
- [x] Lower initial quotas where needed to keep compile/startup under the intended 100 ms worker threshold or browser interaction responsive. Do not raise a security ceiling without separate review. Remaining compiler scheduling is explicitly gated by Task 7.2a.
- [x] Add boundary tests for the final values and document what users see when each quota is exceeded.
- [x] Commit: `perf(canvas): freeze measured runtime quotas`

### Task 7.2a: Resolve measured compiler blocking at interactive boundaries

**Reason added:** Task 7.2 measured valid near-limit compilation at roughly
208 ms for a full node plan and 371 ms for combined node/CSS/script limits on
the qualification host. Native preview, served preview, and HTML-block import
can execute this work on the UI/server event loop. This exceeds the repository's
100 ms worker threshold and is a rollout blocker, not a documentation caveat.
The stricter review-fix probe at the final 1,800-node/900-rule ceilings measured
107.189 ms median and 124.874 ms maximum under host load, confirming that lower
quotas alone do not remove the scheduling requirement.

**Files:**

- Modify as needed: `tldw_chatbook/Canvas/native_authority.py`, `service.py`, `gateway.py`
- Create if needed: `tldw_chatbook/Canvas/compilation.py` (bounded pure-compilation admission only)
- Modify as needed: `tldw_chatbook/Chat/console_canvas_controller.py`
- Modify as needed: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify as needed: `tldw_chatbook/Web_Server/serve.py`
- Create/modify: focused compiler-scheduling, lifecycle, and browser tests

- [x] Use the completed Task 7.2 evidence and final quotas to identify remaining interactive compilation costs; keep pure compilation off UI/server event loops and out of locks those loops must acquire.
- [x] Preserve captured conversation/message/branch identity across background work; revalidate lifecycle, expected-parent, and capability authority before publishing or mutating after an await. Cancellation, disable, branch switch, and disposal must not publish late results or revive authority.
- [x] Reuse existing worker/authority ownership where possible, bound admitted compilation work, and avoid a second mutation owner or an unbounded source cache. Any prepared result must remain bound to the exact validated source and runtime profile.
- [x] Add failing event-loop responsiveness and delayed-compile race tests for native/served previews and HTML-block import; also cover shared controller-lock behavior where the measured call path requires it.
- [x] Run focused tests plus a content-free near-limit responsiveness probe, preserving all strict-zero-egress and archive/source invariants. Document measured results and any remaining limits.
- [x] Commit: `fix(canvas): keep compilation off interactive event loops`

### Task 7.3: Write user, model, operations, and recovery guidance

**Files:**

- Modify: relevant user documentation under `Docs/`
- Modify: `tldw_chatbook/Web_Server/README.md` (remote-auth and privilege guidance)
- Modify: Console model/tool guidance
- Modify: `backlog/decisions/121-local-versioned-canvas-artifacts-and-browser-sandbox.md`
- Modify: TASK-31003 only if the implemented local archive boundary changes its assumptions

- [x] Document creation/update, multiple names, Temporary state, save/promotion, active branches, exact revisions, Undo/View previous, source copy/download, confirmed submit/download, compatibility errors, and scripts-disabled recovery.
- [x] Document strict zero egress accurately: generated code has no network/filesystem/cookies/Chatbook/parent DOM, while trusted user-confirmed host actions are outside that runtime. Do not market the system as a general browser sandbox.
- [x] Document remote token setup, TLS or trusted-proxy requirements, the insecure-development override, session revocation, and incident response.
- [x] Tell models when Canvas materially helps, how to call list/read/create/update, that updates are complete documents with expected parent IDs, and which V1 APIs are supported.
- [x] Keep V2 bundled libraries, V3 multi-file VFS, and server synchronization explicitly deferred; do not add speculative compatibility code.
- [x] Commit: `docs(canvas): explain the v1 workflow and security model`

### Task 7.4: Run outermost-path verification and close the rollout

**Files:**

- Modify: `Tests/Canvas/browser/test_canvas_native_flow.py`
- Modify: `Tests/Canvas/browser/test_canvas_served_flow.py`
- Modify: `Tests/Canvas/browser/test_canvas_zero_egress.py`
- Create as needed: focused live-harness helpers/fixtures under `Tests/Canvas/browser/`
- Modify as needed: `tldw_chatbook/Canvas/gateway.py`, `tldw_chatbook/Canvas/static/canvas_shell.js`, `tldw_chatbook/Web_Server/serve.py`, and focused gateway/served-state/browser tests to repair the reproduced served exact-card selection delivery gap and redundant parent synchronization. Preserve exact-selection capability revocation, historical pinning, shell ownership and stale-response fences; do not reload a consumed bootstrap or weaken authentication. Passive synchronization of an already-applied scope must remain distinct from explicit same-revision selection intent. ADR-121 already governs this behavior; no new architecture is authorized.
- Modify as needed: `Tests/Chatbooks/test_chatbook_canvas_round_trip.py`
- Modify: `Tests/DB/test_chachanotes_v65_trace_compaction_migration.py` (remove the stale current-schema literal after the reviewed Canvas schema 67 migration; preserve the genuine v64 upgrade fixture and compaction assertions)
- Modify: `Tests/Chat/test_console_semantic_mutation_inventory.py` and `Docs/Development/console-semantic-mutation-inventory.md` (synchronize exact census totals and owner documentation for the two already-classified Canvas routes; retain the bidirectional structural checks)
- Modify: TASK-31232 with final evidence and notes
- Modify: `tldw_chatbook/Canvas/control_protocol.py`, `tldw_chatbook/Canvas/capabilities.py`, the already scoped gateway/served-parent/shell, and focused protocol/capability/gateway/browser tests to fence a queued navigation against its original selection intent before child mutation. Preserve a child-owned opaque generation through scope/capability/bootstrap round trips and validate the browser's issued selection epoch; explicit same-revision pin changes intent, passive snapshots do not. Missing served expectations fail closed. See ADR-121's selection-intent amendment; do not add a legacy served bypass.
- Modify: `backlog/docs/lessons-*.md` only if this work produced a repeatable, incident-backed lesson

- [x] Native live flow: create, automatic browser open, interact, update/hot reload, submit to unsent draft, passive download, exact revision reopen, historical branch update, temporary promotion, and unsaved destruction.
- [x] Served live flow: authenticated login, sibling split view, create/update, branch switch, exact card reopen, control-channel failure, reconnect, proxy/TLS configuration, and two-browser isolation. Verify durable exact-revision recovery by explicitly loading the saved conversation and opening its persisted card in a fresh authenticated child; distinguish this from fresh temporary replacement and automatic resume.
- [x] Archive flow: export a branching Canvas conversation, delete/purge the source as appropriate in a disposable database, import, and verify graph/source/digests/reopen behavior.
- [x] Security flow: rerun the adversarial real-browser suite through native and served outer routes while recording zero attempted egress at the harness boundary.
- [x] Run targeted Canvas, Agents, Console, database migration, Chatbooks, Web Server, packaging, and browser suites. Run formatter/linter only over changed files. Run `git diff --check`. Final incremental evidence reconciliation is in `Docs/Canvas/V1_VERIFICATION.md`; this is not a newly executed full matrix or pristine warning/static-debt claim.
- [x] Ask the user whether they want the full repository test sweep. Do not silently substitute the targeted result for a full-suite claim.
- [x] Perform a final self-review against every design invariant and every TASK-31226 through TASK-31232 acceptance criterion.
- [x] Mark each task Done only after its own Definition of Done is met. TASK-31003 remains To Do as the explicit future sync-contract backlog item.
- [x] Commit: `test(canvas): verify native and served v1 workflows` (initial verification `f41d8ca22a`, selection fixes `0724726a0c`, durable recovery `4ce7fc756f`; subsequent whole-branch findings and scoped corrections are closed in `Docs/Canvas/V1_VERIFICATION.md`).

---

## Final whole-branch correction wave (2026-09-05)

ADR required: no new ADR; existing ADR-121 applies.
ADR path: `backlog/decisions/121-local-versioned-canvas-artifacts-and-browser-sandbox.md`
Reason: these corrections enforce the approved resource, runtime, disabled-mode,
scheduling and historical-selection contracts; they do not introduce a new
storage, permission, provider or runtime boundary.

Final review of `e4652f9d37..facd1e0fb0` requires five Important corrections and
two Minor corrections. Earlier task-level evidence remains historical evidence,
not satisfaction of these newly identified production seams. TASK-31232 AC3 is
reopened and AC10 records the corrective outcomes; AC9 remains unsatisfied.

- [x] Enforce count/revision/byte quotas in the actual Console mutation owner
  before staging, including existing/concurrent history and failure release.
- [x] Preserve bounded DOM identity for move/reinsert and reject cycles atomically.
- [x] Preserve ordinary non-opt-in continuation bytes and assistant text while
  retaining fail-closed sensitive Canvas projection in mixed rounds.
- [x] Discover transcript HTML-fence identity cheaply and compile compatibility
  only through bounded off-loop validation, retaining stale-message fences.
- [x] Keep valid historical pins on ordinary publication; preserve explicit
  selection and Follow semantics, including same-revision pin intent.
- [x] Validate asset helper arguments before iteration using safe-wire integers;
  distinguish inner explicit Close from outer Hide and browser-tab closure.
- [x] Run exact RED/GREEN and affected targeted/statics checks, then one scoped
  final rereview. No second broad review, unrelated baseline fix or full sweep.

Correction `c875bad60f`, scoped rereview through `a7bcc6b094`: I3/I5/M1/M2
addressed; I1/I2/I4 remain open. Residuals are the default temporary-session
8 MiB cap, detached-node edits and empty/false form-state reconstruction, and
late compile-refusal repair after disable. Existing targeted passes do not
cover these cases. The one-wave limit is reached; preserve the worktree and
request explicit authorization for another bounded pass rather than waive
these required behaviors. See `Docs/Canvas/V1_VERIFICATION.md` for exact
evidence, native-versus-served qualifications and independent baseline limits.

### User-authorized additional focused pass

The user explicitly approved one additional pass after the residual I1/I2/I4
handoff. This supersedes the preceding pause, not the product contracts or
the exclusion of unrelated baseline fixes/full-suite work.

ADR required: no new ADR; existing ADR-121 applies.
ADR path: `backlog/decisions/121-local-versioned-canvas-artifacts-and-browser-sandbox.md`
Reason: enforce the existing temporary resource, virtual DOM and stale-effect
contracts without changing storage, permissions or runtime boundaries.

- [x] Enforce exact default 8 MiB temporary admission, including confirmed
  temporary history, concurrent stages, multiple turns, import/rename and abort
  release, while preserving the separate durable ceiling.
- [x] Support detached text/attribute/property/style and subtree edits before
  reinsertion with bounded transactional state; restore supported empty/false
  form values after attributes. Verify through the actual browser renderer.
- [x] Suppress late compile-refusal repair effects after disable or changes to
  the captured owner/session/source block; retain the valid repair control and
  successful import fences with no retained source cache.
- [x] Record exact RED/GREEN, final affected tests/statics/asset integrity and
  owned browser cleanup, then obtain one scoped author-independent rereview.

Implementation `1467bdf0a6`, scoped rereview through `648530ac6`: I1 and I4
addressed; I2 still requires child-dependent select-value restoration after
options exist and per-descendant presence handling during reconstruction.
The latter prevents duplicate creates when a new or live child is added to
an already-detached parent before reinsertion. These are required supported
DOM behaviors, not harmless waivers. The authorized pass is exhausted; retain
the worktree/evidence and request a specifically scoped DOM-only continuation.
TASK-31232 AC3 is now checked; AC9 and AC10 remain open.

### User-authorized DOM-only correction

The user explicitly approved one DOM-only correction and scoped rereview after
the two remaining I2 cases. No quota/repair reopening, baseline repairs, broad
review, full sweep, merge or cleanup is authorized.

ADR required: no new ADR; existing ADR-121 applies.
ADR path: `backlog/decisions/121-local-versioned-canvas-artifacts-and-browser-sandbox.md`
Reason: correct supported DOM reconstruction under the existing bounded runtime.

- [x] Restore explicit select values after their options exist, including a
  non-first option and empty/no-match selection, preserving default-only state.
- [x] Reconstruct mixed-presence subtrees without duplicate native IDs when a
  new or live child joins an absent parent; preserve attachments and limits.
- [x] Add actual-renderer failing regressions, verify final targeted tests and
  asset integrity with owned browser cleanup, and obtain one scoped rereview
  from the previously reviewed HEAD `648530ac6`.

Implementation `981b1f8c1`, scoped rereview through `03cd979df`: both remaining
I2 cases addressed; spec and quality gates pass with no new Critical/Important
fix-diff issue. Exact final browser evidence is 5 passed; runtime assets 20
passed with one optional archive-input skip. See `Docs/Canvas/V1_VERIFICATION.md`
for commands and qualifications. All final-review implementation findings are
closed, and TASK-31232 AC10 is checked. AC9 remains open for six characterized
pre-existing selected-suite failures; do not mark Done or infer authorization
for those repairs, a full sweep, merge or cleanup. Preserve the branch and
review evidence pending separate direction on that acceptance blocker.

### User-authorized six-baseline repair

The user explicitly approved addressing the six failures listed in
`Docs/Canvas/V1_VERIFICATION.md`, after the clean DOM handoff. This supersedes
the preceding baseline exclusion only for those failures and their directly
affected regression coverage. It does not authorize a full suite or integration.

ADR required: no new ADR; existing contracts apply.
ADR paths: `backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md`,
`backlog/decisions/079-console-library-conversation-authority.md`,
`backlog/decisions/094-raw-and-virtual-cli-execution-boundaries.md`, and ADR-121.
Reason: repair existing behavior or stale fixtures, without introducing storage,
permission, provider or runtime boundaries. The later ADR-097 accepted spec and
TASK-23113.2 AC8 explicitly require retained semantic bytes on soft deletion;
do not restore older ADR-090 sidecar clearing to make a stale assertion pass.

- [x] Reproduce all six exact IDs and classify product defects versus fixture
  drift using current governing contracts, without weakening mutation guards.
- [x] Repair image-only send preparation, MCP exclusion coverage, legacy
  promotion-double fidelity, guarded future-thinking export setup, soft-delete
  retention coverage and Settings navigation readiness as diagnosis requires.
- [x] Run exact RED/GREEN plus directly affected targeted tests/statics, review
  the repair diff independently, and record any new or unresolved failure.
- [x] Reconcile AC9 against the selected-suite evidence without presenting
  focused passes as a newly executed full matrix or waiving outstanding checks.

Implementation `11ea68221`, scoped review through `610adcbaf`: all six repairs
pass spec and quality review, with one nonblocking adjacent stale-comment nit.
The exact six cases pass; wider affected coverage is 737 passed / 2 failed.
Those two Canvas retry cases also fail at pre-repair `c8a1211e5`; they are not
introduced by this repair, but remain acceptance blockers. AC9 is still open;
the task is not Done. See `Docs/Canvas/V1_VERIFICATION.md` for exact comparison,
isolation incident, static-debt and resource-warning qualifications. Preserve
the worktree and seek separate direction on the newly identified retry cases;
do not infer a full sweep, integration or another repair scope from this review.

### User-authorized retry correction

The user approved addressing the two remaining failed-assistant retry cases.
This covers their shared cause and directly affected regressions, not unrelated
repairs, a full repository sweep, integration or permission changes.

ADR required: no new ADR; existing ADR-121 applies.
ADR path: `backlog/decisions/121-local-versioned-canvas-artifacts-and-browser-sandbox.md`
Reason: restore or correctly verify existing exact-run settlement, atomic message/
artifact commit, retry and staged-history cleanup contracts without new boundaries.

- [x] Reproduce both failing retry parameter cases, trace settlement ownership
  through failure, successful retry and persistence, and distinguish missing
  durable results from legitimate staging cleanup or stale test expectations.
  Source-free observation identified a native-only SYSTEM failure notice included
  in the durable tool scope. Omit only that class of notice for durable sessions;
  preserve persisted messages, temporary native paths and service validation.
- [x] Correct the cause with regression evidence while preserving stale-run
  fencing, discarded failed revisions, atomic rollback and restart hydration.
- [x] Run the complete retry parameter group and affected Canvas/Console checks,
  static analysis and independent scoped review, then reconcile AC9 honestly.
  Final product `5bba89d3a`; affected 970 tests pass, final focused 11 pass;
  independent review of `b125832cb..8388ae696` passes both gates. Existing
  dependency/resource warnings and inherited static debt remain documented.

All executable tests remain coordinator-owned isolated pytest runs; the worker
is limited to static inspection, edits and exact git operations. Preserve prior
isolation incident records and the nonblocking stale-comment observation.

## Execution Handoff

### User-authorized integration and V2 handoff (2026-09-05)

The user now authorizes investigation of the descriptor warning, a PR against
`dev`, rebasing onto latest `dev`, addressing all Qodo issues/comments once
posted, merging the reviewed PR, and then starting V2. This supersedes earlier
integration exclusions, not the sandbox, isolation or targeted-test policy.

ADR required: no new ADR for integration or direct cleanup repairs.
ADR path: `backlog/decisions/121-local-versioned-canvas-artifacts-and-browser-sandbox.md`
Reason: preserve the approved V1 boundary; V2 library/runtime decisions require
their own design and ADR check before implementation.

- [x] Complete TASK-31732: reproduce/classify the FD-growth signal, correct a
  proven in-scope leak if present, verify and independently review the result.
- [x] Preserve a recoverable pre-rebase ref, fetch latest `origin/dev`, inspect
  exact feature ancestry, and rebase only the feature range. Resolve conflicts
  without losing either side's unrelated work; recheck schema/ADR/task-ID and
  derived-artifact collisions. Prefer rebasing before the first public push.
- [x] Run preflight and affected verification on the rebased tree. No full
  repository sweep without explicit approval. Create the PR against `dev`,
  with the retained verification and warning qualifications.
  TASK-31741 owns the discovered diagnostic-inventory and index-census guard
  gaps, including content-free Canvas failure logging and no-statistics plans.
- [ ] Address Qodo on PR2432 in this active session. A proposed unattended
  follow-up was rejected; persistent scheduling needs explicit user approval.
  Read review bodies, inline threads, issue comments and suggestions; maintain
  an item-by-item resolution ledger. Verify suggestions technically, implement
  valid corrections with tests, and reply with evidence to each applicable
  thread. Do not treat silence as a clean review or bypass protected checks.
- [ ] Recheck current PR head/base, completed Qodo feedback and required CI,
  update from `dev` and retest affected conflicts if necessary, then merge via
  the repository's allowed method without an admin/protection bypass. Verify
  merged state and commit identity. Retain the worktree/evidence; do not touch
  the user's main checkout or delete branches as an implied cleanup step.
- [ ] Only after merge, begin V2 brainstorming: a small offline bundled library
  catalog under the existing zero-egress guarantee. Do not silently add network,
  filesystem, cookies or multi-file VFS; settle design and ADR before coding.

The plan is organized as seven independently reviewable Backlog tasks. Delivery 1 is a hard security gate; Delivery 2 establishes storage; Delivery 3 integrates agent turns; Deliveries 4 and 5 add native and served UX; Delivery 6 can proceed after Delivery 2 without waiting for browser UX; Delivery 7 closes rollout.

Choose one execution mode when implementation begins:

1. **Subagent-Driven (recommended):** execute one Backlog task at a time in this session with a fresh implementation worker and review checkpoint for each task.
2. **Inline:** execute the plan serially in a dedicated session using `superpowers:executing-plans`, stopping at every delivery checkpoint.

In either mode, create a clean worktree/branch from current `origin/dev`, preserve the task dependency order, and stop immediately if the strict zero-egress proof fails.
