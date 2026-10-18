# Canvas

This document describes Canvas: the conversation-owned, local-first artifact system — agent-authored self-contained HTML documents stored as immutable revision graphs, rendered in a hard-sandboxed browser runtime served from a loopback gateway. Canvas is not a Textual drawing widget, and it is unrelated to the Library screen's browse "canvases" (a naming collision).

## Authoritative files

| Concern | File | Key symbols |
| --- | --- | --- |
| Limits | `Canvas/limits.py` | pure constants + validators: 10 canvases/conversation, 100 revisions/canvas, 512 KiB durable source/revision, 50 MiB/conversation, 8 MiB temporary staging/session, 4 KiB titles |
| Models | `Canvas/models.py` | `CanvasRenderPlan`, `CanvasDiagram` (mermaid), `CanvasRenderPlanV2` (`canvas-v2-mermaid-1`), `CanvasScope`, `CanvasRevisionInfo`, `CanvasConflictResult` |
| Compiler | `Canvas/compiler.py`, `compilation.py` | `compile_canvas_document` — compiles untrusted HTML into a closed render plan under profiles `canvas-v1` / `canvas-v2-mermaid-1`; validates `data-canvas-diagram="mermaid"` elements |
| Profiles | `Canvas/profiles.py` | immutable runtime-profile catalog; unknown/revoked profiles degrade to `profile-unavailable` (source stays readable) |
| Service | `Canvas/service.py` | `CanvasService` — durable branch-aware list/read/create/import/update/rename |
| Repository | `Canvas/repository.py` | `CanvasRepository.append_batch_in_transaction` — typed persistence |
| Staging | `Canvas/staging.py` | process-only temporary histories + atomic promotion snapshots |
| Console wiring | `Chat/console_canvas_controller.py` | `ConsoleCanvasController`, `CanvasRunCoordinator`, `CanvasTurnContribution.write_exact`, run states OPEN/READY/COMMITTED/DISCARDED |
| Tool provider | `Agents/canvas_tool_provider.py` | `CanvasToolProvider`, `CanvasToolRegistrationAuthority`, the `CanvasToolCoordinator` protocol, runtime guidance; module-level `build_canvas_tool_catalog()` / `load_canvas_tool_schema()` for catalog listing without a live scoped provider |
| Registry gate | `Agents/tool_catalog.py` | `CANVAS_RESERVED_TOOL_NAMES`, `register_canvas_provider` (exact-type + live-authority + current-scope + exact 5-name catalog, else refuses), `_authenticated_canvas_name` |
| Gateway | `Canvas/gateway.py` | loopback aiohttp app (`/api/boot`, `/api/frame`, `/render`, `/api/bridge/*`, …); refuses non-loopback hosts and non-zero ports |
| Capabilities | `Canvas/capabilities.py` | `CanvasCapabilityStore` — single-use, 300 s TTL capabilities per action (shell_boot, renderer_load, render_plan, source_read, …) |
| Native authority | `Canvas/native_authority.py` | app-owned browser authority for the native Console; gated auto-open receipts |
| Runtime assets | `Canvas/runtime_assets.py`, `Canvas/static/`, `Canvas/mermaid/` | SHA-256-manifest-verified QuickJS runtime bundle and renderer, trusted shell, vendored Mermaid subset; CI independently pins the shell assets |
| Build inputs | `vendor/canvas-mermaid-inputs/`, `scripts/check_canvas_mermaid_assets.py` | exact original pinned sources for offline CI/preflight; explicit fetch-only refresh, outside installed package data (ADR-238) |
| Served mode | `Canvas/control_protocol.py`, `web_auth.py`, `Web_Server/serve.py` | authenticated control channel for child processes; origin-wide browser auth (token env/keyring, session cookie, CSRF header) |
| Persistence | `DB/migrations/chachanotes_v66_to_v67_canvas_revisions.sql`, `DB/canvas_payload_validation.py`, `DB/ChaChaNotes_DB.py` | `canvas_documents`/`canvas_revisions`/`canvas_conversation_hints` + 6 triggers; insert-time sha256/bytes CHECK via a custom DB function |
| Message cards | `Chat/message_metadata.py` | `CanvasCardMetadata` / origin metadata |

## Tools

Five tools, all reserved-prefixed: `canvas_list`, `canvas_read`, `canvas_create`, `canvas_update`, `canvas_guide` (mutations = create/update; classification `REVERSIBLE_CONVERSATION_LOCAL`). The names stay reserved even when Canvas is unavailable. Guidance is offer-first: the companion skill (`Docs/Examples/skills/canvas/`) is user-invocable only, and bare `$canvas` prompts clarify rather than assume.

## Dataflows

1. **Registration and gating**: the console controller builds one scoped provider per session+run and issues a `CanvasToolRegistrationAuthority` (a frozen binding of provider instance, session, run, classification — weakref-tracked). `register_canvas_provider` accepts only the exact provider type, an exact live authority, a current scope, and the exact 5-name catalog; lookalikes fail closed. Catalog listing filters canvas names through the authenticated-name check.
2. **Tool call → staging → durable**: `canvas_create`/`canvas_update` → enablement check → current run/tool-call identity check (from `run_context`) → strict argument validation → document preparation (profile admission + compile) → the revision stages in a process-only run stage. When the assistant run finishes, the turn contribution maps origin ids and appends the batch **inside the caller-owned turn transaction** — canvas rows land atomically with the assistant message. Settlement publishes to the gateway/native authority so a browser session can load the render plan.
3. **Conflicts**: updates carry `expected_parent_revision_id`; a stale or ambiguous ancestry returns a `CanvasConflictResult` (`stale_parent` / `ambiguous_ancestry`) as a JSON conflict the model must resolve by re-reading.
4. **Render**: the gateway is created lazily; the browser shell obtains single-use capabilities per action, fetches shell/renderer/plan, and executes scripts in QuickJS-WASM. The runtime's only host affordances (`canvas.submit(JSON)` / `canvas.download(JSON)`) *request* a later user-confirmed host action through the bridge; nothing executes without it.
5. **Served mode**: the web server builds a gateway with a served-authority proxy; per-conversation child processes attach over the authenticated control protocol.

## Sandbox and durability guarantees

- **Sandbox**: no network, storage, modules, filesystem, Chatbook API, or parent-DOM access from rendered documents. Runtime assets are manifest-SHA-verified; a damaged package disables execution (`RUNTIME_DISABLED_DIAGNOSTIC`) while source and history remain readable.
- **DB invariants** (trigger-enforced): revisions are immutable; deletes require an app-registered authorization function; ownership cannot change after creation; parent/sequence guards; payload sha256+bytes re-validated at insert. The migration refuses to run if canvas objects already exist (schema-ownership check).
- **Source privacy**: durable/display projections strip HTML — only ids, title, hashes, byte counts, sequence, and origin survive; malformed source-bearing payloads collapse to a content-free error code. Errors pass through a whitelist.
- **Quotas**: enforced in `limits.py` and the repository (`CanvasQuotaError` / `CanvasLimitError`).

## Config keys

`[canvas] enabled` (strict bool — anything but literal true disables, with a diagnostic; an observed disable **latches until restart**), `[canvas] auto_open_on_create`; env overrides `TLDW_CANVAS_ENABLED`, `TLDW_CANVAS_AUTO_OPEN_ON_CREATE`. Served mode auth: `TLDW_CHATBOOK_WEB_ACCESS_TOKEN`.

## Governing decisions

ADR-121 (`121-local-versioned-canvas-artifacts-and-browser-sandbox.md`; note the renumbering from a collided 115), ADR-124 (`124-canvas-mermaid-subset-and-immutable-runtime-profiles.md`), ADR-149 (`149-offer-first-canvas-guide-and-inline-skill.md`; a different ADR-149 also exists — cite by filename). Specs: `Docs/superpowers/specs/2026-09-03-chatbook-canvas-design.md`, `2026-09-06-chatbook-canvas-v2-mermaid-design.md`, `2026-09-10-canvas-guidance-skill-design.md`. Verification records: `Docs/Canvas/V{1,2}_{VERIFICATION,RUNTIME_COMPATIBILITY}.md`.

## Verified gotchas

1. Naming hazard: Library browse widgets are also called "canvases" — unrelated subsystem.
2. Tool-result byte caps: `canvas_read` is bounded at 6×512 KiB + 64 KiB (≈3.06 MiB) per `MAX_CANVAS_TOOL_RESULT_BYTES`, other canvas tools at 64 KiB.
3. The provider's `load_schema` returns a schema object carrying hidden guide/profile fields — provider-owned context, deliberately excluded from ordinary schema serialization.
4. Guidance suppression relies on an exact offer-policy string match in the system prompt; profile-specific guidance is not repeated across history.
5. Conversation history re-parsing keeps per-profile guidance from being re-delivered every turn.

## Related docs

- [tool-catalog.md](./tool-catalog.md) — the authenticated canvas registration seam
- [console.md](./console.md) — the turn transaction canvas rows join atomically
