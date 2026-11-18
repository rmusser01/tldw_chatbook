# Desktop→Phone Approval Relay (Phase 2c) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Approvals raised by desktop terminal sessions (served child App processes) reach the phone: rounds publish over the existing parent↔child control broker, appear in `/m`'s approvals inbox alongside mobile-initiated ones, and decisions route back to the owning child; desktop run completions also push as live cross-surface events.

**Architecture:** Four new message types in the loopback control protocol (`relay.approval.raised` / `relay.approval.cleared` / `relay.run.status` child→parent events; `relay.decide.request`/`.response` parent→child request). A served-child publisher hooks the Console approval seam and forwards rounds through the child's `CanvasControlClient`; the parent's `MobileChatService` subscribes to broker events, merges desktop rounds into `list_approvals`/WS `approval_requested` (marked `surface: "desktop"`), and routes decisions back through `broker.request`. ADR-236 records the contract.

**Tech Stack:** existing (`asyncio` streams, aiohttp, vanilla JS); no new dependencies.

**Spec:** `Docs/superpowers/specs/2026-10-15-served-mode-auth-and-mobile-surface-design.md` §4.3 ("Stated v1 limit" — this phase removes it) + §5 Phase 2
**Spike context:** `Docs/superpowers/spikes/2026-10-15-mobile-engine-binding-spike.md` ("Binding 2 evidence… remains the fallback only if v2 wants phone-side decisions on desktop-running sessions") — this phase implements the relay WITHOUT the headless-child App: the desktop child is the already-running terminal App.

## Global Constraints

- The broker stays loopback-only, secret-authenticated per child (`issue_child` launch secrets), max-frame limits unchanged. New message types extend `_MESSAGE_FIELDS` + `_REQUEST_REPLY` (Canvas/control_protocol.py:60/59) following the existing registry pattern exactly; `relay.*` payloads are JSON-validated dicts (no bytes, depth-limited via the existing `_validate_json`).
- Child→parent events carry: `relay.approval.raised` {round_id, session_id, calls, timeout_seconds}; `relay.approval.cleared` {round_id, resolution}; `relay.run.status` {session_id, status}. Parent→child: `relay.decide.request` {round_id, decisions} → `relay.decide.response` {round_id, delivered: bool}.
- Child-side hooking is SERVED-CHILD-ONLY: the publisher activates only under the existing served-child detection (the `CHATBOOK_SERVED_CHILD` environment wiring in app.py ~1080-1120); native TUI runs never publish. The hook must not alter TUI approval behavior (wrap-and-forward, never replace).
- Parent-side: desktop rounds join `list_approvals()` with `surface: "desktop"` + `child_id`; mobile rounds unchanged (`surface: "mobile"`). `decide_approval` routes by surface: desktop → `broker.request(child_id, "relay.decide.request", ...)`; unknown/dead child → round dropped from the list, `False` returned. WS `approval_requested`/`run_status` events for desktop activity reuse the existing event envelope (`MobileEvent.to_wire()`).
- Security: relay payloads carry tool names/arguments/options — never credentials, cookies, or file contents beyond what the approval card already shows; decisions from the phone ride the authenticated mobile API → authed + CSRF (existing middleware), and the broker hop is loopback + child-secret. The phone's decision authority equals the desktop user's (same operator, ADR-233 posture).
- Failure modes: child disconnect with pending rounds → parent clears its rounds (fail-closed: the child's own timeout/deny semantics govern); broker absent (canvas disabled) → relay inert, desktop approvals invisible on the phone (documented); no event may block the child's approval flow (publish is fire-and-forget with bounded queue).
- Tests: in-process broker + scripted child connection (mirror `Tests/Canvas/` control-protocol tests — locate and reuse their fixtures); environmental baselines name-identical; green sets stay green.
- Worktree `/Users/macbook-dev/Documents/GitHub/tldw-auth-phase1a`, branch `feat/served-mobile-phase2c` off `feat/served-mobile-phase2b` (stacked). Targeted pytest only.

---

### Task 1: ADR-236 + protocol extension

**Files:**
- Create: `backlog/decisions/236-served-desktop-approval-relay.md`
- Modify: `tldw_chatbook/Canvas/control_protocol.py` (`_MESSAGE_FIELDS` += 5 entries; `_REQUEST_REPLY` += relay.decide pair), backlog task file (backlog-py create)
- Test: extend the existing control-protocol test file (locate: `grep -rln "encode_control_frame\|CanvasControlBroker" Tests/ --include="*.py"`)

ADR-236 (summary): context (phone couldn't approve desktop sessions — spec v1 limit); decision (relay message types over the existing loopback broker; child publishes approval rounds + run status; parent merges into the mobile approvals surface; decisions route back; served-child-only hook; fail-closed on disconnect); consequences (phone gains desktop approval authority — same single-operator trust; no cross-process secret handoff; broker inert when canvas disabled); alternatives (polling the child's DB — rejected: approvals are in-process state; a second broker — rejected: one loopback transport with per-child state already exists).

Steps: write ADR + backlog task; failing protocol tests (the 5 new types round-trip through encode/decode; unknown-type rejection still works; relay.decide registers in _REQUEST_REPLY); implement registry entries; green; commit `feat(canvas): relay message types in the control protocol (ADR-236)`.

### Task 2: Child-side relay publisher

**Files:**
- Create: `tldw_chatbook/Web_Server/served_relay_publisher.py` (name mirrors the served-child wiring location; module is importable without Textual app context)
- Modify: `tldw_chatbook/app.py` (served-child wiring block ~1080-1120) and the Console approval seam holder (read `UI/Screens/chat_screen.py::_set_console_pending_approval` + how `controller.set_pending_approval` is bound; wrap at the binding site — the wrapper calls through to the original then publishes)
- Test: `Tests/Web_Server/test_served_relay_publisher.py`

Interfaces:
```python
class ServedRelayPublisher:
    """Forwards served-child Console approval rounds + run status to the parent broker."""
    def __init__(self, client: CanvasControlClient | None) -> None: ...   # None-safe no-op
    def approval_raised(self, payload: Mapping[str, Any]) -> None: ...    # fire-and-forget
    def approval_cleared(self, round_id: str, resolution: str) -> None: ...
    def run_status(self, session_id: str, status: str) -> None: ...
```
Wire-up: in the served-child block, after the existing canvas control client construction, bind the publisher: the console controller's `set_pending_approval` gets wrapped (original called first — TUI card behavior unchanged; then `approval_raised`/`approval_cleared(None-payload)`); run-status hooks the controller's terminal-run observation seam (the same place run-state transitions render — anchor by reading how chat_screen observes run completion; wrap identically). All publishes bounded + exception-swallowing (a relay failure must never break the TUI).

Tests: wrapper passes through to the original callback (order asserted); publishes the right frame types with a fake client recording events; None client = no-op; exceptions in publish swallowed; TUI path proven unaltered (original invoked even when publish raises).

Commit: `feat(web-server): served-child approval/run relay publisher (ADR-236)`.

### Task 3: Parent-side relay in MobileChatService

**Files:**
- Modify: `tldw_chatbook/Web_Server/mobile_service.py` (broker handle + event subscription + merged approvals + decision routing), `tldw_chatbook/Web_Server/serve.py` (pass the broker into the service when canvas enabled; children registry access — read `bind_served_browser`/`_canvas_control_broker` usage)
- Test: `Tests/Web_Server/test_mobile_service.py` (extend with an in-process broker + scripted child)

Interfaces (service additions):
```python
def attach_relay(self, broker, *, child_ids: Callable[[], Iterable[str]]) -> None:
    # start an event-drainer task per broker (canvas.events.request-style read loop
    # extended to relay kinds); desktop rounds land in self._desktop_approvals
    # {round_id: {**payload, child_id}}; cleared/resolution removes; run.status ->
    # MobileEvent(type="run_status", session_id, payload={"status","surface":"desktop"})
```
`list_approvals()` returns mobile + desktop rounds (each entry gains `surface`; desktop adds `child_id`); `decide_approval(round_id, decisions)` routes desktop rounds via `broker.request(child_id, "relay.decide.request", {"round_id":…, "decisions":…})` with a short timeout (2 s; timeout/dead child → drop round, return False); child-disconnect cleanup fails rounds closed. `shutdown()` stops the drainer.

Tests (real broker + scripted child client on an ephemeral port, reusing the protocol test fixtures): raised round appears in list_approvals with surface desktop + fires approval_requested event on subscribers; decision routes to the child (scripted handler resolves → response True → round cleared); dead-child decide → False + round dropped; run.status → event; disconnect → rounds cleared; mobile rounds untouched throughout.

Commit: `feat(web-server): parent-side desktop approval relay in the mobile service (ADR-236)`.

### Task 4: Phone UI + docs + close-out

**Files:**
- Modify: `Web_Server/static/mobile/app.js` (approvals inbox renders `surface` badge; desktop run_status events trigger the existing session resync for the open chat), `Web_Server/README.md` (relay paragraph: what works, canvas-disabled caveat, fail-closed semantics), backlog task file (close-out: ACs, notes, Done)
- Test: `Tests/Web_Server/test_mobile_api.py` (surface field passthrough) + `test_mobile_page.py` (JS contains the surface marker)

Steps: failing tests → UI (a "desktop"/"phone" tag on approval cards; run_status → syncOpenSession) → matrix green + baselines → README + close-out → commit `feat(web-server): phone surface for desktop approvals + relay docs (ADR-236)`.

---

## Self-Review

**Spec coverage:** §4.3 v1-limit removal (desktop approvals + live cross-surface completion events) → Tasks 1-4; mobile-initiated path untouched; fail-closed on disconnect documented. Out of scope: full live streaming mirroring of desktop transcripts (still completion-level via run_status → resync — the spec's Phase 2 named "desktop approvals and live cross-surface updates"; run_status + existing resync delivers update-level sync; note this in README).
**Placeholder scan:** the two anchored lookups (chat_screen's approval binding site; the controller's run-observation seam; children-registry accessor) are named files+seams for at-implementation reading — the wrapper contract (call-through-first, bounded publish) is fully specified.
**Type consistency:** relay frame field names identical in Task 1 registry, Task 2 publisher, Task 3 service; `surface` literal ("desktop"/"mobile") consistent across service/API/JS; round_id threading uniform.
