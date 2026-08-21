# Console interrupt host (sub-project C) — design

Date: 2026-08-20
Status: design approved in session; spec pending user review
Parent program: `Docs/superpowers/specs/2026-08-19-console-user-interaction-design.md` (§4: PR0 → **C** → A → B → D → E)
Prior art: PR0 merged as PR #1836 (`10509d286`, task-15661) — round-keyed retained payloads, FIFO head, shared park/head/unpark/remount helpers, plus Qodo riders (`deadline_monotonic` remaining-time snapshots; `_remount_head(None)` = resolve-active).

## 1. Scope decision (user-approved)

**Unify plumbing + card host, same UX.** One round-lifecycle spine and one card-host contract that the existing cards migrate onto, rendering byte-identically. The three recorded gaps (silent same-session queue, boolean approvals chip, card-rebuild selection loss) are fixed **only where the unification makes the fix free** — which turns out to be exactly one of them (§5). No cross-bridge FIFO, no countdown changes, no new affordances.

Two boundary decisions, also user-approved:

- **The spine is extracted to a new module**, `Chat/console_interrupt_rounds.py`. The controller is >12,400 lines and 600KB — large enough that a whole-file read killed a subagent during PR0 — and the lifecycle becomes testable without a controller.
- **The resume panel is out of scope.** It shares `ChatTaskCards` but not the lifecycle: no worker thread waits on it, nothing times it out, nothing revokes it. Forcing it into a round model distorts both. (Codex parity note: its five unified kinds are all request/response; resume has no analogue.)

## 2. What exists today (post-PR0, dev @ #1868 era)

Three bridges — `request_mcp_approvals`, `request_skill_install_confirm`, `request_skill_script_confirm` — each with:

- its own round registry (`_pending_approval_rounds`, `_pending_skill_install_rounds`, `_pending_skill_script_rounds`) under its own lock (`_approval_state_lock`, `_pending_skill_install_lock`, `_pending_skill_script_lock`);
- a shared payload layer (PR0): per-kind `_parked_*_payloads` maps, all under `_approval_state_lock`, driven by the five generic helpers (`_park_round_payload`, `_head_round_payload`, `_unpark_round_payload`, `_remount_head`, `_head_round_payload_locked`);
- its own copy of the arm → park/mount → `event.wait(1.0)` poll → resolve/timeout/revoke → teardown-re-derive lifecycle, inlined in the bridge method;
- its own screen setter (`set_pending_approval` / `set_pending_skill_install` / `set_pending_skill_script`) feeding `TaskResumeState` fields, which `ChatTaskCards.sync_state` fans out to per-kind cards.

Uniform contracts already in place that C relies on:

- all three `resolve_pending_*` fail closed on a `None` round/request id (TASK-913);
- the two payload-layer invariants: the mounted-card decision is computed inside the `call_from_thread` callable, and `_remount_head(session_id=None)` resolves the active session at callback time;
- `remount`-family re-derive call sites: `new_session`, `switch_session`, `close_session` (three calls each, one per bridge), plus the public `remount_pending_approval_for_active_session`.

## 3. C1 — the spine: `Chat/console_interrupt_rounds.py`

### 3.1 Shape: one lifecycle, one lock, per-kind storage

`InterruptRoundHost` owns:

- **One lock.** Today's three-lock split exists only to serve three inlined copies of the same lifecycle; the revoke path's documented "sequential, never nested" two-lock dance disappears with it. Verified safe before this design was written: no path nests a skill-registry lock with `_approval_state_lock` (the two candidate sites in the teardown `finally` blocks are dedented/sequential).
- **Per-kind registries and payload maps** — `dict[kind, dict[round_id, ...]]` internally, with the controller **aliasing the six legacy attribute names** (`_pending_approval_rounds`, `_parked_approval_payloads`, …) to the host's per-kind dict objects at construction, and the three legacy lock names to the host's single lock. This is a hard requirement, not a courtesy: eleven test files and **production** `ChatScreen._current_park_round_ids` read those attribute names directly. Alias staleness is not a risk: a repo-wide grep confirms none of the six dict names is ever *reassigned* outside `__init__` (production or tests — mutation only), so the aliased objects can never be silently swapped out from under the host. Per-kind storage with aliases means zero churn at every one of those sites; a merged kind-tagged map would have forced edits across all of them and destroyed the parity oracle (§6).
- The five PR0 helpers, moved verbatim (same invariants; the tests that pin them keep passing).
- A generic `run_round(kind, payload, *, session_id, legacy, run_id, wiring) -> Resolution` where `Resolution` is `decided(decision) | timeout | cancelled | revoked`, and the poll loop reproduces each bridge's exact wait conditions **via wiring, not hardcoding**:
  - approvals: session-cancel probe (round-scoped cancel event) + deadline;
  - skill bridges: cancel probe + visit-teardown event + shutdown flag + ADR-067 optional deadline (`timeout <= 0` arms none);
  - all kinds: `revoked` re-read after wake.
- `resolve(kind, round_id, decision) -> bool` — fail-closed on `None`/unknown ids (the TASK-913 contract, now in one place).
- `revoke_for_run(run_id)` — the sweep, per kind, with per-round unpark + remount.
- `remount_for_session(session_id)` — pushes every kind's head in one call; the four re-derive call sites collapse from three calls each to one.

**Construction-order trap:** the controller's `.app` handle is assigned at *screen attach* (`ChatScreen` wiring), long after `__init__` constructs the host — and several UI tests swap in controller doubles before re-running that wiring. The host must therefore take app access as a **late-bound callable** (e.g. `get_app: Callable[[], object | None]` reading `controller.app` at call time), never capture the handle at construction. Same for the setters, which are also attach-time assignments.

**Shutdown is a wait condition, not a sweep.** `begin_shutdown()` sets the never-reset `_shutdown_requested` Event and cancels the headless visit event; armed rounds observe both through their cancel probes and deny themselves. There is no registry sweep to move — `begin_shutdown`/`_cancel_headless_rounds` stay controller-side untouched, and the host sees shutdown only through the injected probe callables.

### 3.2 What stays in the controller

The three bridge methods keep their **exact names, signatures, and return types** — they are the tested public seam. Bodies shrink to: resolve timeout (the per-bridge config seams — `mcp_approval_timeout_seconds` etc. — are test-fixture surface and stay controller attributes), build payload, call `host.run_round`, map the `Resolution` to the bridge's return shape, plus the legs that are genuinely bridge-specific and stay local: MCP's detached-view announce (task-15860), inspector counts, cancelled-decision audit logging; the skill bridges' badge/park-toast callbacks remain injected controller callbacks either way.

Timeout defaults on resolution mapping: approvals stamp `"timeout"` per undecided name; install returns `False`; script returns `{"allow": False}` — byte-identical to today.

### 3.3 Locking hazard, stated up front

The alias makes `_pending_skill_install_lock is _approval_state_lock` — one **non-reentrant** lock behind three historical names. Nothing nests them today (verified by inspection of every acquisition site, tests included — the six test-side acquisitions are all single grabs of `_approval_state_lock`; no test touches a skill lock at all), but any future nesting becomes an immediate self-deadlock instead of a latent ordering hazard. This is the same property `_approval_state_lock` already has — it self-deadlocked a PR0 implementer who nested `_unpark_round_payload` inside a `with` block — so the plan carries a grep-check step and the host docstring carries the warning.

## 4. C2 — the card host

`ChatTaskCards.sync_state` today calls all four cards' setters unconditionally on **any** `TaskResumeState` change; `ChatApprovalCard.set_batch` does `remove_children()` and rebuilds every row, so an unrelated update discards a user's unsubmitted per-tool selections. C2:

- replaces the unconditional fan-out with a kind→card routing table (screen keeps the three setter names and `TaskResumeState` fields — same seam, same UX);
- adds a **round-identity guard in each card's setter**: skip the rebuild when the incoming payload's `round_id`/`request_id` matches the currently rendered one. **Not dict equality** — the PR #1836 Qodo rider makes `_head_round_payload` return a fresh snapshot with a different remaining-time `timeout_seconds` on every call, so consecutive pushes are never `==` for deadline-carrying payloads and an equality guard would simply never fire. Identity is the stable key.
- **the guard resets on hide.** Skip requires *same id AND the card is currently visible*; a `None`/empty push clears the remembered identity. Without this, the common switch-away-and-back sequence — hide (`None` push), then re-mount of the SAME round — would match on id, skip the rebuild, and leave the card `display=False` forever. The remembered id lives and dies with the visible render.
- the approval card, on an identity match, still refreshes its `#approval-deadline` `Static` from the incoming snapshot's `timeout_seconds` — countdown stays current *and* selections survive. The skill cards render no countdown; their guard is a plain skip.

The guard lives in the widget layer deliberately: the PR0 final review's prohibition was against identity logic in `_remount_head._apply` (that is what made the deleted guards order-dependent); the card setter is the sanctioned home and protects against every caller, not just `sync_state`.

This is the **one** known-gap fix in C's scope. Silent same-session queue and the boolean approvals chip stay recorded in the parent spec for later sub-projects.

## 5. The `"question"` kind

Reserved only: enum value, host support for a fourth kind's wiring, nothing else. The tool, card, `TaskResumeState.pending_question` field, and setter are sub-project A's, riding this host as its first new renderer.

## 6. Testing

- **Parity oracle:** the existing battery across the interrupt suites must pass **unchanged** — any C1 edit to an existing test invalidates the byte-parity claim and is a review flag. Baseline at PR0 merge time was 4 known pre-existing dev failures (2× `test_skill_install_concurrent_confirms` shutdown-flag, 2× `test_console_parallel_runs` navigation); re-verify the baseline at the current dev tip before starting, in a detached worktree, exactly as PR0 did.
- **Shutdown-flag caveat:** the two red shutdown tests exercise the poll loop C1 rewrites. Their pre-existing redness could mask a C1 regression in shutdown handling. The plan must include a characterization step — establish *why* they fail at the current tip before migration — so post-C1 behavior is comparable. If C1 turns them green, report it; do not silently absorb it.
- **New:** host unit tests with no controller — arm/queue/promote/timeout/revoke/legacy/resolve-fail-closed per kind, plus a wiring test proving per-kind wait conditions are honored (visit-teardown wakes a skill round but not an approval round, etc.). This isolation is the testability the extraction buys.

## 7. Phasing

Two PRs:

- **C1** — the spine: module + host, migrate all three bridges, aliases, re-derive consolidation. Pure plumbing; UX byte-identical.
- **C2** — the card host: routing table + per-card identity guards + approval-card deadline refresh. One deliberate behavior fix (selection preservation), everything else byte-identical.

Per-bridge incremental migration was considered and rejected: three transition states where two round systems coexist is where drift bugs live, and PR0 demonstrated same-shape×3 in one reviewed pass.

## 8. Out of scope

No UX changes beyond §4's selection fix; no countdown machinery (A.2); no cross-bridge FIFO; resume untouched; no question tool/card; no chip changes; no silent-queue indicator.

## 9. Review deltas already incorporated

- The C2 guard was originally specified as payload equality in `sync_state`; review found the PR #1836 remaining-time snapshot defeats equality, and moved the guard to identity in the card setters (§4).
- The spine was originally "one registry"; review found 11 test files + production `chat_screen.py` reading the per-kind names, and corrected to per-kind storage with attribute aliases (§3.1).
- The single-lock collapse was verified against every acquisition site before being kept (§3.3).
- Anchors in this spec are symbolic; `chat_screen.py` shifted ~500 lines in one day during the PR0 cycle.
- Second review pass added: the identity guard's reset-on-hide rule (§4 — without it a same-round remount after switch-away-and-back would strand a hidden card), the late-bound app/setter injection (§3.1 — `.app` is attach-time, not construction-time), the shutdown-is-a-wait-condition clarification (§3.1), and the alias-staleness / test-side lock-nesting verifications. The parent program spec's §4 table was amended in the same commit: it had still listed resume among C's deliverables.
