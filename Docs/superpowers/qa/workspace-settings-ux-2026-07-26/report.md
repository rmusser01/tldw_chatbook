# Workspace Settings UX — UAT Baseline + Senior UX/HCI Review (2026-07-26)

Scope: the full cross-screen workspace journey — Console rail (Session/Details trays, switcher modal), Settings (Overview read-only rows, Storage `workspaces_db_path`), and the Library workspace surfaces the journey depends on.

Method: live UAT against `origin/dev` @ `73def5e8c` in an isolated worktree, fresh scratch profile (`TLDW_CONFIG_PATH`, `users_name = "uat_ws"`), tmux driver per `.claude/skills/verify/SKILL.md` (SGR click injection, capture-pane observation), live llama.cpp server at `:9099`. Act 1 = fresh first-run; Act 2 = seeded (3 workspaces, 6 conversations, 1 ghost membership). All captures in `captures/` (`cap-NN-*.txt`); every finding cites a capture and/or `file:line`. Findings observed only in code (not live-driven) are labeled **code-verified**.

The first structural observation: **there is no feature called "Workspace settings."** The concept is split across three screens with different owners — context switching in Console, management in Library (buried in a disclosure), status in Settings (read-only by documented decision, `Docs/superpowers/plans/2026-05-29-settings-configuration-hub.md` Stage 5). No surface explains what a workspace is or owns the whole concept.

---

## Part 1 — UAT baseline narrative

### Act 1: fresh first-run

| # | Scenario | Outcome | Evidence |
|---|----------|---------|----------|
| 1 | Cold discovery | **Workspace UI not reachable on first launch**: Console shows the "Get started" provider-setup card with the composer locked; the left rail (all workspace controls) is not visible until a provider is configured. Nothing anywhere explains what a workspace is. | cap-01 |
| 2 | Provider setup detour | Setup card routes to Settings → Providers & Models. Provider select's local providers only appear after scrolling a 20+-item dropdown. Test → Save worked well (llama.cpp @ :9099, "endpoint reachable (1 model)"). | cap-02 |
| 3 | Rail first sight | Session section shows `Workspace  Default`, a **Switch** button, a **RAG Scope** button, and static copy "Add another workspace before switching." — but **no visible New button** (see Finding C1). Scope row renders blank. | cap-03 |
| 4 | Empty browser | Starred/Workspaces empty copy fine ("No starred conversations." / "No workspace conversations."). | cap-03 |
| 5 | Settings seam | No Workspace category (19 categories listed). Overview shows read-only rows; recovery copy is circular (Finding m2). | cap-02, cap-11 |
| 6 | Create first workspace | Only achievable by clicking an **invisible button** (Finding C1). Creation is **silent** (Finding M1): no name prompt, no toast; "Workspace 1" appears + activates + new tab "Workspace 1 Chat". | cap-04, cap-05 |
| 7 | Switcher modal | Works: header "Change Workspace", explainer "Switching changes Console context only; Library and Notes stay globally visible.", current row inert, Escape dismisses. | cap-06, cap-07 |
| 8 | First chat + Scope row | Live send OK. Scope row shows **"This conversation"** — humanized. **Hypothesis "raw UUID in Scope row" is KILLED** (dev improved it). Default-workspace chat files under **Chats**, not Workspaces (Finding m3). | cap-08 |

### Act 2: seeded (Default + Workspace 1 "Client A" + Workspace 2 "Client B" + ghost row)

| # | Scenario | Outcome | Evidence |
|---|----------|---------|----------|
| 9 | Grouped browser | Good: active workspace group expanded, others collapsed; search summary line; per-row star + age. | cap-12, cap-13 |
| 10 | Cross-workspace row click | **Silently switches the active workspace** (Workspace 1 → 2). No toast/confirmation; the Workspace status row was scrolled out of view when it changed. | cap-14, cap-15 |
| 11 | Search | Works: "1 match", force-expands collapsed groups, Clear restores. | cap-16 |
| 12 | Starring | Works: ★ row appears under Starred with "Workspace 2 - active" annotation. | cap-17 |
| 13 | Details handoff rows | Populated: "Client B kickoff - reference", "Client B scope - reference". But the tray has label defects + jargon (Finding M4). | cap-18, cap-19 |
| 14 | Library management | Reached only via rail → **Details** disclosure (collapsed) → scroll to bottom. "Create local workspace" worked (DB-confirmed) but: no visible confirmation, the rail **recomposed to top with Details re-collapsed**, and it **silently switched Console's active workspace to Workspace 3** (DB `active=1`; Console confirmed on return). | cap-20 – cap-25 |
| 15 | Blocked staging | "Handoff · 0 eligible, ● 1 blocked" renders; clicking disabled "Use in Console" gives **zero visible feedback** (explanation is tooltip-only — a hover affordance many terminal users never see). | cap-21, cap-22 |
| 16 | Import sources | **Code-verified**: button renders only when there are zero source rows (`library_screen.py:3437-3444`); absent in the seeded state (cap-21). |  |
| 17 | workspaces_db_path | Storage category shows the field with "Restart required" guidance. Nuance: the input displays the default template path (`~/.local/share/tldw_cli/tldw_chatbook_workspaces.db`) while the resolved path below includes the profile dir (`.../uat_ws/...`) — two different truths in adjacent rows. Path-move/restart cycle not driven (code-verified restart gate, ADR 004). | cap-27 |
| 18 | Rail-state persistence | **My Details toggle on Workspace 3 was lost** after switching away and back. Persisted keys are per *workspace + conversation* pairs; the config held keys `workspace-local-1:7a6275af…` and `workspace-local-2:7a6275af…` — the same conversation under two workspaces, and no key at all for the toggle I made. Layout preferences effectively have amnesia. | scenario log + scratch config dump |
| 19 | Ghost membership row | "Ghost chat" (role `source`) renders exactly like an openable conversation; clicking it does **nothing** — no toast, no navigation, verified twice with fast capture. The mapped toast ("Open this saved conversation from Library…") never fired for this row type. | cap-26 |
| 20 | Live separation proof | **PASS at the data layer**: "Say only: alpha" → `workspace-local-3`, "Say only: beta" → `workspace-default`; browser groups matched; no bleed. | cap-28, cap-29 + DB query |

---

## Part 2 — Expert review findings

Severity: **Critical** = a core journey is blocked or invisibly mutates state; **Major** = journey completes but with high error-proneness or comprehension failure; **Minor** = friction/polish.

### C1 (Critical) — The Console "New workspace" button is invisible but clickable
The Session action row gets `margin: 0 0 0 12` (`css/components/_agentic_terminal.tcss:3150-3154`, mirrored in `tldw_cli_modular.tcss`), leaving ~25 columns for two buttons whose effective width is 16 each. "Switch" fits; **"New" (`#console-new-workspace`, `console_workspace_context.py:743-749`) starts at the clip edge — its label renders entirely outside the rail.** A blank ~5-column strip remains clickable: sweep-clicking it created "Workspace 1" (cap-04 → cap-05). The adjacent copy "Add another workspace before switching." tells the user to do something the UI gives them no visible way to do. Bitter irony: the comment at `console_workspace_context.py:751-766` (task-14) documents this exact overflow failure mode and moved RAG Scope to its own row because of it — the margin re-broke the original pair.
Also note the accident vector: an invisible state-mutating click target means stray clicks near the rail edge **silently create and activate workspaces** (that happened during this UAT — one unintended workspace).

### M1 (Major) — Workspace creation is silent, unnamed, and self-activating
`on_console_new_workspace` (`chat_screen.py:1639-1680`) has no success notification — only error paths notify. No name prompt; names are forced "Workspace N" (`registry_service.py::next_local_workspace_identity`). Creation immediately activates the workspace and opens a tab. From Library the same pattern plus a cross-screen side effect: "Create local workspace" changed Console's active context while the user was on the Library screen (cap-23 – cap-25); no visible confirmation was captured there either, and the rail recompose (below) hides the updated "Active" row.

### M2 (Major) — Active-workspace changes happen silently from three different triggers
(a) clicking a conversation row in another workspace's group (cap-14/15), (b) creating a workspace from Console, (c) creating one from Library. None produce a confirmation, and in (a) and (c) the `Workspace` status row is typically out of view when it changes. The switcher modal is the only path that makes the change explicit. Consequence observed live: active workspace = Workspace 3 while the focused tab held a Workspace 2 conversation whose Scope row says just "This conversation" (cap-25) — the user has no cue that their context and their view have diverged.

### M3 (Major) — No rename, archive, or delete anywhere
With forced "Workspace N" names, silent creation, and no lifecycle controls, real usage accumulates indistinguishable workspaces (this UAT produced 4, one accidental, in ~10 minutes). Nothing in Console, Library, or Settings can rename or remove one. `#library-workspace-create-local-copy` is now a dim static "Server sync WIP · local only" (`library_screen.py` — id suggests a retired button).

### M4 (Major) — The Details tray misrenders and overwhelms
Live captures (cap-10, cap-18, cap-19) show, in a 37-column tray:
- "Server      Not configured" followed by an orphaned lowercase "handoff" line — the "Server handoff" label wraps mid-phrase and reads as a stray word.
- **Two different rows labeled "Handoff"** (the package list and the ACP status row).
- Truncated values: "File tools  Off in Default work…", "Handoff  ACP handoff: Not co…".
- First-run jargon with no anchor: "handoff package", "ACP task/run package handoff", "Audit: no ACP package was sent." — none of these are actionable anywhere in the UI today (server/sync/runtime/ACP states are unreachable: no production writer for `save_runtime_binding`, server adapter state, or `workspace_acp_handoff_state` — code-verified). The tray is mostly aspirational copy presented as live status.

### M5 (Major) — Library is the designated "management home" but is buried and loses your place
The Console redirects users to Library, yet the workspace surface there is: rail → collapsed **Details** disclosure → scroll past Status to the bottom (cap-20/21). After "Create local workspace", the rail recomposes: scroll returns to top, the disclosure re-collapses, and the "Active" row that would confirm the action is hidden again (cap-24). Disabled "Use in Console" explains itself only via tooltip (cap-22). "Import sources" appears only when the source list is empty (`library_screen.py:3437-3444`), so the escape hatch exists exactly when users are least likely to be there.

### M6 (Major) — Non-resumable membership rows are dead on click
A membership row whose conversation can't be resumed (role `source` / missing record) renders identically to openable rows and gives zero feedback when clicked (cap-26, verified with 0.4s capture). The mapped recovery toast for missing conversations did not fire for this row type.

### M7 (Major) — Rail layout preferences are keyed per workspace+conversation and get lost
`[console.rail_state]` keys are `console_rail_state:<workspace>:<conversation>` — a new key per conversation, so section-open preferences reset with every new chat, and a toggle made moments earlier is gone after a switch round-trip (scenario 18). Observed keys also pair one conversation with two different workspaces. If per-workspace layout memory is the intent (a good one), the conversation component defeats it.

### m1 (Minor) — Settings label stutter
"Workspace default: Workspace: Workspace 3 (workspace-local-3); Authority: local-only; Sync: not-configured" and "Sync safety: Collections: Sync: dry-run only; Workspaces: Sync: dry-run only" (cap-11, cap-27) — label-in-value duplication makes rows read like debug output.

### m2 (Minor) — Settings recovery copy is circular
"Open the matching Settings category or destination to change behavior; sync and workspace status here is read-only." — never names which destination owns workspaces. A user who came to Settings looking for workspace management leaves with no pointer (cap-11). The code's own fallback copy elsewhere still says "Library > Workspaces", a mode that no longer exists (`display_state.py`, code-verified — the live path shows a disclosure under Library rail Details instead).

### m3 (Minor) — Default-workspace chats aren't "workspace conversations"
DB rows carry `workspace_id=workspace-default, scope_type=workspace`, yet the browser files them under "Chats" while other workspaces get named groups (cap-08, cap-29 + DB query). Defensible design (keep everyday chat unlabeled), but it contradicts the switcher's framing that Default is a workspace like the others; users who later adopt workspaces will look for their old chats in the wrong bucket.

### m4 (Minor) — Storage shows two different workspace-DB paths
Input value = default template path; resolved-path caption = actual per-profile path (cap-27). Adjacent contradictory truths in a restart-gated, data-loss-adjacent setting.

### m5 (Minor) — Library "Conversations (1)" vs 6 real conversations
The rail Browse count showed 1 while 6 non-deleted conversations existed (cap-21/24). Not fully traced; likely counts only non-workspace-scoped items. Whatever the rule, it disagrees with the Console browser and reads as data loss.

### m6 (Minor) — No keyboard path to any workspace action
`ChatScreen.BINDINGS` has nothing workspace-related; Switch/New are mouse-only compact buttons, and the switcher modal's only binding is Escape. For a TUI, the core context-switch operation deserves a binding and command-palette parity ("Console: Session settings…" exists; there is no "Console: Switch workspace…").

### Hypothesis audit (pre-seeded from code reading)
- Raw UUID in Scope row — **KILLED** (live shows "This conversation").
- Silent cross-workspace switch — **CONFIRMED** live.
- Invisible/clipped New button — not predicted by the map (predicted "New renders"); **found live, worse than hypothesized**.
- Dead-end "Copy or link…" staging copy — **CONFIRMED** as tooltip-only on a disabled button; no copy/link affordance exists.
- Details tray aspirational states (server/sync/runtime/ACP) — **CONFIRMED** code-verified unreachable.
- Staged-source send-block (S18) — **NOT TESTED** live (no UI path sets staged `workspace_id`; code-verified test-only).
- "No Workspace category in Settings" — **CONFIRMED** (cap-02 full category list).

---

## Part 3 — Upgrade proposals

Ordered by leverage; items 1–4 are near-mechanical fixes, 5–8 are design work.

1. **Fix the action row** (C1): drop/shrink the 12-col margin or stack Switch/New vertically like RAG Scope; add a geometry regression test asserting both buttons' regions fit the rail clip (pattern exists from task-14).
2. **Make workspace mutations loud** (M1/M2): success toast on create ("Created and switched to Workspace 2"), a status-line flash or toast on any active-workspace change not initiated in the switcher, and a subtle highlight pulse on the `Workspace` status row when its value changes.
3. **Repair the Details tray** (M4): unique labels (Package handoff / ACP handoff), no mid-label wraps at rail width, full-value tooltips for truncations — and consider hiding rows whose feature has no production writer behind a single "Server features: not configured" line until they're real.
4. **Fix dead rows** (M6): non-resumable rows get a distinct style + explanatory click toast, or aren't rendered as rows at all.
5. **Naming and lifecycle** (M3): name prompt on create (default "Workspace N", Enter to accept), rename via the switcher modal or Library panel, archive with guardrails. Without rename, every other workspace investment depreciates fast.
6. **Give management a real home** (M5): promote the Library workspace body from a bottom-of-disclosure appendix to a proper canvas panel (Library already has the canvas pattern), preserve disclosure/scroll state across recompose, and make "Use in Console" explain itself inline when disabled. Alternatively (bigger call, revisits the Stage-5 boundary decision): a Settings "Workspaces" category that hosts management while Console keeps context switching.
7. **Connect the tri-screen story** (m2): Settings workspace rows state where to act ("Manage in Library › Details › Workspace; switch in Console"); update the stale "Library > Workspaces" copy; Console's single-workspace recovery copy should point at its own New button once that button is visible.
8. **Keyboard + palette parity** (m6): a binding for the switcher, palette entries "Switch workspace…" / "New workspace", and per-workspace rail-state keys (drop the conversation component) so layout prefs survive (M7).

## Appendix — environment
Worktree `/private/tmp/tldw-ws-ux-review` @ origin/dev `73def5e8c`; scratch profile `uat_ws` (deleted after); llama.cpp `:9099` (gemma-4-26B Q4_K_M); tmux socket `wsuat`, 235×52. Seeding script embedded in `uat-script.md`.
