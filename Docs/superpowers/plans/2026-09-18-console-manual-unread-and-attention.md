# Console Manual Unread and Attention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Subagent-driven development is an alternative only when explicitly selected and available.

**Goal:** Implement TASK-32826: durable manual unread and one representative right-hand conversation action icon.

**Architecture:** ConversationLocalMarksService owns manual marks and conditional clearing. A pure Workspaces-layer projection combines semantic facts supplied by existing runtime/receipt owners; existing row widgets and the action menu consume it. Explicit navigation routes acknowledge manual unread after the exact conversation paints.

**Tech Stack:** Python >=3.12, Textual 8.x, SQLite, pytest, existing Rich cell measurement.

**Spec:** [Approved design](../specs/2026-09-18-console-conversation-review-and-attention-design.md), sections 2–6.

ADR required: yes
ADR path: `backlog/decisions/171-console-conversation-review-and-attention.md`
Reason: reuse the accepted manual-read and attention contract; no additional ADR or schema is intended.

## Global Constraints

- The **Inspector sidebar is explicitly out of scope**: do not redesign its sections, layout, navigation, or behavior.
- Run targeted checks only; a full suite requires a separate user request.
- Manual marks never change operational receipt acknowledgement, sync, workspace membership, conversation status or custom appearance storage.
- No new dependencies, global glyph replacements, or polling timers.
- Apply ADR-031 key rules and ADR-150 tokens; rebuild generated CSS with `python tldw_chatbook/css/build_css.py` when source styles change.
- Preserve current worktree changes. Establish an isolated implementation checkout at execution time; reconcile the reviewed source state deliberately rather than silently dropping existing uncommitted controller changes.
- Before code, read TASK-32826, relevant local AGENTS files, `backlog/docs/design-language.md`, and lessons-testing-evidence, lessons-live-verification, lessons-console-wiring and lessons-textual.
- At execution start: `backlog task edit 32826 -a @codex -s "In Progress" --plan "Execute Docs/superpowers/plans/2026-09-18-console-manual-unread-and-attention.md under ADR-171; targeted checks and live row verification only."`

## File responsibilities

| File | Responsibility |
| --- | --- |
| `tldw_chatbook/Chat/conversation_local_marks_service.py` | Manual mark persistence, callback tokens and batch reads |
| `tldw_chatbook/Workspaces/conversation_attention.py` (new) | Semantic facts, priority and content-free presentation |
| `tldw_chatbook/Chat/console_chat_controller.py` | Narrow public read of live attention facts, preserving existing run-marker API |
| `tldw_chatbook/UI/Console_Modules/workspace.py` | Batch enrichment and menu mutation workers |
| `tldw_chatbook/UI/Console_Modules/session.py` | Explicit activation intent and post-paint manual read acknowledgement |
| `tldw_chatbook/Workspaces/conversation_browser_state.py`, `workspace_tree_state.py` | Carry appearance/attention through keyed projections and hidden-row summaries |
| `tldw_chatbook/Widgets/Console/console_workspace_context.py`, `console_workspace_tree.py` | One right action control; no row-density redesign in this task |
| `tldw_chatbook/Chat/console_conversation_actions.py`, `tldw_chatbook/Widgets/Console/console_conversation_action_menu.py` | Menu model, state text, appearance/read actions and geometry |
| `tldw_chatbook/UI/Screens/chat_screen.py` | Existing menu/opening route adapters and anchor placement only |
| `tldw_chatbook/Chat/console_switcher_state.py`, `tldw_chatbook/Widgets/Console/console_session_switcher_modal.py` | Add manual-unread annotation to existing results without changing membership/search semantics |
| `tldw_chatbook/Widgets/glyph_fallback.py` | Add mappings without changing existing consumers' glyph meanings |

## Task 1: Durable marks with an atomic stale-callback guard

**Tests:** extend `Tests/Chat/test_conversation_local_marks_service.py`.

**Interfaces to add in the marks service:**

```python
@dataclass(frozen=True, slots=True)
class ManualUnreadToken:
    service_epoch: str
    conversation_id: str
    generation: int
```

Methods: `mark_unread(conversation_id: str) -> ManualUnreadToken`,
`unread_token(conversation_id: str) -> ManualUnreadToken | None`,
`mark_read(conversation_id: str, *, expected: ManualUnreadToken | None = None) -> bool`,
and `unread_ids_for(conversation_ids: Sequence[str]) -> frozenset[str]`.
`expected=None` means explicit unconditional Mark as read; automatic acknowledgement
must carry a non-null captured token. Database exceptions propagate to UI workers.

- [ ] Add this regression with the existing real-SQLite `_db` helper:

```python
def test_reapplied_unread_survives_stale_visit_with_same_clock(tmp_path, monkeypatch):
    service = ConversationLocalMarksService(_db(tmp_path))
    monkeypatch.setattr(service, "_now", lambda: "2026-09-18T12:00:00Z")
    old = service.mark_unread("conv-a")
    newer = service.mark_unread("conv-a")
    assert old != newer
    assert service.mark_read("conv-a", expected=old) is False
    assert service.unread_ids_for(["conv-a"]) == frozenset({"conv-a"})
    assert service.mark_read("conv-a", expected=newer) is True
```

- [ ] Run `pytest Tests/Chat/test_conversation_local_marks_service.py -q`; confirm the new regression fails because the new service behavior is absent.
- [ ] Add `MANUAL_UNREAD = "manual_unread"` to accepted mark types. Use one service-local reentrant operation lock, an opaque instance epoch, a monotonic counter, and current generations for marked IDs. On a successful mark/re-mark, advance the counter even when `_now()` repeats. A loaded durable mark receives a current process generation when first read. Clear removes that ID's current generation; a re-created mark gets a new counter value. Failed transactions do not publish a new generation or invalidate caches as if committed.
- [ ] Implement comparison and deletion inside the same operation lock and database transaction. Retain the existing parameterized SQL:

```sql
DELETE FROM conversation_local_marks
 WHERE conversation_id = ? AND mark_type = ?
```

Check service epoch, conversation ID and generation before this statement when
`expected` is supplied. All generic `set_mark`/`clear_mark` calls for MANUAL_UNREAD
must route through these operations so no public writer bypasses the guard;
use existing cursor helpers internally to avoid recursive routing.

- [ ] Implement batch reads with bounded parameterized ID chunks, deduplication and one transaction. Do not use the default-limited `list_marked_conversation_ids`. Add tests for 150 marked chats, an empty ID list, unrelated profile/service tokens, persistence through a fresh service, SQL failure, repeated clear, deleted/orphan IDs, and no changes to starred/fleet/terminal marks or sync metadata. Test the guard with a threading barrier to cover check/delete interleaving, not just sequential calls.
- [ ] Run the targeted marks suite and touched-file lint. Commit only this service and its tests: `feat: persist manual unread with guarded acknowledgement`.

## Task 2: Semantic attention and both row projections

**Create:** `tldw_chatbook/Workspaces/conversation_attention.py`, `Tests/Workspaces/test_conversation_attention.py`.
**Extend tests:** `Tests/Workspaces/test_console_conversation_browser_state.py`, `Tests/Workspaces/test_workspace_tree_state.py`, `Tests/Chat/test_console_run_state_per_session.py`.

**Interfaces:** `AttentionKind` is a Literal of `approval`, `blocked`, `failed`,
`running`, `paused`, `stopped`, `unread`, `ready`, `outcome_unknown`.
`ConversationAttentionFact(kind: AttentionKind, label: str)` is frozen.
`ConversationAttentionPresentation(icon: str, label: str, summary: str, css_class: str)` is frozen.
`present_conversation_attention(facts: Sequence[ConversationAttentionFact], *, custom_icon: str = "", ascii_mode: bool = False) -> ConversationAttentionPresentation` resolves the approved priority and glyph table.

- [ ] Add a concrete mixed-state test:

```python
def test_approval_overrides_unread_without_losing_its_explanation():
    facts = (
        ConversationAttentionFact("unread", "Unread"),
        ConversationAttentionFact("approval", "Approval required"),
    )
    view = present_conversation_attention(facts, custom_icon="💡")
    assert view.icon == "✋"
    assert view.label == "Approval required"
    assert "Unread" in view.summary
    assert present_conversation_attention((), custom_icon="💡").icon == "💡"
```

- [ ] Run `pytest Tests/Workspaces/test_conversation_attention.py -q`; confirm failure, then implement the pure vocabulary and sorting. Resolve known states via an explicit map copied from the spec; keep outcome_unknown distinct from ready. Map css_class to existing token-backed status classes. Arbitrary raw exception text is not an attention label.
- [ ] Add `conversation_attention_for(session_id: str) -> tuple[ConversationAttentionFact, ...]` to the existing chat controller. Derive approval/running/unvisited facts from the same data as `run_marker_for`, without collapsing priority there or changing that API. Combine these in the workspace controller with cached `ConsoleActivityReceiptService.unseen_snapshot()`, existing terminal attention evidence and coarse fleet fallback. No DB read occurs in row painting. A fallback with no known outcome yields outcome_unknown, not ready; paused/blocked appear only with actual authoritative evidence.
- [ ] Add default-empty semantic fields to browser input/output and tree conversation projections; carry the saved `icon`/`color` into tree rows, which currently drop them. Batch manual unread enrichment off-loop and validate profile/query/projection generation before committing it. Keep the existing legacy run-marker field for unrelated consumers. Resolve collapsed/capped attention from semantic facts without making Workspaces import Chat models; retain existing aggregation coverage bounds.
- [ ] Add cached/manual unread annotation to already returned Ctrl+K results. Do not alter Active membership, receipt acknowledgement or expand search into a new unread corpus.
- [ ] Test every state in Unicode/ASCII, unknown background outcome, duplicate facts, deterministic priority, hidden/capped rows, custom appearance restoration, unavailable marks and late old-profile reads. Run the three named projection suites plus affected controller tests. Commit: `feat: project representative conversation attention states`.

## Task 3: One right action control and a complete menu

**Tests:** `Tests/Chat/test_console_conversation_actions.py`, `Tests/UI/test_console_conversation_action_menu.py`, `Tests/UI/test_console_workspace_tree.py`.

**Interfaces:** add action IDs `mark_unread`, `mark_read`, `change_appearance`;
add target fields `manual_unread: bool | None = False`, `attention_summary: str = ""`,
`icon: str = ""`, `color: str = ""`. None means unknown/unavailable, not read.

- [ ] Add pure menu assertions using the existing `ConversationMenuTarget` and `build_conversation_menu`:

```python
def test_saved_read_chat_offers_unread_and_appearance():
    items = build_conversation_menu(ConversationMenuTarget(conversation_id="c"))
    ids = {item.action_id for item in items}
    assert {"mark_unread", "change_appearance"} <= ids
    assert "mark_read" not in ids
```

- [ ] Run the action-model suite to see the new failure. Then add the conditional read action beside Favourite, and Change icon and colour before More. Preserve unsaved-chat and unavailable-service explanations. Route appearance through `_open_console_conversation_appearance_picker`, preserving its existing validation and persistence; route mark mutations through captured-service off-loop workers and refresh on commit. Revalidate target existence/profile before writes; do not claim success on errors.
- [ ] In `_compose_conversation_browser_row`, remove the left appearance control and replace the right asterisk label using semantic presentation. Do the same in tree render/hit-zone code while retaining stable node identity and existing menu messages. Preserve original public row IDs where feasible. Keep glyph changes within this action presentation: do not replace global run-marker constants or Inspector sidebar glyphs.
- [ ] Add a status summary/title to the menu. Replace `ROOT_PAGE_HEIGHT`'s six-item assumption with actual measured/built content height in both chat-menu anchor routes in `chat_screen.py`; clamp to viewport with scrollable content when needed. Retain Escape, outside-click, submenu Back and focus restoration. All status variants still emit the same menu-opening event.
- [ ] Update mounted assertions from the retired `*` label to the actual semantic/default icon, not just removal of assertions. Test top/bottom viewport anchors, Unicode and nine-cell ASCII action slots, unavailable marks, unsaved chats, disabled reasons and appearance changes under an unread override. Rebuild CSS, run the named suites plus `Tests/UI/test_console_modal_dismissal.py`. Commit: `feat: unify conversation appearance and attention actions`.

## Task 4: Explicit revisit acknowledgement

**Tests:** new `Tests/UI/test_console_manual_unread.py`; extend `Tests/UI/test_console_session_controller.py` and `Tests/UI/test_console_conversation_archive_flow.py` where handoff assertions belong.

**Interfaces:** add a frozen `ConsoleManualReadVisit` in the session controller
module with `profile_key: str`, `marks: ConversationLocalMarksService`,
`conversation_id: str`, `session_id: str`, `token: ManualUnreadToken | None`,
and `navigation_generation: int`. Keep annotation-only imports under TYPE_CHECKING.
Add async `_prepare_manual_read_visit(self, session_id: str, *, conversation_id: str, navigation_generation: int) -> ConsoleManualReadVisit | None`
and async `_acknowledge_manual_read_visit(self, visit: ConsoleManualReadVisit) -> bool`.
The former captures a token off-loop for a deliberate target transition; the
latter verifies the actual painted target before scheduling `mark_read(expected=token)`.

- [ ] Add a mounted two-conversation regression in the new test module using `make_console_pilot` from `Tests/UI/test_console_left_rail.py`: open A, choose Mark as unread from A's real menu, verify durable A mark; open/close the Inspector modal and verify it remains; activate B then A via actual tab controls, await target paint and verify A's mark is cleared. Use the real marks service and session activation; do not stub the mutation under test.
- [ ] Reproduce the failing behavior with `pytest Tests/UI/test_console_manual_unread.py -q` before wiring. Instrument the existing `_activate_native_console_session`, `_resume_console_workspace_conversation`, and explicit handoff completion paths with a captured visit, rather than adding read writes to generic UI refresh.
- [ ] Use this guarded call only after the UI identity/generation checks:

```python
if visit.token is not None:
    cleared = await asyncio.to_thread(
        visit.marks.mark_read,
        visit.conversation_id,
        expected=visit.token,
    )
```

Here `visit.marks` is the captured ConversationLocalMarksService and every field
listed above is carried by the visit record. Report storage failure without
blocking successful conversation opening. Re-enrich rows after a committed clear.

- [ ] Record explicit top-level destination departure separately from generic ScreenSuspended/ScreenResume; opening or closing any modal is not a departure. Switching duplicate tabs sharing one conversation is also not departure. Automatic restore and background wake do not create user visit intents.
- [ ] Cover row/tree/Ctrl+K/Alt-tab routes, already-current click, duplicate tabs, explicit top-level departure/return, automatic startup, failed/cancelled activation, profile replacement, delayed paint, same-clock re-mark and receipt independence. Use completion signals rather than arbitrary sleep as evidence. Run new tests and affected existing session/native flow suites. Commit: `feat: clear manual unread on deliberate conversation revisit`.

## Task 5: Focused qualification and closeout

- [ ] Update `Docs/User_Guide/console/sessions-tabs-workspaces.md` with unread lifecycle, action icon priority, ASCII mode, menu access and the distinction from operational acknowledgements.
- [ ] Run targeted service/projection/menu/session suites above, token/bundle checks (`Tests/UI/test_design_token_governance.py`, `Tests/UI/test_css_bundle_sync_guard.py`), and affected import budgets (`Tests/Performance/test_screen_preimport_payload_budget.py`, `Tests/Performance/test_boot_worker_census.py`). Do not increase budgets to accommodate a new eager dependency.
- [ ] Run the repository's existing formatter/linter for changed files only; preserve unrelated pre-existing formatting. Review generated CSS separately from source edits.
- [ ] Live-check both row surfaces at 80x24, 120x40 and wide dimensions using disposable profile data; inspect real pointer targets, modal-close preservation, status text, Unicode/ASCII glyphs and restart persistence. Follow the existing iTerm2/Windows Terminal evidence requirements; missing platform evidence blocks Done, not unrelated implementation progress.
- [ ] Self-review for no Inspector-sidebar changes and no global receipt/glyph authority changes. Update TASK-32826 notes with ADR-171, changed files, checks, live evidence and deviations. Mark only verified ACs; use `backlog task edit 32826 -s Done` only when the repo DoD is met. Commit the documentation/task record with explicit paths.

## Coverage and handoff

Spec manual lifecycle/races/batch marks: tasks 1 and 4. Representative states,
unknown outcomes, custom appearance and hidden-state visibility: task 2.
Menu/appearance/keyboard/geometry: task 3. Privacy of metadata, tokens, docs and
live evidence: task 5. TASK-32827 consumes these interfaces; TASK-32828 is
independent. No implementation has been performed by writing this plan.
