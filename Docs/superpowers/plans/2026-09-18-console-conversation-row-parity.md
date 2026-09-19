# Console Conversation Row Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Subagent-driven development is an alternative only when explicitly selected and available.

**Goal:** Implement TASK-32827: Conversations rows visually and behaviorally match workspace chat rows.

**Architecture:** Keep the flat list and workspace tree as separate owners. Share pure cell-width and semantic presentation helpers while retaining each native widget's scrolling, selection and keyed identity. Consume TASK-32826's right action icon and attention facts.

**Tech Stack:** Python >=3.12, Textual 8.x, Rich terminal-cell measurement, existing TCSS tokens, pytest.

**Spec:** [Approved design](../specs/2026-09-18-console-conversation-review-and-attention-design.md), sections 2–3 and 6.

ADR required: yes
ADR path: `backlog/decisions/171-console-conversation-review-and-attention.md`
Reason: implement the accepted shared row interaction contract, preserving ADR-083 ownership.

## Global Constraints

- The **Inspector sidebar is explicitly out of scope**: do not redesign its sections, layout, navigation, or behavior.
- Run targeted checks only; a full suite requires a separate user request.
- TASK-32826 must provide representative action icons, semantic attention data and the combined menu before this task integrates them.
- Conversations stays flat; Workspaces stays a tree. Do not change sorting, ownership, search/paging scope, subagent hierarchy or Ctrl+K membership.
- No new dependencies or generalized list framework. Respect ADR-031 and `backlog/docs/design-language.md`.
- Never edit generated `tldw_cli_modular.tcss`; rebuild with `python tldw_chatbook/css/build_css.py`.
- Read TASK-32827, applicable AGENTS and lessons-testing-evidence, lessons-live-verification, lessons-console-wiring and lessons-textual. Establish isolated execution state before editing.
- At execution start: `backlog task edit 32827 -a @codex -s "In Progress" --plan "Execute Docs/superpowers/plans/2026-09-18-console-conversation-row-parity.md after TASK-32826 under ADR-171."`

## File responsibilities and inherited interfaces

- Modify `tldw_chatbook/Widgets/Console/console_workspace_context.py`: flat-row composition, title width and focus text.
- Modify `tldw_chatbook/Widgets/Console/console_workspace_tree.py`: matching title/action alignment and state text, preserving native node identity.
- Create `tldw_chatbook/Widgets/Console/conversation_row_presentation.py`: only shared terminal-cell layout helpers.
- Modify owning `BUNDLED_CSS` strings and, where existing selectors live there, `tldw_chatbook/css/features/_chat.tcss`; add any missing semantic token to `tldw_chatbook/css/core/_variables.tcss` first.
- Consume `present_conversation_attention(facts, *, custom_icon="", ascii_mode=False) -> ConversationAttentionPresentation` and the semantic/appearance fields from TASK-32826. Do not create another priority table.
- Test `Tests/UI/test_console_conversation_row_wrap.py`, `Tests/UI/test_console_workspace_context_rail.py`, `Tests/UI/test_console_workspace_tree.py`, `Tests/UI/test_console_workspace_tree_performance.py`, `Tests/Workspaces/test_conversation_browser_subagents.py`.

## Task 1: Compact title/action layout with truthful state text

**New helper contract:**
`conversation_title_cells(title: str, available_cells: int) -> str` returns a
single physical line including ellipsis if needed, fitting max(0, available_cells).
`conversation_action_width(*, ascii_mode: bool) -> int` returns a stable slot
width for the supported presentation vocabulary (9 ASCII cells; measured Unicode
width including valid custom-icon max width). Use existing appearance validation
limits rather than permitting arbitrary multi-line custom glyphs.

- [ ] Replace only flat-row wrapping expectations with compact-row requirements; retain the old wrap helper's tests if unrelated consumers still use it. Add the pure test:

```python
from rich.cells import cell_len

def test_compact_title_respects_narrow_wide_character_budget():
    title = conversation_title_cells("日本語の長い会話タイトル", 7)
    assert "\n" not in title
    assert cell_len(title) <= 7
    assert title.endswith("…")
    assert conversation_title_cells("Chat", 0) == ""
```

- [ ] Run `pytest Tests/UI/test_console_conversation_row_wrap.py -q` to confirm the new behavior is absent.
- [ ] Implement cell-aware cropping with Rich Text, which avoids slicing a wide glyph incorrectly:

```python
def conversation_title_cells(title: str, available_cells: int) -> str:
    width = max(0, available_cells)
    text = Text(" ".join(str(title).splitlines()))
    text.truncate(width, overflow="ellipsis")
    return text.plain
```

Import Text from rich.text. Budget the action slot, favourite marker, and
spacing before this call. Remove the old minimum-ten-cell floor for these new
compact rows; an impossibly narrow title must not push the menu outside bounds.

- [ ] Compose a one-line resting title/action row. Keep Favourite as a property marker and selection as the existing token-backed row treatment. Remove routine repeated status/age subtitle text from resting rows; retain full title and metadata in focus/hover text. For actual pending decisions, queue/subagent progress or failures, add concise truthful activity text without discarding children or progress actions. No added left appearance control or duplicated run-marker prefix.
- [ ] Keep the action zone's right edge stable through glyph changes, window resize and scrollbar appearance. Size rest/hover/focus/disabled using existing token classes. Reuse the tree's row treatment as the visual reference, not a new web-style card.
- [ ] Add mounted assertions using `make_console_pilot(size=(120, 40), production_styles=True)`: resting row height is one, opener is visible, title and action regions do not overlap, and a status update changes its glyph without changing the action rectangle. Repeat Unicode/ASCII at narrow width. Run the named row suites. Commit: `feat: align Console conversation rows with workspace density`.

## Task 2: Navigation, progressive activity and stable identity

**Interfaces:** existing row IDs, tree node keys, `WorkspaceTreeMenuRequested`,
and conversation menu events remain the public routing boundary. Keyboard `m`
opens the menu only when the row owns focus; composer/search text entry retains
that character.

- [ ] Add mounted pointer/keyboard tests to both row suites. Select a conversation by keyboard, open its menu with `m`, press Escape and assert focus returns to that exact row. Verify `m` in the composer/search inserts text and does not open a row menu.
- [ ] Bind flat-row menu dispatch to captured stable row_key/conversation/profile, using TASK-32826's same action route. Preserve pointer-down identity checks and reject release after the target moved/disappeared. Do not rebuild the whole tree to update an icon or rename.
- [ ] Preserve active subagent/progress rows and existing explicit drill-in actions. Use semantic attention for collapsed workspace and capped flat-list summaries. A hidden failure still surfaces at the owning group/header; a visible row does not unnecessarily duplicate its marker on the group header.
- [ ] Extend paging/search tests with long Unicode titles and simultaneously arriving status updates. Assert selected ID, scroll position and node identity remain unchanged. Run `pytest Tests/UI/test_console_workspace_tree_performance.py Tests/Workspaces/test_conversation_browser_subagents.py -q`; retain existing performance budgets.
- [ ] Test at 80x24, 120x40 and wide dimensions with actual production CSS, including focused and hovered states. Verify tooltip/status text does not parse user titles as markup. Update `Docs/User_Guide/console/sessions-tabs-workspaces.md` for compact rows and keyboard menu access.
- [ ] Rebuild CSS and run `pytest Tests/UI/test_design_token_governance.py Tests/UI/test_css_bundle_sync_guard.py -q`, plus touched-file lint/format. Compare before/after real row surfaces in supported terminals and ASCII mode. Do not claim Windows rendering was verified without access.
- [ ] Review the diff for Inspector-sidebar changes and unrelated layout churn. Record ADR-171, test evidence and any missing platform evidence in TASK-32827. Mark Done only after every AC and repository DoD item is satisfied; commit explicit changed paths with `feat: preserve conversation row navigation and attention visibility`.

## Coverage and handoff

Task 1 implements density, width, selection/focus, appearance and meaningful
activity. Task 2 implements keyboard/pointer parity, keyed stability, overflow,
subagent preservation, documentation and visual evidence. Keep TASK-32826's
unread/receipt rules unchanged. No implementation has been performed by writing
this plan.
