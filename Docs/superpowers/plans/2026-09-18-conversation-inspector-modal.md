# Conversation Inspector Modal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Subagent-driven development is an alternative only when explicitly selected and available.

**Goal:** Implement TASK-32828: usable Context and Usage & cost inspection in the existing Conversation Inspector modal, retaining Exchange history.

**Architecture:** Keep the current modal and capture/snapshot/export owners. Extract only pure section/turn presentation and bounded list/detail widgets to keep the already large modal manageable. Entry points pass frozen target identity; each view loads on explicit demand and all trace-bearing views share disclosure invalidation.

**Tech Stack:** Python >=3.12, Textual 8.x, existing usage/snapshot/trace services, pytest with consolidated production CSS.

**Spec:** [Approved design](../specs/2026-09-18-console-conversation-review-and-attention-design.md), sections 1 and 6 and the source-backed review corrections.

ADR required: yes
ADR path: `backlog/decisions/171-console-conversation-review-and-attention.md`
Reason: implement the accepted modal information architecture while preserving ADR-069/097 disclosure contracts.

## Global Constraints

- The **Inspector sidebar is explicitly out of scope**: do not redesign its sections, layout, navigation, or behavior.
- Run targeted checks only; a full suite requires a separate user request.
- Keep Context, Usage & cost and Exchange history in one modal. No Overview page, new app destination or sidebar redesign.
- No new provider calls, schema, pricing engine, dependencies or background polling.
- Preserve Safe/Full masks, live automatic-instruction preview restrictions, permitted historical capture inspection and ephemeral export guards.
- Use design tokens, truthful local bindings and the existing SafeModalDismissMixin. Rebuild generated CSS from sources.
- Read TASK-32828, applicable AGENTS, ADR-171/069/097/150, and lessons-testing-evidence, lessons-live-verification, lessons-console-wiring and lessons-textual. Use isolated execution state; current snapshot/controller files already have unrelated work to preserve.
- At execution start: `backlog task edit 32828 -a @codex -s "In Progress" --plan "Execute Docs/superpowers/plans/2026-09-18-conversation-inspector-modal.md under ADR-171; preserve capture and accounting boundaries and verify only the modal."`

## File responsibilities

| File | Responsibility |
| --- | --- |
| `tldw_chatbook/Widgets/Console/console_conversation_inspector.py` | Modal lifecycle, tabs, target/disclosure guards, load orchestration and existing export routes |
| `tldw_chatbook/Widgets/Console/console_inspector_presentation.py` (new) | Pure context-section and usage-row presentation/identity |
| `tldw_chatbook/Widgets/Console/console_inspector_detail_pane.py` (new) | Reusable modal-local list/detail layout, Back and retained focus/scroll |
| `tldw_chatbook/UI/Screens/chat_screen.py` | Modal entry-point factories and immutable target callbacks only |
| `tldw_chatbook/Widgets/Console/console_context_controls.py` | Existing context entry copy, only if its modal labels change |
| `tldw_chatbook/css/core/_variables.tcss` | Missing modal geometry tokens only, without changing existing token values |
| `Docs/User_Guide/console/context-and-rag.md` | Context/usage/modal navigation, freshness and accounting explanation |

Existing `console_exchange_export_dialog.py`, `trace_export_profile_ui.py`,
`Chat/console_cost_tracker.py` and snapshot/ledger builders remain authoritative.
Do not rewrite them for visual convenience. Generated widget CSS is rebuilt,
never hand-edited. This plan is independent of TASK-32826/32827.

## Task 1: Explicit lazy view entry and target authority

**Tests:** extend `Tests/UI/test_console_conversation_inspector.py`, `Tests/UI/test_console_cost_chip_screen.py`, `Tests/UI/test_console_context_modal.py`.

**Interfaces:** preserve `TAB_COSTS`, `TAB_EXCHANGE`, `TAB_NEXT_SEND` IDs for
compatibility while changing their visible labels. Add required keyword-only
`conversation_title: str`, `target_profile_key: str`, and
`target_is_current: Callable[[], bool]` from the existing factory. The callback
checks profile/session/conversation existence and disclosure authority, not
merely which tab is active in the app. Update `_default_kwargs()` and direct
modal/harness constructions in the affected tests with explicit fixture title,
profile identity and validity callback; do not add a production permissive default.

- [ ] Add this lazy-load regression with the existing harness/helpers:

```python
@pytest.mark.asyncio
async def test_usage_entry_does_not_prepare_hidden_context():
    calls = []
    async def snapshot_factory():
        calls.append("prepared")
        return ConsoleContextSnapshot(current_messages=[], next_send_payload={})
    app = InspectorHarness(
        conversation_title="Test chat", target_profile_key="test-profile",
        target_is_current=lambda: True,
        rows=[_row()], totals=_totals(), turns=[_turn()],
        exchanges_loader=_empty_exchanges_loader,
        snapshot_factory=snapshot_factory, initial_tab=TAB_COSTS,
    )
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert calls == []
```

InspectorHarness forwards these keyword arguments to the real modal and loads
the production stylesheet. Do not replace it with a parallel test app.

- [ ] Run the named test; confirm the existing unconditional on_mount load fails it. Change mount so only active Context preview requests `_load_snapshot`. Tab/section activation schedules one exclusive worker; Usage and Exchange mounts do not prepare hidden live content. Keep generic entry Context and direct context entry its next-send preview; cost chip opens Usage & cost.
- [ ] Add a generation counter for target/snapshot requests. Reuse existing capture_revision_provider for trace authority and target_is_current for profile/session validity. Before committing a loaded body or export, reject stale generation/target/disclosure. On invalid authority, remove content and disable Copy/Save with a reason; on ordinary refresh failure with valid authority, retain content visibly labeled stale.
- [ ] Render title and target identity safely as literal text. Keep Close/Escape and correct focus return. Test delayed completion after profile change, deletion, dismissal and Safe/Full toggling; no hidden body may reappear. Run the three suites and commit: `refactor: load Inspector modal views on explicit entry`.

## Task 2: Context sections and a readable detail pane

**Create tests:** `Tests/UI/test_console_inspector_detail_pane.py`, `Tests/Chat/test_console_inspector_presentation.py`.
**Extend:** `Tests/UI/test_console_conversation_inspector.py`.

**Interfaces:**

```python
@dataclass(frozen=True, slots=True)
class InspectorSection:
    key: str
    group: str
    label: str
    item_count: int

@dataclass(frozen=True, slots=True)
class InspectorDetail:
    key: str
    text: str
    raw_json: str | None
    truncated: bool = False
```

Presentation exports `context_sections(snapshot: ConsoleContextSnapshot) -> tuple[InspectorSection, ...]`
and `context_detail(snapshot: ConsoleContextSnapshot, key: str) -> InspectorDetail`.
Section keys are `current:messages`, `preview:system`, `preview:messages`,
`preview:tools`, `preview:model`, plus present optional payload fields identified
by `preview:<field>`. Metadata-only section enumeration never flattens or copies
automatic instruction bodies into row labels. Preserve existing personal-context
and project-instruction disclosure paths rather than interpreting their raw internals.

`ConsoleInspectorDetailPane` takes section metadata and posts
`SectionSelected(key: str)`; the modal supplies detail after authority validation.
It does not own services, export policy or provider requests.

- [ ] Add a pure regression asserting sections separate current and preview:

```python
def test_section_labels_do_not_include_prompt_bodies():
    snapshot = ConsoleContextSnapshot(
        current_messages=[],
        next_send_payload={"system": "private body", "messages": [], "tools": []},
    )
    rows = context_sections(snapshot)
    assert "preview:system" in {row.key for row in rows}
    assert all("private body" not in row.label for row in rows)
```

- [ ] Run the new presentation suite before implementation, then build descriptors from the actual snapshot fields. Detail selection alone formats the chosen body. Use stable message IDs for current history, snapshot-scoped positions for prepared payload items, and invalidate selection detail on snapshot replacement. Never claim prepared payload indexes are durable identities.
- [ ] Replace nested Next Send TabbedContent and chains of Collapsible/TextArea widgets with section selection plus one read-only detail control. Raw JSON toggles the selected section only. Preserve a clearly labeled full-payload export route using existing snapshot serialization and policy guards; do not silently change an existing whole-payload export to selected-only output.
- [ ] Place model, estimated prepared-input count, budget/output reservation and freshness above detail. Source all budget values from current estimate/capacity services; unknown values remain unavailable. Bind summary and export to the same snapshot revision. Surface existing project-instruction recovery controls when their metadata indicates action is needed, preserving exact captured-target routing.
- [ ] Add mounted tests for explicit preview entry, zero messages, nonempty draft, tools/staged sources, compaction, project instructions, personal context, response prefill, raw/readable toggle, Refresh disabled during a run, stale preview, failed refresh and ephemeral export refusal. Preserve >1 MiB safeguards: show an explicit bounded/truncated view or existing export guidance, never render the whole body synchronously. Keep secrets out of labels/errors/logs.
- [ ] Run new pure/detail suites and the existing context/Inspector suites. Commit: `feat: make Inspector context a section and detail workflow`.

## Task 3: Usage list, identity and historical call detail

**Tests:** `Tests/Chat/test_console_inspector_presentation.py`, `Tests/UI/test_console_conversation_inspector.py`, `Tests/Chat/test_console_cost_tracker.py`.

**Interfaces:** `InspectorUsageItem(message_key: str, row: ConsoleCostRow, title: str)`
is a frozen presentation value. `usage_items(rows: Sequence[ConsoleCostRow], turns: Sequence[InspectorTurn]) -> tuple[InspectorUsageItem, ...]`
joins the existing row.index to turn.index once per immutable snapshot, producing
native_message_id keys. Missing/duplicate identity is an unavailable detail,
not a nearest-row guess. Refresh preserves selection by key, not index.

- [ ] Add a pure test with two rows/turns, then reorder the turn sequence; assert row detail still belongs to its native message ID. Add this cost-truth regression to the existing Inspector test module, where `_row` and `replace` already exist:

```python
def test_mixed_pricing_remains_unavailable_total():
    rows = [_row(), replace(_row(index=1), cost_usd=None)]
    totals = build_cost_rows_totals(rows)
    assert totals.total_cost_usd is None
```

Import build_cost_rows_totals from console_cost_tracker. This is a preservation
check; new displayed-label assertions must
fail until the modal says unavailable rather than presenting a partial sum.

- [ ] Render an aligned turn list with title, input/output tokens, cost and estimate/reported basis. Put totals and actual coverage above it. Keep cache/audio/transcription buckets in selected detail with plain labels; audio input is a subset, not another token addend. Do not relabel transcript estimates as exact spend.
- [ ] Reuse the exchanges_loader for one selected native message. Preserve abandoned flags and ordering by (created_at, seq); do not sum call prices into already aggregated row/totals. Link from selected usage detail to that same call in Exchange history without losing identity or disclosure profile.
- [ ] Reuse the new list/detail pane for historical turn/call inspection; keep adapter-boundary caveat, missing capture reasons, existing sanitization and exports. Safe/Full must be reachable from either trace-bearing view. Switching profile clears both views' rendered and cached bodies before new content is shown. Capture controls state that they apply to future capture and remain distinct from viewer disclosure.
- [ ] Add mounted tests for estimated/missing/zero prices, mixed provider models, abandoned/cancelled turns, duplicate/missing turn IDs, delayed old-call completion, capture off, purged history and Full-to-Safe export. Ensure a failure in trace loading does not blank valid usage totals.
- [ ] Run targeted presentation, modal and cost-tracker suites plus existing export/disclosure tests found by `rg -l 'TraceViewerProfile|ConsoleExchangeExportDialog' Tests/UI Tests/Chat`. Select those covering changed paths rather than running both entire directories. Commit: `feat: clarify Inspector usage and historical call detail`.

## Task 4: Adaptive modal, production rendering and closeout

**Tests:** `Tests/UI/test_console_inspector_detail_pane.py`, `Tests/UI/test_console_conversation_inspector.py`, `Tests/UI/test_console_modal_dismissal.py`.

- [ ] In the detail-pane tests, mount at wide width, select a section and scroll detail, shrink to 80x24, activate Back, and assert selection/scroll restoration. Widen again and assert the same selected key. Verify Tab/Enter/arrows/Escape and pointer Back/Close operate the actual focused controls, not offscreen DOM elements.
- [ ] Implement wide side-by-side layout and narrow list-to-detail using the same pane state. Determine the breakpoint from required list/detail widths and action labels; express it as a documented token-backed size law. Hide the inactive narrow pane without destroying its selection/scroll state. One scroll owner per pane; header/navigation/footer stay reachable. Avoid adding a second top-level modal or custom widget framework.
- [ ] Compose rest/hover/focus/disabled styles from existing tokens. Raw user titles/content use literal Text; no markup interpretation. Keep copy/export/Refresh controls within the selected view, label export scope, and retain visible Close. Do not retheme or change existing tokens globally.
- [ ] Rebuild CSS. Run `pytest Tests/UI/test_console_conversation_inspector.py Tests/UI/test_console_context_modal.py Tests/UI/test_console_cost_chip_screen.py Tests/UI/test_console_inspector_detail_pane.py Tests/UI/test_console_modal_dismissal.py Tests/UI/test_design_token_governance.py Tests/UI/test_css_bundle_sync_guard.py -q`, plus new presentation and affected export suites. Run touched-file lint/format and existing import/CSS budgets affected by the new widgets; preserve budgets.
- [ ] Use a disposable profile to exercise actual cost-chip and context entry routes with empty, populated, in-progress and long-capture conversations. Inspect 80x24, 120x40 and a wide viewport with production CSS; check actual painted labels, selection, keyboard focus, Back/Close, readable data and export scope. Capture evidence in supported terminal environments, recording any missing Windows evidence honestly.
- [ ] Update `Docs/User_Guide/console/context-and-rag.md` without overwriting existing edits. Record modal-only scope, navigation, freshness, accounting coverage and historical disclosure. Self-review to confirm no Inspector-sidebar changes. Add TASK-32828 notes, ADR-171 and evidence; mark Done only once every AC and repository DoD item is met. Commit explicit paths with `feat: finish adaptive Conversation Inspector modal`.

## Coverage and handoff

Entry points, identity, privacy and lazy loading: task 1. Context/preparation,
budget/freshness, recovery and export scope: task 2. Usage, estimates, historical
inspection and cross-view masks: task 3. Narrow navigation, tokens, keyboard,
documentation and production evidence: task 4. This plan does not change the
Inspector sidebar or claim implementation/visual verification has occurred.
