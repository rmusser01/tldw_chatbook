# Home Recents, Resume Deep-Links, and Next-Up Suggestions — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A "What you were working on last" recents stream on Home (conversations/notes/media, one-click resumable), a real conversation deep-link into Console, and "what's next" suggestions fed by open-task queues.

**Architecture:** Task 1 adds a Console navigation-context deep-link (store-and-defer consumption on the freshly-mounted screen) plus its ADR. Task 2 computes content recents inside HomeScreen's existing `HomeContentSnapshot` worker (async scope-service seams), merges them with the adapter's recents in the pure `dashboard_state` layer, promotes the resume banner to the newest content item (adding media as a resume kind), and wires row/banner dispatch. Task 3 feeds the fixed-priority ladder two new queue inputs (eval runs, read-it-later) via provider callables on the existing adapter, and upgrades the terminal suggestion to "Resume last conversation".

**Tech Stack:** Python ≥3.11, Textual ≥8.0.0,<9, SQLite (existing schemas only), pytest (+ Textual pilot for screen tests).

**Spec:** `Docs/superpowers/specs/2026-08-29-home-recents-resume-and-next-up-design.md` — the plan argues from the spec; executors read both.

## Global Constraints

- No new dependencies; no DB schema migrations; no new `config.toml` keys (spec Non-goals).
- User titles: RAW at the data layer, rich-markup-escaped EXACTLY ONCE at the Button-label build (`build_home_resume_control` / item-build for rail rows — existing hazard pattern, spec §1).
- All new queries run off the UI event loop, through the existing worker/`asyncio.to_thread` seams only (spec §1, §6).
- Parameterized SQL only (repo rule) — no new raw SQL is added by this plan.
- Targeted test runs only; never a full-suite sweep unless the user opts in (AGENTS.md testing rule).
- Conventional commit subjects (`feat:`, `test:`, `docs:`), matching repo history.
- Execute on a fresh branch off `main` (the current `docs/lesson-adr-number-collisions` branch is unrelated); use the using-git-worktrees skill at execution time.
- Every task: backlog task via CLI (create → In Progress → plan → Done with Implementation Notes), per AGENTS.md DoD.

## Deviations from spec (verified during planning — intent preserved)

1. **Content recents are computed in HomeScreen's `HomeContentSnapshot` pipeline, not in adapter providers.** Spec §1 named `_local_*_recent_items()` methods on `LocalNotificationHomeActiveWorkAdapter`, but the content seams (`notes_scope_service.list_notes`, `chat_conversation_scope_service.list_conversations`, `media_reading_scope_service.list_media_items`) are **async**, while the adapter's `build_dashboard_input` is synchronous + TTL-cached; the per-visit snapshot worker (`home_screen.py:309-366`) is the established home for exactly these seams and matches spec §5's per-visit freshness contract. Merging with adapter recents still happens in pure `dashboard_state`.
2. **Media recency = `Media.last_modified` alone** (spec's documented fallback): no list-level ReadingProgress seam exists — only per-media `get_reading_progress` (Client_Media_DB_v2.py:2730).
3. **`failed_schedule_count` producer skipped** (spec's decision rule): verified no source query exists (`Scheduling/db/scheduled_tasks_db.py` has no failed-status listing). Task 4 files a follow-up backlog task instead.
4. **Read-it-later suggestion routes to the Media screen root** (`TAB_MEDIA`): no registered deep-link view id exists for the read-it-later list.

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `backlog/decisions/NNN-console-conversation-deep-link-nav-context.md` | ADR for the cross-module seam | Create (Task 1) |
| `tldw_chatbook/Constants.py:44-61` | Nav-context contract keys | Add `CONSOLE_NAV_CONTEXT_CONVERSATION_ID` (Task 1) |
| `tldw_chatbook/UI/Screens/chat_screen.py` | Console screen | `apply_navigation_context` + deferred consume (Task 1) |
| `tldw_chatbook/Home/dashboard_state.py` | Pure Home state | Content-item constants/fields, merge, banner media branch, row Open control, ladder inputs/branches/terminal (Tasks 2-3) |
| `tldw_chatbook/UI/Screens/home_screen.py` | Home screen | Snapshot limit-N + content item builders + dispatch (Task 2); primary-action context (Task 3) |
| `tldw_chatbook/Home/active_work_adapter.py` | Adapter | Open-tasks provider plumbing (Task 3) |
| `tldw_chatbook/app.py` | App wiring | Provider closures (Task 3) |
| `Tests/UI/test_console_nav_context.py`, `Tests/Home/test_dashboard_state.py`, `Tests/Home/test_active_work_adapter.py`, `Tests/UI/test_home_screen.py` | Tests | Create/extend (Tasks 1-3) |
| `Docs/User_Guide/home.md` | User-facing docs | Update (Task 4) |

---

### Task 1: Console conversation deep-link seam + ADR

**Files:**
- Create: `backlog/decisions/NNN-console-conversation-deep-link-nav-context.md` (NNN = next free number)
- Modify: `tldw_chatbook/Constants.py:44-61`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (`__init__` at 3403, `on_mount` at 13437, near the other `_consume_pending_*` methods)
- Test: `Tests/UI/test_console_nav_context.py` (create)

**Interfaces:**
- Consumes: `NavigateToScreen(route, screen_context)` dispatch at `app.py:8937-8948` (calls `apply_navigation_context` pre-`switch_screen`, swallow-logged); `ConsoleWorkspaceController._console_session_id_for_workspace_conversation(conversation_id) -> str | None` (workspace.py:2100); `await ConsoleWorkspaceController._resume_console_workspace_conversation(conversation_id) -> bool | None` (workspace.py:2121); `ChatScreen._ensure_console_chat_controller()` (chat_screen.py:5428) whose controller has `.store.active_session_id` and `.switch_session(session_id)` (chat_screen.py:19876-19883).
- Produces: `CONSOLE_NAV_CONTEXT_CONVERSATION_ID = "conversation_id"` (Constants); `ChatScreen.apply_navigation_context(context) -> None`; `ChatScreen._consume_pending_console_nav_conversation() -> None`; pending state attr `_pending_console_nav_conversation_id: str`. Tasks 2-3 navigate with `NavigateToScreen(TAB_CHAT, {CONSOLE_NAV_CONTEXT_CONVERSATION_ID: <id>})`.

- [ ] **Step 1: Create the backlog task**

```bash
backlog task create "Console conversation deep-link nav context" -d "Add CONSOLE_NAV_CONTEXT_CONVERSATION_ID and ChatScreen.apply_navigation_context so Home/ladder can deep-link a specific conversation into Console (spec 2026-08-29 §3)" --ac "Nav context with a conversation id lands that conversation in Console,Nav context wins over a pending CONSOLE_LIVE_WORK handoff,Missing id is a no-op,ADR written and linked" -s "In Progress"
```

Note the printed task id for the Done step.

- [ ] **Step 2: Write the ADR**

Pick the number: `ls backlog/decisions/ | grep -E '^[0-9]+' | sort -n | tail -1` → NNN = that + 1 (repo is mid-renumbering; verify no collision).

```markdown
# NNN. Console conversation deep-link via navigation context

Date: 2026-08-29
Status: Accepted

## Context

Home's resume control and next-best-action ladder need to open a specific
conversation in Console. Until now nothing app-level could: the capability
existed only inside the Console workspace controller
(`_resume_console_workspace_conversation`), and Home's resume button routed
bare to `chat`, dropping the conversation id (task-190 limitation).

## Decision

Deep-link conversations through the existing navigation-context contract:
a new `CONSOLE_NAV_CONTEXT_CONVERSATION_ID` key; `ChatScreen` implements
`apply_navigation_context` (the 6th screen to do so, after Library,
Watchlists, Personas, Settings, STTs). The framework calls it BEFORE the
screen is mounted (`handle_screen_navigation`, app.py:8937), so the
implementation is store-and-defer: record the id synchronously, consume it
from the mount timer chain. Consumption prefers switching to a live session
holding that conversation over rehydrating from the DB.

Precedence: an explicit nav deep link outranks any pending handoff staged
for the same mount; applying the context clears a pending
`CONSOLE_LIVE_WORK` handoff (the nav context is the user's explicit,
most-recent intent).

## Alternatives considered

- Pending-handoff single-slot store: ephemeral, single-slot, not a durable
  deep-link contract; reserved for runtime handoffs (live work, fleet
  completions), not navigation.
- Always rehydrate from DB: loses open live sessions and their draft state;
  the session-map check is cheap and correct.

## Consequences

Console gains a second entry-path that must not fight the handoff
consumers; ordering is pinned by timer registration (nav at 0.15s alongside
the other consumers, clearing the live-work handoff at apply time).
```

- [ ] **Step 3: Write the failing tests**

Create `Tests/UI/test_console_nav_context.py`. Reuse the sibling harness: read `Tests/UI/test_console_live_work_handoffs.py:100-200` first and mirror its app-construction + `ChatScreen` push (`ConsoleHarness` at line 128 pushes `ChatScreen(self.app_instance)` and `_wait_for_production_chat_screen` at 141 polls for a mounted screen). The tests below are harness-agnostic once you have a mounted `screen`:

```python
"""Console navigation-context deep-link tests (spec 2026-08-29 §3)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.Constants import CONSOLE_NAV_CONTEXT_CONVERSATION_ID


def _install_fakes(screen, *, live_session_id=None):
    """Replace Console collaborators with recorders (logic-level fakes)."""
    resume = AsyncMock(return_value=True)
    screen._workspace = SimpleNamespace(
        _console_session_id_for_workspace_conversation=lambda cid: live_session_id,
        _resume_console_workspace_conversation=resume,
        _set_active_workspace_for_console_session=lambda sid: None,
    )
    switched = []
    screen._ensure_console_chat_controller = lambda: SimpleNamespace(
        store=SimpleNamespace(active_session_id="other-session"),
        switch_session=switched.append,
    )
    workers = []
    screen.run_worker = lambda coro, **kwargs: workers.append(coro)
    screen.call_after_refresh = lambda *args, **kwargs: None
    screen._sync_native_console_chat_ui = AsyncMock()
    screen._focus_console_composer_if_needed = lambda **kwargs: None
    return SimpleNamespace(resume=resume, switched=switched, workers=workers)


def test_apply_navigation_context_stores_conversation_id(screen):
    assert screen._pending_console_nav_conversation_id == ""
    screen.apply_navigation_context({CONSOLE_NAV_CONTEXT_CONVERSATION_ID: "42"})
    assert screen._pending_console_nav_conversation_id == "42"


def test_apply_navigation_context_ignores_missing_or_non_mapping(screen):
    screen.apply_navigation_context({})
    screen.apply_navigation_context({"unrelated": "x"})
    assert screen._pending_console_nav_conversation_id == ""


def test_consume_prefers_live_session_over_rehydrate(screen):
    fakes = _install_fakes(screen, live_session_id="sess-7")
    screen._pending_console_nav_conversation_id = "42"
    screen._consume_pending_console_nav_conversation()
    assert screen._pending_console_nav_conversation_id == ""
    assert fakes.switched == ["sess-7"]
    fakes.resume.assert_not_awaited()
    assert fakes.workers == []


def test_consume_rehydrates_when_no_live_session(screen):
    fakes = _install_fakes(screen, live_session_id=None)
    screen._pending_console_nav_conversation_id = "42"
    screen._consume_pending_console_nav_conversation()
    assert fakes.switched == []
    assert len(fakes.workers) == 1  # the resume coroutine was scheduled


def test_consume_noop_without_pending_id(screen):
    fakes = _install_fakes(screen, live_session_id="sess-7")
    screen._consume_pending_console_nav_conversation()
    assert fakes.switched == []
    assert fakes.workers == []
```

If direct import of the sibling harness is awkward, copy its app-builder into this file (bounded copy, referenced by line range above). Fixtures `screen` come from that harness (module-scoped pilot fixture returning the mounted `ChatScreen`).

- [ ] **Step 4: Run tests to verify they fail**

Run: `pytest Tests/UI/test_console_nav_context.py -v`
Expected: FAIL — `AttributeError: 'ChatScreen' object has no attribute '_pending_console_nav_conversation_id'` (or ImportError on the new constant).

- [ ] **Step 5: Add the constant**

In `tldw_chatbook/Constants.py`, after the Watchlists nav-context block (line 61):

```python
# Console navigation-context contract keys.
# Applied pre-mount by handle_screen_navigation (app.py) -- ChatScreen
# stores the id and consumes it from its mount timer chain. See
# backlog/decisions/NNN-console-conversation-deep-link-nav-context.md.
CONSOLE_NAV_CONTEXT_CONVERSATION_ID = "conversation_id"
```

(Use the real ADR number in the comment.)

- [ ] **Step 6: Implement store-and-defer on ChatScreen**

In `chat_screen.py` `ChatScreen.__init__` (line 3403), next to the other pending-handoff state:

```python
        # Navigation-context deep link (spec 2026-08-29 §3): stored
        # synchronously by apply_navigation_context BEFORE mount, consumed
        # from the mount timer chain below.
        self._pending_console_nav_conversation_id: str = ""
```

In `on_mount` (line 13437), next to the other handoff timers (after line 13461's `_consume_pending_chat_handoff` timer):

```python
        # Nav-context deep link: same settle hedge as the handoff timers.
        self.set_timer(0.15, self._consume_pending_console_nav_conversation)
```

Near the other `_consume_pending_*` methods:

```python
    def apply_navigation_context(self, context: Mapping[str, Any]) -> None:
        """Store a shell-navigation deep link for post-mount consumption.

        Called by ``handle_screen_navigation`` BEFORE this screen is
        mounted, so this must stay synchronous: record the conversation id
        and consume it from the mount timer chain. Precedence (ADR NNN):
        an explicit nav deep link outranks any pending handoff staged for
        this mount, so the live-work handoff is cleared here.
        """
        if not isinstance(context, Mapping):
            return
        conversation_id = str(
            context.get(CONSOLE_NAV_CONTEXT_CONVERSATION_ID, "") or ""
        ).strip()
        if not conversation_id:
            return
        self._pending_console_nav_conversation_id = conversation_id
        try:
            self.app_instance.pending_handoff_store.clear_pending(
                HandoffChannel.CONSOLE_LIVE_WORK
            )
            logger.debug(
                "Console nav deep link claimed this mount; cleared pending "
                "CONSOLE_LIVE_WORK handoff."
            )
        except AttributeError:
            # Test doubles may not expose the store; the deep link itself
            # still works.
            pass

    def _consume_pending_console_nav_conversation(self) -> None:
        """Switch to (or rehydrate) the nav-deep-linked conversation."""
        conversation_id = self._pending_console_nav_conversation_id
        self._pending_console_nav_conversation_id = ""
        if not conversation_id:
            return
        session_id = (
            self._workspace._console_session_id_for_workspace_conversation(
                conversation_id
            )
        )
        if session_id is not None:
            controller = self._ensure_console_chat_controller()
            if controller.store.active_session_id != session_id:
                self._workspace._set_active_workspace_for_console_session(
                    session_id
                )
                controller.switch_session(session_id)
                self.call_after_refresh(self._sync_native_console_chat_ui)
            self.call_after_refresh(
                self._focus_console_composer_if_needed, force=True
            )
            return
        self.run_worker(
            self._workspace._resume_console_workspace_conversation(
                conversation_id
            ),
            exclusive=False,
        )
```

Add imports at the top of `chat_screen.py` (verify against how `_consume_pending_chat_handoff` accesses the store — if it uses a different accessor, match it): `from ..Navigation.pending_handoff_store import HandoffChannel` and `from ...Constants import CONSOLE_NAV_CONTEXT_CONVERSATION_ID` merged into the existing Constants import.

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest Tests/UI/test_console_nav_context.py -v`
Expected: PASS (5 tests).

- [ ] **Step 8: Run targeted regression tests**

Run: `pytest Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_screen_navigation.py -v`
Expected: PASS (no behavior changed for existing callers — nobody sends Console contexts yet).

- [ ] **Step 9: Commit + close the backlog task**

```bash
git add tldw_chatbook/Constants.py tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_nav_context.py backlog/decisions/
git commit -m "feat(console): conversation deep-link navigation context seam"
backlog task edit <id> -s Done --notes "Added CONSOLE_NAV_CONTEXT_CONVERSATION_ID + ChatScreen.apply_navigation_context (store-and-defer, live-session-first, live-work handoff precedence). ADR linked."
```

---

### Task 2: Content recents stream + resume banner + row Open control

**Files:**
- Modify: `tldw_chatbook/Home/dashboard_state.py` (constants ~114-123, `HomeDashboardInput` 181-225, `HomeContentSnapshot` 228-243, `apply_home_content_snapshot` 246-280, `build_home_resume_control` 684-720, `build_home_controls` 417-500, `build_home_triage_state` 1140-1354)
- Modify: `tldw_chatbook/UI/Screens/home_screen.py` (`_build_home_content_snapshot` 367-422, `_home_resume_fields` 201-230 (retire), dispatch 751-833)
- Test: `Tests/Home/test_dashboard_state.py` (extend), `Tests/UI/test_home_screen.py` (extend)

**Interfaces:**
- Consumes (Task 1): `CONSOLE_NAV_CONTEXT_CONVERSATION_ID`.
- Produces: `HOME_RESUME_KIND_MEDIA`, `LOCAL_CONVERSATION_ITEM_ID_PREFIX = "local:conversation:"`, `LOCAL_NOTE_ITEM_ID_PREFIX = "local:note:"`, `LOCAL_MEDIA_ITEM_ID_PREFIX = "local:media:"`, `HOME_OPEN_ITEM_CONTROL_ID = "home-open-item"`, `HOME_RECENT_WORK_LIMIT = 8`, `content_item_kind(item_id) -> str | None`, `combined_recent_work_items(state) -> tuple[HomeActiveWorkItem, ...]`; `HomeDashboardInput.content_recent_items` / `.resume_updated_at`; `HomeContentSnapshot.content_recent_items` / `.resume_updated_at`; `HomeScreen._open_content_item(target_id) -> None`.

- [ ] **Step 1: Create the backlog task**

```bash
backlog task create "Home content recents stream + resume banner" -d "Merge conversations/notes/media into Home's Recent section from the content snapshot pipeline, promote the resume banner to the newest content item (media becomes a resume kind), retire the limit-1 resume queries, add a row Open control (spec 2026-08-29 §1-2)" --ac "Recent section shows mixed content recents newest-first capped at 8,Banner resumes newest content item incl. media with relative age,Conversation banner/row opens that conversation in Console via nav context,Note/media rows open their Library views,limit-1 resume seam queries retired" -s "In Progress"
```

- [ ] **Step 2: Write the failing pure-state tests**

Append to `Tests/Home/test_dashboard_state.py` (inline-input construction pattern, per the existing tests):

```python
def _content_item(item_id, updated_at, *, source="Notes", route="library"):
    return HomeActiveWorkItem(
        item_id=item_id,
        title=f"title {item_id}",
        source=source,
        status="ready",
        detail_route=route,
        updated_at=updated_at,
    )


def test_content_item_kind_maps_prefixes():
    assert content_item_kind("local:conversation:42") == "conversation"
    assert content_item_kind("local:note:7") == "note"
    assert content_item_kind("local:media:9") == "media"
    assert content_item_kind("local:ingest:3") is None
    assert content_item_kind("") is None


def test_combined_recent_merges_and_caps_by_recency():
    adapter_recents = (_content_item("local:ingest:1", "2026-08-29T10:00:00+00:00"),)
    content_items = (
        _content_item("local:conversation:5", "2026-08-29T12:00:00+00:00",
                      source="Conversations", route="chat"),
        _content_item("local:note:2", "2026-08-29T11:00:00+00:00"),
        _content_item("local:media:3", "2026-08-28T09:00:00+00:00", source="Media"),
    )
    state = HomeDashboardInput(
        recent_work_items=adapter_recents, content_recent_items=content_items
    )
    merged = combined_recent_work_items(state)
    assert [item.item_id for item in merged] == [
        "local:conversation:5",
        "local:note:2",
        "local:ingest:1",
        "local:media:3",
    ]


def test_combined_recent_caps_at_limit():
    items = tuple(
        _content_item(f"local:note:{i}", f"2026-08-29T1{i:02d}:00:00+00:00")
        for i in range(12)
    )
    state = HomeDashboardInput(content_recent_items=items)
    assert len(combined_recent_work_items(state)) == HOME_RECENT_WORK_LIMIT


def test_triage_recent_section_includes_content_items_and_open_control():
    content = (
        _content_item("local:conversation:5", "2026-08-29T12:00:00+00:00",
                      source="Conversations", route="chat"),
    )
    state = HomeDashboardInput(
        model_ready=True, content_recent_items=content, console_ready=True
    )
    triage = build_home_triage_state(state, selected_row_id="local:conversation:5")
    recent_section = next(s for s in triage.sections if s.section_id == "recent")
    assert [row.row_id for row in recent_section.rows] == ["local:conversation:5"]
    control_ids = {c.control_id for c in triage.canvas.actions}
    assert HOME_OPEN_ITEM_CONTROL_ID in control_ids
    open_control = next(
        c for c in triage.canvas.actions if c.control_id == HOME_OPEN_ITEM_CONTROL_ID
    )
    assert open_control.target_id == "local:conversation:5"
    assert open_control.target_route == "chat"


def test_resume_control_supports_media_kind_with_age():
    state = HomeDashboardInput(
        resume_kind=HOME_RESUME_KIND_MEDIA,
        resume_id="9",
        resume_title="Long read",
        resume_updated_at="2026-08-29T11:00:00+00:00",
    )
    control = build_home_resume_control(
        state, now=datetime(2026, 8, 29, 11, 30, tzinfo=timezone.utc)
    )
    assert control is not None
    assert control.control_id == HOME_RESUME_LATEST_CONTROL_ID
    assert control.target_route == "library"
    assert control.target_id == "local:media:9"
    assert control.label == "Resume reading: Long read (30m)"
```

Add the new names to the file's existing `dashboard_state` import block.

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest Tests/Home/test_dashboard_state.py -v -k "content_item_kind or combined_recent or triage_recent_section or resume_control_supports_media"`
Expected: FAIL — `ImportError` (names not defined).

- [ ] **Step 4: Implement the pure-state layer in `dashboard_state.py`**

Constants (next to `HOME_RESUME_KIND_*`, lines 114-117):

```python
HOME_RESUME_KIND_MEDIA = "media"
# Shared recents cap (the adapter keeps its own alias; this module is the
# canonical home so the pure merge can enforce it without importing the
# adapter).
HOME_RECENT_WORK_LIMIT = 8
LOCAL_CONVERSATION_ITEM_ID_PREFIX = "local:conversation:"
LOCAL_NOTE_ITEM_ID_PREFIX = "local:note:"
LOCAL_MEDIA_ITEM_ID_PREFIX = "local:media:"
# Canvas control that opens the SELECTED content recents row (screen-level
# dispatch, like home-resume-latest -- not a HOME_CONTROL_METHODS hook).
HOME_OPEN_ITEM_CONTROL_ID = "home-open-item"


def content_item_kind(item_id: str) -> str | None:
    """Return the content kind for a prefixed content item id, else None."""
    for kind, prefix in (
        (HOME_RESUME_KIND_CONVERSATION, LOCAL_CONVERSATION_ITEM_ID_PREFIX),
        (HOME_RESUME_KIND_NOTE, LOCAL_NOTE_ITEM_ID_PREFIX),
        (HOME_RESUME_KIND_MEDIA, LOCAL_MEDIA_ITEM_ID_PREFIX),
    ):
        if item_id.startswith(prefix):
            return kind
    return None


def combined_recent_work_items(
    state: HomeDashboardInput,
) -> tuple[HomeActiveWorkItem, ...]:
    """Merge adapter recents with content recents, newest-first, capped."""
    merged = list(state.recent_work_items) + list(state.content_recent_items)
    merged.sort(key=lambda item: item.updated_at, reverse=True)
    return tuple(merged[:HOME_RECENT_WORK_LIMIT])
```

`HomeDashboardInput` — add after `resume_title` (line 225):

```python
    # Content recents (conversations/notes/media) from the HomeScreen
    # content-snapshot pipeline; merged into the Recent rail section by
    # combined_recent_work_items. Titles are markup-escaped at build.
    content_recent_items: tuple[HomeActiveWorkItem, ...] = ()
    resume_updated_at: str = ""
```

`HomeContentSnapshot` — add the same two fields. `apply_home_content_snapshot` — add both to the `replace(...)` call:

```python
        content_recent_items=snapshot.content_recent_items,
        resume_updated_at=snapshot.resume_updated_at,
```

`build_home_resume_control` — media branch + age + prefixed target id:

```python
def build_home_resume_control(
    state: HomeDashboardInput,
    *,
    now: datetime | None = None,
) -> HomeControl | None:
    """Build the one-click resume control for the newest content item.

    The raw user title is markup-escaped HERE, exactly once (Button labels
    parse Rich markup). Conversation targets carry the Console nav-context
    prefix; note/media targets carry their Library prefixes so dispatch
    (_open_content_item) can route by kind.
    """
    if not state.resume_id:
        return None
    if state.resume_kind == HOME_RESUME_KIND_NOTE:
        kind_label = "note"
        target_route = "library"
        fallback_title = "Latest note"
        prefix = LOCAL_NOTE_ITEM_ID_PREFIX
    elif state.resume_kind == HOME_RESUME_KIND_CONVERSATION:
        kind_label = "conversation"
        target_route = "chat"
        fallback_title = "Latest conversation"
        prefix = LOCAL_CONVERSATION_ITEM_ID_PREFIX
    elif state.resume_kind == HOME_RESUME_KIND_MEDIA:
        kind_label = "reading"
        target_route = "library"
        fallback_title = "Latest media"
        prefix = LOCAL_MEDIA_ITEM_ID_PREFIX
    else:
        return None
    title = state.resume_title.strip() or fallback_title
    label = f"Resume {kind_label}: {escape_markup(title)}"
    age = format_console_relative_age(
        state.resume_updated_at, now=now or datetime.now(timezone.utc)
    )
    if age:
        label = f"{label} ({age})"
    return HomeControl(
        HOME_RESUME_LATEST_CONTROL_ID,
        label,
        target_route,
        "resume_latest",
        f"{prefix}{state.resume_id}",
    )
```

(`format_console_relative_age` is already imported at dashboard_state.py:23-25.)

`build_home_controls` — after the `detail_item` resolution (line 500), add the selected-content Open control:

```python
    # Content recents rows (conversations/notes/media) get a single Open
    # control that deep-links the item (screen-level dispatch, same as
    # home-resume-latest).
    if selected_item is not None and content_item_kind(selected_item.item_id):
        controls.append(
            HomeControl(
                HOME_OPEN_ITEM_CONTROL_ID,
                "Open",
                selected_item.detail_route,
                "open_content",
                selected_item.item_id,
            )
        )
```

(Place it before the approval controls so it does not disturb the existing control order assertions — verify against existing `control_ids` tests; if ordering assertions break, append at the end of the function instead.)

`build_home_triage_state` — two edits:

```python
    recent_rows = tuple(
        _item_row(item, "recent", reference_now)
        for item in combined_recent_work_items(state)
    )
```

and the selected-row item lookup (line 1244-1248) must also see content items:

```python
            item = next(
                i
                for i in tuple(state.active_work_items)
                + combined_recent_work_items(state)
                if i.item_id == selected.row_id
            )
```

Update the Recent section's empty copy (line 1208) to `"Conversations, notes, media, runs, chatbooks, and imports will appear here."`

- [ ] **Step 5: Run the pure-state tests to verify they pass**

Run: `pytest Tests/Home/test_dashboard_state.py -v`
Expected: PASS (new + existing; existing `summarize`/control tests unaffected — `content_recent_items` defaults to `()`).

- [ ] **Step 6: Write the failing screen-level tests**

Extend `Tests/UI/test_home_screen.py` (it already pilots Home and asserts nav contexts — follow its `_build_test_app` + `_home_dashboard_test_input` pattern):

```python
def test_open_content_item_routes_by_prefix(app_with_home):
    home = app_with_home  # mounted HomeScreen from the file's existing helper
    sent = []
    home.post_message = lambda message: sent.append(message)

    home._open_content_item("local:conversation:42")
    home._open_content_item("local:note:7")
    home._open_content_item("local:media:9")

    assert sent[0].screen_context == {CONSOLE_NAV_CONTEXT_CONVERSATION_ID: "42"}
    assert sent[1].screen_context == {LIBRARY_NAV_CONTEXT_NOTE_ID: "7"}
    assert sent[2].screen_context == {
        LIBRARY_NAV_CONTEXT_OPEN_SOURCE_TYPE: "media",
        LIBRARY_NAV_CONTEXT_OPEN_SOURCE_ID: "9",
    }
    assert sent[0].screen_name == TAB_CHAT
```

(Adapt attribute names — `screen_name`/`screen_context` — to `NavigateToScreen`'s actual fields at `UI/Navigation/main_navigation.py:81`; the file's existing tests already assert on them.)

- [ ] **Step 7: Run to verify failure**

Run: `pytest Tests/UI/test_home_screen.py -v -k open_content_item`
Expected: FAIL — `AttributeError: '_activate...' object has no attribute '_open_content_item'`.

- [ ] **Step 8: Implement the snapshot builders + dispatch in `home_screen.py`**

Replace `_home_resume_fields` (lines 201-230) and extend `_build_home_content_snapshot` (367-422). New module-level code (keep `_HOME_RESUME_TITLE_MAX_CHARS`):

```python
_HOME_CONTENT_FETCH_LIMIT = 8


def _home_content_records(
    notes_result: Any,
    conversations_result: Any,
    media_result: Any,
) -> list[tuple[str, Mapping[str, Any]]]:
    """Merge the three content seams into (kind, raw record) newest-first."""
    merged: list[tuple[str, Mapping[str, Any]]] = []
    for kind, result, records_key in (
        ("conversation", conversations_result, "items"),
        ("note", notes_result, None),
        ("media", media_result, "items"),
    ):
        records = _home_records_from_result(result, records_key)
        for record in records:
            if isinstance(record, Mapping) and record.get("id") not in (None, ""):
                merged.append((kind, record))
    merged.sort(
        key=lambda pair: _home_record_timestamp(pair[1]), reverse=True
    )
    return merged


def _home_records_from_result(
    result: Any, records_key: str | None
) -> list[Any]:
    """Extract the record list from a seam response (list or dict)."""
    if isinstance(result, Mapping):
        records = result.get(records_key) if records_key else result.get("items")
        # list_notes-style seams may return a bare list under "items"; the
        # local notes seam returns the list directly.
        if records is None and records_key is None:
            records = result.get("notes")
        return list(records or [])
    if isinstance(result, list):
        return result
    return []


def _home_content_resume_fields(
    merged: list[tuple[str, Mapping[str, Any]]],
) -> tuple[str, str, str, str]:
    """(resume_kind, resume_id, raw truncated title, updated_at) of the top."""
    if not merged:
        return "", "", "", ""
    kind, record = merged[0]
    title = " ".join(str(record.get("title") or "").split())
    if len(title) > _HOME_RESUME_TITLE_MAX_CHARS:
        title = title[: _HOME_RESUME_TITLE_MAX_CHARS - 1].rstrip() + "…"
    return (
        kind,
        str(record.get("id")),
        title,
        _home_record_timestamp(record).isoformat(),
    )


_HOME_CONTENT_SOURCE_LABELS = {
    "conversation": ("Conversations", "chat"),
    "note": ("Notes", "library"),
    "media": ("Media", "library"),
}
_HOME_CONTENT_ID_PREFIXES = {
    "conversation": LOCAL_CONVERSATION_ITEM_ID_PREFIX,
    "note": LOCAL_NOTE_ITEM_ID_PREFIX,
    "media": LOCAL_MEDIA_ITEM_ID_PREFIX,
}


def _home_content_recent_items(
    merged: list[tuple[str, Mapping[str, Any]]],
    *,
    exclude_id: str,
) -> tuple[HomeActiveWorkItem, ...]:
    """Build rail-ready content items, excluding the banner item.

    Titles are markup-escaped HERE (Button labels parse Rich markup); the
    banner's raw title is handled separately by _home_content_resume_fields.
    """
    items: list[HomeActiveWorkItem] = []
    for kind, record in merged:
        item_id = f"{_HOME_CONTENT_ID_PREFIXES[kind]}{record.get('id')}"
        if item_id == exclude_id:
            continue
        source, route = _HOME_CONTENT_SOURCE_LABELS[kind]
        raw_title = " ".join(str(record.get("title") or "").split())
        title = escape_markup(raw_title) or escape_markup(
            f"{kind.title()} {record.get('id')}"
        )
        items.append(
            HomeActiveWorkItem(
                item_id=item_id,
                title=title,
                source=source,
                status="ready",
                detail_route=route,
                console_available=(kind == "conversation"),
                updated_at=_home_record_timestamp(record).isoformat(),
            )
        )
    return tuple(items)
```

Delete `_home_resume_fields` (its two call sites go away). In `_build_home_content_snapshot`, change the three seam calls from `limit=1`/`results_per_page=1` to the fetch limit and build the new fields:

```python
        notes_result = await self._home_content_seam_call(
            getattr(notes_service, "list_notes", None),
            scope="local_note",
            limit=_HOME_CONTENT_FETCH_LIMIT,
            user_id=notes_user_id,
        )
        conversations_result = await self._home_content_seam_call(
            getattr(conversation_service, "list_conversations", None),
            mode="local",
            scope_type="all",
            limit=_HOME_CONTENT_FETCH_LIMIT,
            offset=0,
        )
        media_result = await self._home_content_seam_call(
            getattr(media_service, "list_media_items", None),
            mode="local",
            page=1,
            results_per_page=_HOME_CONTENT_FETCH_LIMIT,
            include_keywords=False,
        )

        merged = _home_content_records(
            notes_result, conversations_result, media_result
        )
        resume_kind, resume_id, resume_title, resume_updated_at = (
            _home_content_resume_fields(merged)
        )
        return HomeContentSnapshot(
            console_ready=console_ready,
            conversation_count=_home_response_total(conversations_result),
            note_count=(
                note_count_result if isinstance(note_count_result, int) else None
            ),
            media_count=_home_response_total(media_result),
            resume_kind=resume_kind,
            resume_id=resume_id,
            resume_title=resume_title,
            resume_updated_at=resume_updated_at,
            content_recent_items=_home_content_recent_items(
                merged,
                exclude_id=(
                    f"{_HOME_CONTENT_ID_PREFIXES[resume_kind]}{resume_id}"
                    if resume_kind and resume_id
                    else ""
                ),
            ),
        )
```

Extend imports in `home_screen.py`: `LOCAL_CONVERSATION_ITEM_ID_PREFIX, LOCAL_MEDIA_ITEM_ID_PREFIX, LOCAL_NOTE_ITEM_ID_PREFIX` from `Home.dashboard_state`, `escape` from `rich.markup`, `CONSOLE_NAV_CONTEXT_CONVERSATION_ID, TAB_CHAT` from `Constants` (merge into existing imports).

Dispatch — replace `_activate_home_resume_latest` (lines 806-832) and add the two new methods; also intercept the Open control in `_activate_home_control` (after the `HOME_RESUME_LATEST_CONTROL_ID` intercept, line 761):

```python
        if button_id == HOME_OPEN_ITEM_CONTROL_ID:
            self._activate_home_open_item()
            return
```

```python
    def _activate_home_resume_latest(self) -> None:
        """Route the resume-latest control to its one-click destination."""
        control = next(
            (
                item
                for item in self._current_canvas_controls
                if item.control_id == HOME_RESUME_LATEST_CONTROL_ID
            ),
            None,
        )
        if control is None or not control.target_id:
            return
        self._open_content_item(control.target_id)

    def _activate_home_open_item(self) -> None:
        """Route the selected content row's Open control."""
        control = next(
            (
                item
                for item in self._current_canvas_controls
                if item.control_id == HOME_OPEN_ITEM_CONTROL_ID
            ),
            None,
        )
        if control is None or not control.target_id:
            return
        self._open_content_item(control.target_id)

    def _open_content_item(self, target_id: str) -> None:
        """Deep-link a prefixed content item id to its surface."""
        kind = content_item_kind(target_id)
        if kind == HOME_RESUME_KIND_CONVERSATION:
            self.post_message(
                NavigateToScreen(
                    TAB_CHAT,
                    {
                        CONSOLE_NAV_CONTEXT_CONVERSATION_ID: (
                            target_id.removeprefix(
                                LOCAL_CONVERSATION_ITEM_ID_PREFIX
                            )
                        )
                    },
                )
            )
        elif kind == HOME_RESUME_KIND_NOTE:
            self.post_message(
                NavigateToScreen(
                    TAB_LIBRARY,
                    {
                        LIBRARY_NAV_CONTEXT_NOTE_ID: target_id.removeprefix(
                            LOCAL_NOTE_ITEM_ID_PREFIX
                        )
                    },
                )
            )
        elif kind == HOME_RESUME_KIND_MEDIA:
            self.post_message(
                NavigateToScreen(
                    TAB_LIBRARY,
                    {
                        LIBRARY_NAV_CONTEXT_OPEN_SOURCE_TYPE: "media",
                        LIBRARY_NAV_CONTEXT_OPEN_SOURCE_ID: (
                            target_id.removeprefix(LOCAL_MEDIA_ITEM_ID_PREFIX)
                        ),
                    },
                )
            )
```

Update the `dashboard_state` import in `home_screen.py` to add `HOME_OPEN_ITEM_CONTROL_ID, HOME_RESUME_KIND_CONVERSATION, HOME_RESUME_KIND_MEDIA, HOME_RESUME_KIND_NOTE, content_item_kind` (drop the now-unused `_home_resume_fields` import if it was imported; it was module-local, so likely nothing to remove). Also update the docstring of `_activate_home_resume_latest` — conversations now carry the id.

- [ ] **Step 9: Run screen tests**

Run: `pytest Tests/UI/test_home_screen.py Tests/Home/ -v`
Expected: PASS. If an existing test asserted `"Runs, chatbooks, imports, and schedules will appear here."` or the old resume-label format, update it to the new copy.

- [ ] **Step 10: Manual verification (live)**

Run the app (`python3 -m tldw_chatbook.app`), navigate Home: Recent shows conversations/notes/media rows; selecting a conversation row shows Open; Open lands that conversation in Console (live session switch if open); the idle banner shows the newest content item with age; a media banner resumes into the Library item view.

- [ ] **Step 11: Commit + close the backlog task**

```bash
git add tldw_chatbook/Home/dashboard_state.py tldw_chatbook/UI/Screens/home_screen.py Tests/Home/test_dashboard_state.py Tests/UI/test_home_screen.py
git commit -m "feat(home): content recents stream, media resume banner, row open deep-links"
backlog task edit <id> -s Done --notes "Content recents via HomeContentSnapshot pipeline; merged in pure dashboard_state; banner promoted incl. media kind; limit-1 resume queries retired; row Open control dispatches by prefix."
```

---

### Task 3: Ladder feeds (eval runs, read-it-later) + terminal resume suggestion

**Files:**
- Modify: `tldw_chatbook/Home/dashboard_state.py` (`HomeDashboardInput`, `choose_next_best_action` 314-403)
- Modify: `tldw_chatbook/Home/active_work_adapter.py` (`__init__` 217-245, `build_dashboard_input` 297+)
- Modify: `tldw_chatbook/app.py` (adapter wiring ~6912, new provider methods near `_local_flashcards_due_count`)
- Modify: `tldw_chatbook/UI/Screens/home_screen.py` (`_home_primary_action_context` 97-113, call site 747)
- Test: `Tests/Home/test_dashboard_state.py`, `Tests/Home/test_active_work_adapter.py`

**Interfaces:**
- Consumes (Task 1): `CONSOLE_NAV_CONTEXT_CONVERSATION_ID`.
- Produces: `HomeDashboardInput.pending_eval_run_count: int`, `.failed_eval_run_count: int`, `.read_later_count: int | None`; adapter `__init__` kwargs `eval_open_runs_provider: Callable[[], Mapping[str, int]] | None`, `read_later_count_provider: Callable[[], int | None] | None`, method `refresh_open_tasks_snapshot() -> None`; app methods `TldwCli._local_eval_open_run_counts() -> dict[str, int]`, `TldwCli._local_read_later_count() -> int | None`; new action ids `review_eval_runs`, `review_read_later`, `resume_last_conversation`.

- [ ] **Step 1: Create the backlog task**

```bash
backlog task create "Home ladder open-task feeds + terminal resume suggestion" -d "Feed pending/failed eval runs and read-it-later count into the next-best-action ladder; terminal start-console becomes Resume last conversation when one exists (spec 2026-08-29 §4)" --ac "Ladder suggests reviewing eval runs when pending/failed runs exist,Ladder suggests read-it-later when queue non-empty,Terminal suggestion deep-links the newest conversation via nav context,running eval runs never counted,Default adapter and missing services degrade to no suggestion" -s "In Progress"
```

- [ ] **Step 2: Write the failing ladder tests**

Append to `Tests/Home/test_dashboard_state.py`:

```python
def test_ladder_suggests_eval_runs_over_import_sources():
    action = choose_next_best_action(
        HomeDashboardInput(
            model_ready=True,
            has_library_content=True,
            rag_ready=True,
            console_ready=True,
            pending_eval_run_count=2,
        )
    )
    assert action.action_id == "review_eval_runs"
    assert action.target_route == TAB_EVALS


def test_ladder_never_counts_running_eval_runs():
    action = choose_next_best_action(
        HomeDashboardInput(
            model_ready=True, has_library_content=True, rag_ready=True,
            console_ready=True, pending_eval_run_count=0, failed_eval_run_count=0,
        )
    )
    assert action.action_id != "review_eval_runs"


def test_ladder_suggests_read_later():
    action = choose_next_best_action(
        HomeDashboardInput(
            model_ready=True, has_library_content=True, rag_ready=True,
            console_ready=True, read_later_count=3,
        )
    )
    assert action.action_id == "review_read_later"
    assert action.target_route == TAB_MEDIA


def test_ladder_terminal_suggestion_resumes_last_conversation():
    action = choose_next_best_action(
        HomeDashboardInput(
            model_ready=True, has_library_content=True, rag_ready=True,
            console_ready=True,
            resume_kind=HOME_RESUME_KIND_CONVERSATION,
            resume_id="42",
        )
    )
    assert action.action_id == "resume_last_conversation"
    assert action.target_route == TAB_CHAT


def test_ladder_terminal_falls_back_to_start_conversation_without_recents():
    action = choose_next_best_action(
        HomeDashboardInput(
            model_ready=True, has_library_content=True, rag_ready=True,
            console_ready=True,
        )
    )
    assert action.action_id == "start_console"
```

Add `choose_next_best_action` and `TAB_*` constants to the test imports (`from tldw_chatbook.Constants import TAB_EVALS, TAB_MEDIA, TAB_CHAT`).

- [ ] **Step 3: Run to verify failure**

Run: `pytest Tests/Home/test_dashboard_state.py -v -k "ladder"`
Expected: FAIL — `HomeDashboardInput` has no such fields / branches return `search_library`.

- [ ] **Step 4: Implement the ladder in `dashboard_state.py`**

Extend the Constants import (line 13-18) with `TAB_CHAT, TAB_EVALS, TAB_MEDIA`. Add fields to `HomeDashboardInput` after `flashcards_due_count` (line 207):

```python
    # Open-task queue feeds (spec §4). Eval counts include only
    # pending/failed -- 'running' rows are orphaned forever by a crash and
    # would pin the suggestion. read_later None = unknown, renders nothing.
    pending_eval_run_count: int = 0
    failed_eval_run_count: int = 0
    read_later_count: int | None = None
```

In `choose_next_best_action`, after the `notification_count` branch (line 376) and before `has_library_content` (line 377):

```python
    if state.pending_eval_run_count or state.failed_eval_run_count:
        return HomeAction(
            "review_eval_runs",
            "Review eval runs",
            TAB_EVALS,
            "Pending or failed eval runs need attention.",
        )
    if state.read_later_count:
        return HomeAction(
            "review_read_later",
            f"Read-it-later: {state.read_later_count} items",
            TAB_MEDIA,
            "Your saved reading queue is waiting.",
        )
```

Replace the terminal `console_ready` branch (lines 391-400):

```python
    if state.console_ready:
        if (
            state.resume_kind == HOME_RESUME_KIND_CONVERSATION
            and state.resume_id
        ):
            return HomeAction(
                "resume_last_conversation",
                "Resume last conversation",
                TAB_CHAT,
                "Pick up where you left off.",
            )
        return HomeAction(
            "start_console",
            "Start a conversation",
            TAB_CHAT,
            "Console is ready for a task.",
        )
```

- [ ] **Step 5: Run ladder tests**

Run: `pytest Tests/Home/test_dashboard_state.py -v`
Expected: PASS (if an existing test pinned the old terminal branch ordering, update its expectation to the new contract — the spec explicitly changes it).

- [ ] **Step 6: Write the failing adapter + wiring tests**

Append to `Tests/Home/test_active_work_adapter.py`:

```python
def test_open_tasks_providers_feed_dashboard_input():
    adapter = LocalNotificationHomeActiveWorkAdapter(
        eval_open_runs_provider=lambda: {"pending": 1, "failed": 2},
        read_later_count_provider=lambda: 5,
    )
    adapter.refresh_open_tasks_snapshot()
    state = adapter.build_dashboard_input(
        providers_models={}, has_recent_work=False
    )
    assert state.pending_eval_run_count == 1
    assert state.failed_eval_run_count == 2
    assert state.read_later_count == 5


def test_open_tasks_providers_degrade_quietly():
    def boom():
        raise RuntimeError("db unavailable")

    adapter = LocalNotificationHomeActiveWorkAdapter(
        eval_open_runs_provider=boom,
        read_later_count_provider=boom,
    )
    adapter.refresh_open_tasks_snapshot()
    state = adapter.build_dashboard_input(
        providers_models={}, has_recent_work=False
    )
    assert state.pending_eval_run_count == 0
    assert state.failed_eval_run_count == 0
    assert state.read_later_count is None


def test_open_tasks_absent_providers_default_off():
    adapter = LocalNotificationHomeActiveWorkAdapter()
    adapter.refresh_open_tasks_snapshot()
    state = adapter.build_dashboard_input(
        providers_models={}, has_recent_work=False
    )
    assert state.pending_eval_run_count == 0
    assert state.read_later_count is None
```

- [ ] **Step 7: Run to verify failure**

Run: `pytest Tests/Home/test_active_work_adapter.py -v -k open_tasks`
Expected: FAIL — unexpected keyword `eval_open_runs_provider`.

- [ ] **Step 8: Implement adapter + app wiring**

`active_work_adapter.py` — extend `__init__` (lines 217-245) with kwargs `eval_open_runs_provider: Callable[[], Mapping[str, int]] | None = None` and `read_later_count_provider: Callable[[], int | None] | None = None`, store them plus two cache attrs:

```python
        self.eval_open_runs_provider = eval_open_runs_provider
        self.read_later_count_provider = read_later_count_provider
        self._eval_open_counts: Mapping[str, int] = {"pending": 0, "failed": 0}
        self._read_later_count: int | None = None
```

Add the refresh method (mirror `refresh_flashcards_due_snapshot`'s degrade pattern):

```python
    def refresh_open_tasks_snapshot(self) -> None:
        """Refresh cached open-task counts off the Home compose path."""
        if callable(self.eval_open_runs_provider):
            try:
                counts = self.eval_open_runs_provider() or {}
                self._eval_open_counts = {
                    "pending": max(0, int(counts.get("pending", 0) or 0)),
                    "failed": max(0, int(counts.get("failed", 0) or 0)),
                }
            except Exception as e:
                logger.debug(f"Failed to fetch eval run counts for Home: {e}")
                self._eval_open_counts = {"pending": 0, "failed": 0}
        if callable(self.read_later_count_provider):
            try:
                count = self.read_later_count_provider()
                self._read_later_count = (
                    max(0, int(count)) if count is not None else None
                )
            except Exception as e:
                logger.debug(f"Failed to fetch read-it-later count for Home: {e}")
                self._read_later_count = None
```

In `LocalNotificationHomeActiveWorkAdapter.build_dashboard_input` (line 297), where the input is assembled, add:

```python
            pending_eval_run_count=self._eval_open_counts.get("pending", 0),
            failed_eval_run_count=self._eval_open_counts.get("failed", 0),
            read_later_count=self._read_later_count,
```

(The base `UnavailableHomeActiveWorkAdapter.build_dashboard_input` keeps dataclass defaults — no change.)

In `home_screen.py` `_refresh_home_chatbook_artifact_snapshot` (the thread worker, lines 279-303), after the flashcards block:

```python
        refresh_open_tasks = getattr(adapter, "refresh_open_tasks_snapshot", None)
        if callable(refresh_open_tasks):
            refresh_open_tasks()
```

In `app.py` — extend the adapter construction (~6912-6924):

```python
            eval_open_runs_provider=lambda: self._local_eval_open_run_counts(),
            read_later_count_provider=lambda: self._local_read_later_count(),
```

and add the two providers next to `_local_flashcards_due_count` (find it via `grep -n "_local_flashcards_due_count" tldw_chatbook/app.py`):

```python
    def _local_eval_open_run_counts(self) -> dict[str, int]:
        """Count pending/failed local eval runs for Home (spec §4).

        Never counts 'running' -- a crashed app orphans running rows
        forever, which would permanently pin the review suggestion.
        """
        service = getattr(self, "local_evaluation_service", None)
        if service is None:
            return {"pending": 0, "failed": 0}
        try:
            pending = len(service.list_runs(status="pending", limit=50))
            failed = len(service.list_runs(status="failed", limit=50))
        except Exception as e:
            logger.debug(f"Failed to fetch eval run counts for Home: {e}")
            return {"pending": 0, "failed": 0}
        return {"pending": pending, "failed": failed}

    def _local_read_later_count(self) -> int | None:
        db = getattr(self, "media_db", None)
        if db is None:
            return None
        try:
            return len(db.list_read_it_later_media_ids())
        except Exception as e:
            logger.debug(f"Failed to fetch read-it-later count for Home: {e}")
            return None
```

- [ ] **Step 9: Wire the terminal suggestion's deep-link context**

In `home_screen.py`, change the signature and add the branch:

```python
def _home_primary_action_context(
    action: object, dashboard_input: HomeDashboardInput | None = None
) -> dict[str, object]:
    action_id = getattr(action, "action_id", None)
    if action_id == "resume_last_conversation":
        resume_id = str(getattr(dashboard_input, "resume_id", "") or "")
        if resume_id:
            return {CONSOLE_NAV_CONTEXT_CONVERSATION_ID: resume_id}
    ...  # existing branches unchanged
```

Update the single call site (`_activate_home_primary_action`, line 747):

```python
            screen_context=_home_primary_action_context(
                dashboard.next_action, self._current_dashboard_input
            ),
```

`CONSOLE_NAV_CONTEXT_CONVERSATION_ID` joins the Constants import in `home_screen.py` (from Task 2).

- [ ] **Step 10: Run adapter + screen tests**

Run: `pytest Tests/Home/test_active_work_adapter.py Tests/UI/test_home_screen.py -v`
Expected: PASS.

- [ ] **Step 11: Commit + close the backlog task**

```bash
git add tldw_chatbook/Home/dashboard_state.py tldw_chatbook/Home/active_work_adapter.py tldw_chatbook/app.py tldw_chatbook/UI/Screens/home_screen.py Tests/Home/
git commit -m "feat(home): eval/read-it-later ladder feeds and resume-last-conversation terminal suggestion"
backlog task edit <id> -s Done --notes "Ladder gains review_eval_runs + review_read_later branches (pending/failed only); terminal suggestion deep-links newest conversation via Console nav context; providers degrade quietly."
```

---

### Task 4: Docs, follow-up tasks, and wrap-up

**Files:**
- Modify: `Docs/User_Guide/home.md`
- Create: two follow-up backlog tasks (To Do)

**Interfaces:** none (documentation + hygiene).

- [ ] **Step 1: Create the backlog task**

```bash
backlog task create "Home recents/resume docs + follow-ups" -d "Update the Home User Guide for content recents, media resume, new suggestions; file follow-up tasks for failed_schedule_count producer and phase-2 opens journal (spec 2026-08-29 wrap-up)" --ac "User Guide reflects new Recent mix, banner kinds, and suggestions,Follow-up tasks filed for failed_schedule_count and opens journal,All feature tasks Done with notes" -s "In Progress"
```

- [ ] **Step 2: Update `Docs/User_Guide/home.md`**

Update the sections describing the Recent rail (now conversations/notes/media + runs/chatbooks/imports, newest-first, capped at 8), the idle banner (resume conversation/note/reading with age; conversations open at that conversation in Console), and Next (new eval-runs and read-it-later suggestions; terminal "Resume last conversation"). Keep the documented "snapshot with buttons" quirk as-is (task-2763 still open). Verify against the rendered guide conventions in that file.

- [ ] **Step 3: File follow-up tasks**

```bash
backlog task create "Produce failed_schedule_count for Home" -d "HomeDashboardInput.failed_schedule_count is a documented dead input (no producer). Needs a failed-schedules query in Scheduling/db/scheduled_tasks_db.py first -- skipped per spec 2026-08-29 decision rule" --ac "failed_schedule_count populated from real schedule state,Ladder recover_schedules branch reachable"
backlog task create "Home opens journal (phase 2 recents)" -d "Persistent opens journal (IDs+timestamps, model_catalog_cache pattern) so read-only sessions count as recent work; feeds recents ranking and task-18921 usage-ranked suggestions (spec 2026-08-29 Non-goals)" --ac "Opening an item bumps its recency without edits,Journal storage contract documented,Usage-ranked suggestion feasibility assessed"
```

- [ ] **Step 4: Lessons check**

Review `backlog/docs/lessons-*.md` against what actually happened. Only add an entry if a generalizable trap surfaced (most tasks produce nothing — do not invent one). Candidate only if encountered: e.g. the async-scope-seams vs sync-adapter mismatch that moved content recents into the snapshot pipeline.

- [ ] **Step 5: Self-review + close**

Confirm every AC checkbox on all four backlog tasks is checked, Implementation Notes sections exist, ADR is linked from tasks 1-3 notes, and commit:

```bash
git add Docs/User_Guide/home.md
git commit -m "docs(home): content recents, media resume, and next-up suggestions"
backlog task edit <id> -s Done --notes "Guide updated; follow-ups filed for failed_schedule_count and opens journal."
```

---

## Plan Self-Review (completed)

- **Spec coverage:** §1 recents stream → Task 2 (deviation 1 documented); §2 banner + media kind + retirement → Task 2; §3 seam + ADR + precedence → Task 1; §4 feeds + terminal + eval-staleness rule → Task 3 (deviation 3: schedules follow-up); §5 freshness regression guard → covered by the unchanged on-mount path + existing `Tests/UI/test_home_dashboard_seams.py` (no new refresh machinery per spec); §6 edge cases → escape-once (Task 2 code), missing-record toasts (Task 1 reuses TASK-717 path), empty states (default `()` fields), off-loop reads (existing workers); testing section → all four test groups present.
- **Placeholders:** none — every code step carries verbatim code; the two located adaptations (sibling test harness reuse in Task 1 Step 3, `NavigateToScreen` field names in Task 2 Step 6) are bounded with exact references.
- **Type consistency:** `content_item_kind`/`combined_recent_work_items`/`HOME_OPEN_ITEM_CONTROL_ID` names and the `local:{kind}:{id}` prefix scheme are identical across Tasks 2-3; `_home_primary_action_context(action, dashboard_input)` signature change has its single call site updated in the same task; `CONSOLE_NAV_CONTEXT_CONVERSATION_ID` is produced in Task 1 and consumed in Tasks 2-3.
