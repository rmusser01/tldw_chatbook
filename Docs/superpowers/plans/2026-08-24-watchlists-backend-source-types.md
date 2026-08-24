# Watchlists backend-aware source types implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Watchlists New source form expose and submit only the source types and fields supported by the backend shown when the user presses Create.

**Architecture:** Local and Server services publish separate create-form source-type contracts while retaining their broader persistence contracts. The scope service and UI controller expose the selected contract; `SourcesPane` owns labels, form state, pre-dispatch validation, and backend-specific payload shaping; `WatchlistsCollectionsScreen` keeps mounted panes synchronized and carries the submission-time backend through creation, destination filing, and confirmation.

**Tech Stack:** Python 3.11+, Textual 8.x reactives/messages/widgets, pytest/pytest-asyncio, existing Watchlists local/server services and full-shell UI harness.

**Design spec:** `Docs/superpowers/specs/2026-08-24-watchlists-backend-source-types-design.md`

**ADR required:** no  
**ADR path:** N/A  
**Reason:** This is a bounded contract correction inside the existing Watchlists local/server routing boundary. It changes no schema, persistence ownership, API shape, dependency, security boundary, or long-lived application structure.

---

## File map

- `tldw_chatbook/Subscriptions/local_watchlists_service.py` — publish the Local form subset and keep full Local payload validation separate; validate before opening the DB.
- `tldw_chatbook/Subscriptions/server_watchlists_service.py` — publish the Server form contract and reuse it for Server source-type validation.
- `tldw_chatbook/Subscriptions/watchlist_scope_service.py` — return the active backend's form contract without UI labels.
- `tldw_chatbook/UI/Watchlists_Modules/watchlists_backend_controller.py` — expose the scope contract to the screen through its existing controller seam.
- `tldw_chatbook/UI/Watchlists_Modules/sources_pane.py` — split filter/create vocabularies, render backend-specific fields, preserve complete drafts, validate at the form boundary, and include the submission backend in the create message.
- `tldw_chatbook/UI/Screens/watchlists_collections_screen.py` — seed and live-sync the pane contract, mirror Active/cadence drafts, and route/file/confirm against the captured backend.
- `Tests/Subscriptions/test_local_watchlists_service.py` — pin the broader Local service vocabulary and pre-DB rejection.
- `Tests/Subscriptions/test_server_watchlists_service.py` — pin the Server contract and pre-client rejection.
- `Tests/Subscriptions/test_watchlist_scope_service.py` — pin backend contract routing and real Server call shape.
- `Tests/Watchlists/test_watchlists_backend_controller.py` — pin controller passthrough.
- `Tests/Watchlists/test_watchlists_sources_pane.py` — pin form options, payloads, validation copy, and draft preservation.
- `Tests/UI/full_app_destination_context.py` — teach the existing full-shell service double the new synchronous contract query.
- `Tests/UI/test_watchlists_source_create_form.py` — pin live backend switching, submission-time backend capture, focus order, and geometry.
- `backlog/tasks/task-2510 - Source-type-options-offer-values-the-local-service-rejects.md` — record the plan, checked acceptance criteria, evidence, and implementation notes.

No CSS change is planned: the Server form removes controls and the existing tags row already expands with `width: 1fr`. Modify `tldw_chatbook/css/features/_watchlists.tcss` and regenerate `tldw_chatbook/css/tldw_cli_modular.tcss` only if the full-shell geometry test demonstrates a real defect.

### Task 1: Publish and route backend create-form contracts

**Files:**
- Modify: `Tests/Subscriptions/test_local_watchlists_service.py`
- Modify: `Tests/Subscriptions/test_server_watchlists_service.py`
- Modify: `Tests/Subscriptions/test_watchlist_scope_service.py`
- Modify: `Tests/Watchlists/test_watchlists_backend_controller.py`
- Modify: `tldw_chatbook/Subscriptions/local_watchlists_service.py`
- Modify: `tldw_chatbook/Subscriptions/server_watchlists_service.py`
- Modify: `tldw_chatbook/Subscriptions/watchlist_scope_service.py`
- Modify: `tldw_chatbook/UI/Watchlists_Modules/watchlists_backend_controller.py`

- [ ] **Step 1: Write failing service-contract tests**

Add focused tests with these assertions:

```python
def test_local_form_source_types_are_a_narrow_subset_of_accepted_payload_types():
    assert LocalWatchlistsService.CREATE_FORM_SOURCE_TYPES == ("rss", "atom", "url")
    assert LocalWatchlistsService._local_type_for_source_type("sitemap") == "sitemap"
    assert "sitemap" not in LocalWatchlistsService.CREATE_FORM_SOURCE_TYPES


@pytest.mark.asyncio
async def test_local_create_rejects_unknown_type_before_opening_database():
    db_factory = Mock(side_effect=AssertionError("database must not open"))
    service = LocalWatchlistsService(db_factory=db_factory)
    with pytest.raises(ValueError, match="playlist"):
        await service.create_source({"name": "Bad", "url": "https://x", "source_type": "playlist"})
    db_factory.assert_not_called()


@pytest.mark.asyncio
async def test_server_create_rejects_unknown_type_before_client_dispatch():
    client = FakeWatchlistsClient()
    service = ServerWatchlistsService(client=client)
    with pytest.raises(ValueError, match="rss, site, and forum"):
        await service.create_source(name="Bad", url="https://x", source_type="playlist")
    assert client.calls == []
```

Give `FakeLocalWatchlists` and `FakeServerWatchlists` matching `CREATE_FORM_SOURCE_TYPES` constants, then add:

```python
def test_scope_service_routes_create_form_source_types_to_active_backend():
    scope = WatchlistScopeService(
        local_service=FakeLocalWatchlists(),
        server_service=FakeServerWatchlists(),
    )
    assert scope.create_form_source_types(runtime_backend="local") == ("rss", "atom", "url")
    assert scope.create_form_source_types(runtime_backend="server") == ("rss", "site", "forum")
```

Also parameterize `rss`, `site`, and `forum` through a real
`ServerWatchlistsService(FakeWatchlistsClient())` mounted behind a real
`WatchlistScopeService`. For each value, call `create_watch_item` with only the
common form payload and assert the fake client's `create_watchlist_source`
request carries that exact type. This pins the real `payload` → `**kwargs` →
`SourceCreateRequest` signature for every Server form option.

Extend `FakeScopeService` and add a controller passthrough assertion.

- [ ] **Step 2: Run the new contract tests and confirm they fail for the missing API/order**

Run:

```bash
python -m pytest -q \
  Tests/Subscriptions/test_local_watchlists_service.py::test_local_form_source_types_are_a_narrow_subset_of_accepted_payload_types \
  Tests/Subscriptions/test_local_watchlists_service.py::test_local_create_rejects_unknown_type_before_opening_database \
  Tests/Subscriptions/test_server_watchlists_service.py::test_server_create_rejects_unknown_type_before_client_dispatch \
  Tests/Subscriptions/test_watchlist_scope_service.py::test_scope_service_routes_create_form_source_types_to_active_backend \
  Tests/Subscriptions/test_watchlist_scope_service.py::test_scope_routes_every_server_form_type_through_real_service_signature \
  Tests/Watchlists/test_watchlists_backend_controller.py::test_create_form_source_types_routes_to_scope_service
```

Expected: FAIL because the constants/accessors do not exist and Local opens its DB before validating the type.

- [ ] **Step 3: Implement the minimal service and routing contracts**

Add only these public constants and accessors:

```python
class LocalWatchlistsService:
    CREATE_FORM_SOURCE_TYPES = ("rss", "atom", "url")

    async def create_source(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        local_type = self._local_type_for_source_type(payload.get("source_type"))
        db = self._db()
        ...


class ServerWatchlistsService:
    CREATE_FORM_SOURCE_TYPES = ("rss", "site", "forum")

    @classmethod
    def _validate_source_type(cls, source_type: Any) -> str:
        normalized = str(source_type or "").strip()
        if normalized not in cls.CREATE_FORM_SOURCE_TYPES:
            raise ValueError(
                "Only rss, site, and forum watchlist sources are supported in this slice."
            )
        return normalized


class WatchlistScopeService:
    def create_form_source_types(
        self, *, runtime_backend: WatchlistBackend | str | None = None
    ) -> tuple[str, ...]:
        backend = self._normalize_backend(runtime_backend)
        service = self._service_for_backend(backend)
        return tuple(service.CREATE_FORM_SOURCE_TYPES)


class WatchlistsBackendController:
    def create_form_source_types(
        self, *, runtime_backend: str | None = None
    ) -> tuple[str, ...]:
        backend = self._normalize_backend(runtime_backend)
        return tuple(
            self.scope_service.create_form_source_types(runtime_backend=backend)
        )
```

Do not narrow `_local_type_for_source_type`; Local imports/programmatic callers must retain `json_feed`, `url_list`, `podcast`, `sitemap`, and `api`.

- [ ] **Step 4: Run the service and routing tests**

Run the exact command from Step 2.

Expected: PASS.

- [ ] **Step 5: Commit the contract slice**

```bash
git add \
  tldw_chatbook/Subscriptions/local_watchlists_service.py \
  tldw_chatbook/Subscriptions/server_watchlists_service.py \
  tldw_chatbook/Subscriptions/watchlist_scope_service.py \
  tldw_chatbook/UI/Watchlists_Modules/watchlists_backend_controller.py \
  Tests/Subscriptions/test_local_watchlists_service.py \
  Tests/Subscriptions/test_server_watchlists_service.py \
  Tests/Subscriptions/test_watchlist_scope_service.py \
  Tests/Watchlists/test_watchlists_backend_controller.py
git commit -m "fix: publish watchlist form source contracts"
```

### Task 2: Make SourcesPane backend-aware and preserve the whole draft

**Files:**
- Modify: `Tests/Watchlists/test_watchlists_sources_pane.py`
- Modify: `tldw_chatbook/UI/Watchlists_Modules/sources_pane.py`

- [ ] **Step 1: Update the harness and write failing pane tests**

Change captured create messages to include `message.runtime_backend`. Add tests that prove:

```python
def option_pairs(select: Select) -> list[tuple[str, object]]:
    return [(str(label), value) for label, value in select._options]
```

- Local create options are `[("RSS", "rss"), ("Atom", "atom"), ("Web page", "url")]`.
- Server create options are `[("RSS", "rss"), ("Site", "site"), ("Forum", "forum")]`.
- `#sources-type-select` retains the existing filter pairs, including Feed, Playlist, Channel, and Web page.
- Calling the pane's backend configuration path Local → Server → Local keeps the form open and preserves name, URL, Active false, tags, destination, frequency `86400`, and selectors while an incompatible `url` type becomes `rss`.
- Server submission includes only `name`, `url`, `source_type`, `active`, `tags`, and `watchlist_id`; cadence and selectors are absent.
- A stale Select option `sitemap` is rejected before a `CreateSourceRequested`
  message even though the broader Local persistence service accepts it. The
  populated form stays open and no event is posted.
- A stale `playlist` option notifies exactly `Local sources don't support
  'Playlist'. Choose RSS, Atom, or Web page.` with `severity="error",
  markup=False`.
- The equivalent Server rejection notifies exactly `Server sources don't
  support 'Playlist'. Choose RSS, Site, or Forum.` with markup disabled.
- Unregistered values containing controls/whitespace, exceeding 40 characters,
  and containing no displayable text follow the exact normalization contract
  from the spec, including the `Unknown` fallback.

Simulate a stale DOM value without weakening the pane contract:

```python
type_select = pane.query_one("#sources-create-type", Select)
type_select.set_options([("Playlist", "playlist")])
type_select.value = "playlist"
```

- [ ] **Step 2: Run the new pane tests and verify red**

Run:

```bash
python -m pytest -q \
  Tests/Watchlists/test_watchlists_sources_pane.py -k \
  "backend_specific or preserves_complete_draft or server_payload or unsupported_form_type or source_type_recovery"
```

Expected: FAIL because the pane still shares `_TYPE_OPTIONS`, always renders/queries Local fields, does not mirror Active/cadence, and the create message has no backend.

- [ ] **Step 3: Split filter labels from create contracts**

Keep the current filter list byte-for-byte in purpose, rename it to `_FILTER_TYPE_OPTIONS`, and add one UI label registry containing the union of filter and form values:

```python
_SOURCE_TYPE_LABELS = {
    "rss": "RSS",
    "atom": "Atom",
    "feed": "Feed",
    "playlist": "Playlist",
    "channel": "Channel",
    "url": "Web page",
    "site": "Site",
    "forum": "Forum",
}
_FILTER_TYPE_OPTIONS = [
    ("All", "all"),
    ("RSS", "rss"),
    ("Atom", "atom"),
    ("Feed", "feed"),
    ("Playlist", "playlist"),
    ("Channel", "channel"),
    ("Web page", "url"),
]
```

Add pane state:

```python
create_runtime_backend = reactive("local")
create_form_source_types = reactive[tuple[str, ...]](
    ("rss", "atom", "url"), recompose=True
)
create_draft_active = reactive(True)
create_draft_frequency = reactive(_DEFAULT_FREQUENCY_SECONDS)
```

Add one `configure_create_backend(backend, source_types)` method that validates `backend`, requires `rss` in the non-empty tuple, updates the backend before the tuple, and normalizes only an incompatible `create_draft_source_type` to `rss`. Use `set_reactive` on an unmounted pane and normal assignments on a mounted pane.

- [ ] **Step 4: Extend draft messages and render backend-specific controls**

Add `active: bool` and `frequency: int` to `CreateFormDraftChanged`, `_post_create_draft_changed`, screen-seeding callers, and reset logic. Seed the Switch and frequency Select from these drafts. Handle `Switch.Changed` and the frequency branch of `Select.Changed` so every user change is mirrored before a recompose.

Build create type options only from `create_form_source_types` and `_SOURCE_TYPE_LABELS`. Render the tags row for both backends, but render `#sources-create-frequency` only for Local; render noise selectors only when Local and the chosen type is URL-family.

- [ ] **Step 5: Add exact pre-dispatch validation and payload shaping**

Extend the message:

```python
class CreateSourceRequested(Message):
    def __init__(self, runtime_backend: str, payload: dict[str, Any]) -> None:
        self.runtime_backend = runtime_backend
        self.payload = payload
        super().__init__()
```

Immediately after reading the mounted type Select, reject a value outside `create_form_source_types`. Implement the fallback label exactly:

```python
raw_text = str(value)
text = strip_control_characters(raw_text)
text = " ".join(text.split())
if len(text) > 40:
    text = f"{text[:39]}…"
display = _SOURCE_TYPE_LABELS.get(raw_text, text or "Unknown")
```

Build the common payload first. Add `check_frequency` and `ignore_selectors` only when `create_runtime_backend == "local"`; never query controls that are absent on Server. Post `CreateSourceRequested(self.create_runtime_backend, payload)` only after validation succeeds.

- [ ] **Step 6: Run all SourcesPane tests**

Run:

```bash
python -m pytest -q Tests/Watchlists/test_watchlists_sources_pane.py
```

Expected: PASS, including existing Local persistence/cadence/noise tests.

- [ ] **Step 7: Commit the pane slice**

```bash
git add \
  tldw_chatbook/UI/Watchlists_Modules/sources_pane.py \
  Tests/Watchlists/test_watchlists_sources_pane.py
git commit -m "fix: render backend-aware watchlist source form"
```

### Task 3: Live-sync backend changes and bind creation to submission context

**Files:**
- Modify: `Tests/UI/full_app_destination_context.py`
- Modify: `Tests/UI/test_watchlists_source_create_form.py`
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py`

- [ ] **Step 1: Teach the full-shell double the contract query**

Add the synchronous method used by the real controller:

```python
def create_form_source_types(self, *, runtime_backend=None):
    return (
        ("rss", "site", "forum")
        if runtime_backend == "server"
        else ("rss", "atom", "url")
    )
```

- [ ] **Step 2: Write failing live backend-switch and Server-layout tests**

In the production-stylesheet harness, open the Local form, populate all fields (including Active false, destination, daily cadence, and selectors), then change `#watchlists-backend-select` to Server. Assert on the same mounted `SourcesPane` that:

- the form remains open;
- type options become RSS/Site/Forum and incompatible `url` becomes RSS;
- cadence and selectors are absent;
- common values and Active false remain visible.

Switch back to Local and assert daily cadence and selectors reappear with their exact draft values. This test must use the real selector/watcher, not call the pane configuration method directly.

In the same red phase, define the Server form's expected focus order:

```python
SERVER_FIELD_ORDER = [
    "sources-create-name",
    "sources-create-url",
    "sources-create-type",
    "sources-create-active",
    "sources-create-watchlist",
    "sources-create-tags",
    "sources-create-submit",
    "sources-create-cancel",
]
```

Parameterize the existing tab-walk and whole-form-fit tests over Local RSS,
Local Web page, and Server RSS at both `(160, 42)` and `(235, 52)`. Drive the
Server case through `#watchlists-backend-select` before opening the form and
assert that neither cadence nor selectors is mounted. These cases must be
written now, before the screen starts pushing backend contracts, so they fail
against the current always-Local form.

- [ ] **Step 3: Write a failing captured-backend race test**

Use an `asyncio.Event`-gated fake controller create. Submit a Local source with a real watchlist destination, wait until the create enters, switch the screen to Server, then release it. Assert:

```python
assert create_call["runtime_backend"] == "local"
assert membership_was_filed_in_the_local_watchlist
assert confirmation_names_the_chosen_watchlist
assert source_reload_backend == "server"
```

Also keep/add a supported-type controller `ValueError` test asserting the existing generic `Failed to create source.` toast, so only the pane's pre-dispatch unsupported-type branch gets specialized copy.

- [ ] **Step 4: Run the new screen tests and verify red**

Run:

```bash
python -m pytest -q \
  Tests/UI/test_watchlists_source_create_form.py -k \
  "backend_switch_preserves or submission_backend or unrelated_create_failure or tab_walks_the_create_form_in_visual_order or whole_create_form_fits_inside_the_sources_pane"
```

Expected: FAIL because backend changes do not update a mounted pane, the screen
mirror lacks Active/cadence, the Server cases still render Local fields, and
`_create_source` reads `self.runtime_backend` after dispatch.

- [ ] **Step 5: Seed and push the active contract**

Add `_create_form_source_types(runtime_backend)` and `_sync_live_source_create_backend()` helpers on the screen. In `_build_detail_pane`, configure a new pane before seeding/opening the form. In `watch_runtime_backend`, normalize the screen's mirrored draft type against the new tuple, then push backend/contract into an already-mounted `SourcesPane` without rebuilding the region.

Extend screen mirror fields with `_source_create_draft_active` and `_source_create_draft_frequency`; seed them into rebuilt panes and update them in `handle_source_create_draft_changed`.

- [ ] **Step 6: Capture backend through the full create completion path**

Use the event's backend in the handler:

```python
self.run_worker(
    self._create_source(event.payload, runtime_backend=event.runtime_backend),
    exclusive=True,
    group="wc_create_source",
)
```

Make `_create_source` require the keyword-only captured backend. Pass it to `_controller.create_source`, `_file_created_source`, and confirmation logic. Let `_load_sources()` continue reading current `self.runtime_backend`, because its job is to refresh the view currently on screen.

Make `_tree_write_disabled_reason` accept an optional backend override, defaulting to current behavior, so `_file_created_source(..., runtime_backend=captured_backend)` can distinguish a captured Local write from a later Server selection while retaining the missing-bundle-service guard.

- [ ] **Step 7: Run the screen create-form tests**

Run:

```bash
python -m pytest -q Tests/UI/test_watchlists_source_create_form.py
```

Expected: PASS.

- [ ] **Step 8: Commit the screen synchronization slice**

```bash
git add \
  tldw_chatbook/UI/Screens/watchlists_collections_screen.py \
  Tests/UI/full_app_destination_context.py \
  Tests/UI/test_watchlists_source_create_form.py
git commit -m "fix: bind watchlist source creation to submitted backend"
```

### Task 4: Run scoped verification and close TASK-2510

**Files:**
- Modify: `backlog/tasks/task-2510 - Source-type-options-offer-values-the-local-service-rejects.md`
- Modify only if a geometry failure proves necessary: `tldw_chatbook/css/features/_watchlists.tcss`
- Regenerate only if CSS changes: `tldw_chatbook/css/tldw_cli_modular.tcss`

- [ ] **Step 1: Confirm the now-green geometry cases and make only demonstrated CSS corrections**

Run:

```bash
python -m pytest -q \
  Tests/UI/test_watchlists_source_create_form.py -k \
  "tab_walks_the_create_form_in_visual_order or whole_create_form_fits_inside_the_sources_pane"
```

Expected: PASS without CSS edits. If and only if a failure shows tags/buttons clipped or misordered, adjust the source stylesheet, run `python tldw_chatbook/css/build_css.py`, and rerun these two tests plus `python -m pytest -q Tests/UI/test_css_bundle_sync.py` (or the repository's matching bundle-sync test selected by `rg -n "bundle_sync" Tests`).

- [ ] **Step 2: Run the complete focused TASK-2510 verification set**

Run only tests related to changed Watchlists functionality/files, per user direction:

```bash
python -m pytest -q \
  Tests/Subscriptions/test_local_watchlists_service.py \
  Tests/Subscriptions/test_server_watchlists_service.py \
  Tests/Subscriptions/test_watchlist_scope_service.py \
  Tests/Watchlists/test_watchlists_backend_controller.py \
  Tests/Watchlists/test_watchlists_sources_pane.py \
  Tests/UI/test_watchlists_source_create_form.py
```

Expected: PASS.

- [ ] **Step 3: Run scoped static checks**

Run:

```bash
python -m ruff check \
  tldw_chatbook/Subscriptions/local_watchlists_service.py \
  tldw_chatbook/Subscriptions/server_watchlists_service.py \
  tldw_chatbook/Subscriptions/watchlist_scope_service.py \
  tldw_chatbook/UI/Watchlists_Modules/watchlists_backend_controller.py \
  tldw_chatbook/UI/Watchlists_Modules/sources_pane.py \
  tldw_chatbook/UI/Screens/watchlists_collections_screen.py \
  Tests/Subscriptions/test_local_watchlists_service.py \
  Tests/Subscriptions/test_server_watchlists_service.py \
  Tests/Subscriptions/test_watchlist_scope_service.py \
  Tests/Watchlists/test_watchlists_backend_controller.py \
  Tests/Watchlists/test_watchlists_sources_pane.py \
  Tests/UI/full_app_destination_context.py \
  Tests/UI/test_watchlists_source_create_form.py
git diff --check
```

Expected: both commands exit 0.

- [ ] **Step 4: Self-review the diff against every acceptance criterion**

Check specifically:

- filter options did not change;
- Local programmatic/import types remain accepted;
- Server payload contains no Local-only keys;
- invalid-type copy cannot be triggered for unrelated exceptions;
- backend changes do not replace the mounted Sources pane;
- the captured backend governs service routing, filing, and confirmation;
- visible-backend refresh semantics remain intact;
- no unrelated files or refactors entered the diff.

- [ ] **Step 5: Update the Backlog task**

Check all six acceptance criteria. Add `## Implementation Notes` containing the approach, backend-capture decision, changed files, focused test commands/results, static-check results, design-spec link, and:

```text
ADR required: no
ADR path: N/A
Reason: bounded correction inside the existing Watchlists routing boundary.
```

If this implementation reveals a reusable testing or runtime trap, add the incident to the relevant `backlog/docs/lessons-*.md`; otherwise explicitly omit a lesson rather than inventing one. Set TASK-2510 to Done only after every Definition-of-Done item is satisfied.

- [ ] **Step 6: Commit the verification/task closeout**

```bash
git add \
  "backlog/tasks/task-2510 - Source-type-options-offer-values-the-local-service-rejects.md"
git commit -m "test: verify backend-aware watchlist source creation"
```

If CSS was proven necessary, include the source and regenerated bundle in this commit. Do not commit unrelated working-tree changes.
