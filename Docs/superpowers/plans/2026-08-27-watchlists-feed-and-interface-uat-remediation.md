# Watchlists Feed and Interface UAT Remediation Implementation Plan

> Execution: use `superpowers:test-driven-development` for each backlog task
> and `superpowers:verification-before-completion` before any completion claim.

**Goal:** Turn feed failures into stable recovery receipts and make direct
Watchlists authoring, source selection, automation, and briefing recovery
usable at both first-time and power-user terminal sizes.

**Architecture:** A pure failure classifier converts transport, policy, HTTP,
and parse failures into bounded domain outcomes before persistence. Existing
run rows remain the durable receipt; their `stats_json` carries machine fields
without a new schema. The Sources pane gets one ID-based selection model and a
bulk authoring surface that uses TASK-22862's transactional command seam. The
Artifacts pane becomes stale-while-refresh: last-good data remains visible
while explicit loading, error, retry, and storage-mismatch state is layered on
top.

**Tech stack:** Python 3.11+, httpx/feedparser exception classification,
SQLite JSON payloads, Textual 8.x, bundled TCSS, pytest/pytest-asyncio.

**Backlog tasks:** TASK-22865, then TASK-22866 after TASK-22862 through
TASK-22864 are complete.

**ADR required:** no

**ADR path:** N/A

**Reason:** TASK-22865 uses existing run-receipt storage and egress policy.
TASK-22866 is workflow/UI remediation over domain boundaries already decided by
ADR-019 and the approved ADR-032 addendum; it does not introduce a new storage,
security, or ownership contract.

## TASK-22865 — Classify feed failures and recovery

### Files

- Create: `tldw_chatbook/Subscriptions/watchlist_failure.py`
- Create: `Tests/Subscriptions/test_watchlist_failure.py`
- Modify: `tldw_chatbook/Subscriptions/monitoring_engine.py`
- Modify: `tldw_chatbook/Subscriptions/local_watchlists_service.py`
- Modify: `tldw_chatbook/Subscriptions/watchlist_normalizers.py`
- Modify: `Tests/Subscriptions/test_local_watchlists_service.py`
- Modify: `Tests/Subscriptions/test_watchlist_content_kind_producer.py`
- Modify: `Tests/Subscriptions/test_subscription_egress_wiring.py`
- Modify: `Tests/UI/test_watchlists_check_now_failure.py`
- Modify: `Tests/Tools/test_watchlists_tool_service.py`
- Modify: `Docs/User_Guide/watchlists.md`

### Step 1: Pin a pure, safe failure vocabulary

Write table-driven RED tests for this exact public shape:

```python
class WatchlistFailureCategory(StrEnum):
    ACCESS_DENIED = "access_denied"
    AUTHENTICATION_REQUIRED = "authentication_required"
    RATE_LIMITED = "rate_limited"
    INVALID_FEED = "invalid_feed"
    CONNECTION_FAILURE = "connection_failure"
    TEMPORARY_SERVER_ERROR = "temporary_server_error"
    POLICY_BLOCKED = "policy_blocked"

@dataclass(frozen=True)
class WatchlistFailure:
    category: WatchlistFailureCategory
    message: str
    retryable: bool
    http_status: int | None
    retry_after_seconds: int | None
    next_action: str

def classify_watchlist_failure(error: BaseException) -> WatchlistFailure:
    """Map one internal failure to a bounded user-safe domain outcome."""
```

Pin HTTP 401 as authentication required, 403 as access denied, 429 as rate
limited, 500/502/503/504 as temporary server error, timeout/connect/DNS as
connection failure, malformed/non-feed payload as invalid feed, and the
existing SSRF/egress-policy exceptions as policy blocked. Unknown exceptions
map to a fixed connection-failure fallback; they never pass through `str(exc)`.

Reject unsafe `Retry-After` values. Accept only an integer delay within the
existing bounded retry policy; do not persist an HTTP-date, response body, URL,
headers, certificate detail, filesystem path, or exception representation.

Run:

```bash
pytest -q Tests/Subscriptions/test_watchlist_failure.py
```

Expected RED: no shared classifier or stable machine vocabulary exists.

### Step 2: Persist classified run outcomes, not raw failures

Add service/monitoring tests that drive each category through a real temporary
`SubscriptionsDB`. Pin the failure payload in `local_watchlist_runs.stats_json`:

```json
{
  "failure_category": "access_denied",
  "retryable": false,
  "http_status": 403,
  "retry_after_seconds": null,
  "next_action": "Check whether this source permits automated access."
}
```

Keep `error_msg`, source `last_error`, and logs to a fixed bounded safe message
derived from the classifier. Route both scheduled checks and Check Now through
the same formatter. Preserve the existing run/item counters, pause/backoff
logic, redirect validation, cross-origin credential stripping, and custom
header behavior.

Extend `watchlist_normalizers` so Runs, status cells, and the operation-status
tool can read the machine fields without parsing presentation copy. Legacy run
rows without those fields keep the existing generic failed state.

Run:

```bash
pytest -q Tests/Subscriptions/test_watchlist_failure.py Tests/Subscriptions/test_local_watchlists_service.py Tests/UI/test_watchlists_check_now_failure.py -k "failure or denied or auth or rate or policy or invalid"
```

### Step 3: Retain the product User-Agent as an end-to-end contract

Add a local `httpx.MockTransport` fixture that returns 403 unless the request
contains exactly the shipped product identity prefix
`tldw-chatbook/1.0 (+https://github.com/tldw/chatbook)`, then returns a minimal
valid RSS document. Drive it through the feed, URL-family, and API request
builders; do not call a live site and do not introduce hostname-specific code.

Add redirects and safe custom-override assertions so the test proves the
header survives the normal egress path without weakening the existing auth or
SSRF policy.

Run:

```bash
pytest -q Tests/Subscriptions/test_watchlist_content_kind_producer.py Tests/Subscriptions/test_subscription_egress_wiring.py -k "user_agent or product_identity or redirect"
```

### Step 4: Render actionable, non-sensitive recovery

Update Check Now failure assertions and the Watchlists guide so a user sees the
category's short message and supported next action. Network/retry categories
may offer Retry; policy blocked and invalid feed must not imply that retrying
unchanged input will help. Never make raw response or exception text available
through Runs, toasts, Inspector, Console receipt tools, or logs at ordinary
verbosity.

Run:

```bash
pytest -q Tests/UI/test_watchlists_check_now_failure.py Tests/Tools/test_watchlists_tool_service.py -k "failure or retry or redact or operation"
```

### Step 5: Verify and commit TASK-22865

```bash
pytest -q Tests/Subscriptions/test_watchlist_failure.py Tests/Subscriptions/test_local_watchlists_service.py Tests/Subscriptions/test_watchlist_content_kind_producer.py Tests/Subscriptions/test_subscription_egress_wiring.py Tests/UI/test_watchlists_check_now_failure.py Tests/Tools/test_watchlists_tool_service.py
ruff check tldw_chatbook/Subscriptions/watchlist_failure.py tldw_chatbook/Subscriptions/monitoring_engine.py tldw_chatbook/Subscriptions/local_watchlists_service.py tldw_chatbook/Subscriptions/watchlist_normalizers.py
git diff --check
```

Commit boundary:

```bash
git add tldw_chatbook/Subscriptions/watchlist_failure.py tldw_chatbook/Subscriptions/monitoring_engine.py tldw_chatbook/Subscriptions/local_watchlists_service.py tldw_chatbook/Subscriptions/watchlist_normalizers.py Tests/Subscriptions/test_watchlist_failure.py Tests/Subscriptions/test_local_watchlists_service.py Tests/Subscriptions/test_watchlist_content_kind_producer.py Tests/Subscriptions/test_subscription_egress_wiring.py Tests/UI/test_watchlists_check_now_failure.py Tests/Tools/test_watchlists_tool_service.py Docs/User_Guide/watchlists.md backlog/tasks/task-22865\ -\ Classify-Watchlists-feed-failures-and-recovery.md
git commit -m "fix: classify Watchlists feed failures"
```

## TASK-22866 — Bulk authoring and Artifacts workflow UX

### Files

- Create: `tldw_chatbook/UI/Watchlists_Modules/bulk_sources_modal.py`
- Create: `Tests/Watchlists/test_watchlists_bulk_source_authoring.py`
- Create: `Tests/Watchlists/test_watchlists_artifacts_refresh_states.py`
- Modify: `tldw_chatbook/UI/Watchlists_Modules/sources_pane.py`
- Modify: `tldw_chatbook/UI/Watchlists_Modules/table_selection.py`
- Modify: `tldw_chatbook/UI/Watchlists_Modules/artifacts_pane.py`
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py`
- Modify: `tldw_chatbook/css/features/_watchlists.tcss`
- Modify: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Modify: `Tests/Watchlists/test_watchlists_sources_pane.py`
- Modify: `Tests/Watchlists/test_watchlists_artifacts_pane.py`
- Modify: `Tests/Watchlists/test_watchlists_collections_screen.py`
- Modify: `Tests/Watchlists/test_watchlists_responsive_layout.py`
- Modify: `Tests/UI/test_watchlists_create_form_destination.py`
- Modify: `Docs/User_Guide/watchlists.md`

### Step 1: Define ID-based multi-selection and filter semantics

Add pure RED tests in `table_selection.py` before changing widgets. The state
stores canonical source IDs, an anchor ID, and the current ordered visible ID
sequence; it never stores row indexes. Pin these semantics:

- Space toggles the highlighted source and sets the range anchor.
- Shift+Up/Shift+Down extends or contracts a contiguous range in the current
  visible order.
- `v` selects every currently visible filtered row; pressing it again clears
  the visible rows while leaving hidden selections unchanged.
- `x` clears all selected IDs, including hidden ones.
- Changing filters keeps hidden selections and reports them as
  “N selected · M hidden by filters.”
- Removing/reloading a source prunes only IDs that no longer exist.

Handle the source-table keys on the focused pane/table and stop propagation.
Using `v` avoids the screen's existing global `a` “Mark all read” binding. Do
not add any terminal-convention control-key binding prohibited by `AGENTS.md`.

Run:

```bash
pytest -q Tests/Watchlists/test_watchlists_sources_pane.py -k "multi or range or filtered or selection"
```

### Step 2: Add one bulk source-entry workflow

Build `BulkSourcesModal` with a persistent “One URL per line” label, a
multiline `TextArea`, source type, optional tags/destination, an explicit
Validate/Create action, Cancel/Escape, and a row result table. Preserve the
draft after validation or write failure. Parse no more than 50 nonblank lines.
The modal posts a typed request to `WatchlistsCollectionsScreen`; the screen
calls TASK-22862's `LocalWatchlistsService.create_sources_exact_batch()`
domain seam. The modal must not import the Console tool adapter, duplicate URL
validation, or call the DB directly.

When results contain invalid/existing rows alongside created rows, display all
row outcomes in input order and present exactly two supported decisions:
“Continue with successful sources” and “Return to draft.” The first posts the
canonical successful IDs to the screen; it does not create or mutate a
Watchlist until the user explicitly chooses the follow-on collection action.

Keep the existing single-source form. Rename its primary action “New source”
and add a peer “Add several…” action so first-time and repeat flows are both
clear.

Run:

```bash
pytest -q Tests/Watchlists/test_watchlists_bulk_source_authoring.py Tests/UI/test_watchlists_create_form_destination.py
```

### Step 3: Create a Watchlist from selected sources

At the 160-column floor, keep Search, New source, Add several…, and one
“Filters…” disclosure in the top strip. Move Type, Status, Active, and Tags
into that existing disclosure with persistent visible labels and retained
values; remove tooltip-only identification from the primary path.

Extend `SourcesPane` with visible selection markers in the Name column, a
persistent count/status line, and a “Create Watchlist from selected…” action
enabled only when at least one valid ID is selected. The action collects IDs
from the selection model and routes through TASK-22862's
`WatchlistBundleService.create_with_sources()` domain seam; it must not import
the Console command adapter or loop over membership dialogs.

Make Space/range/select-visible/clear-all discoverable in the screen help and
command palette. Footer text may include only actions actually active in the
current focus/state. Pin mouse/keyboard parity, focus restoration after modal
dismissal, Escape behavior, sorted/filtered selection, and 100-member command
validation.

Run:

```bash
pytest -q Tests/Watchlists/test_watchlists_sources_pane.py Tests/Watchlists/test_watchlists_collections_screen.py Tests/Watchlists/test_watchlists_bulk_source_authoring.py -k "selected or bulk or create_watchlist or focus or escape"
```

### Step 4: Make Artifacts progressive and stale-while-refresh

Add an explicit pane state containing `idle`, `loading`, `refreshing`,
`failed`, and `storage_mismatch`, plus bounded safe failure/recovery text. On an
empty collection, render Generate briefing and Every 24 hours schedule controls
before export/keep/cast/feed actions. Move downstream actions behind the
selected briefing's detail region or one labeled “More briefing actions”
disclosure.

On refresh/generation, keep the last-good table, selection, Markdown body, and
citations mounted. Overlay an inline non-color progress indicator. On failure,
retain that content and show Retry. If an accepted/complete durable briefing
receipt exists but a reload cannot find the row, show “Briefing saved, but this
view could not reload it” with Retry and an Artifacts/Runs inspection target;
do not replace it with an empty state.

Render TASK-22864's automation receipt with exact interval, app-open
limitation, next eligibility, last attempt, last success, queue reload state,
and category-aware recovery. Use “Every 24 hours,” never “daily.”

Run:

```bash
pytest -q Tests/Watchlists/test_watchlists_artifacts_refresh_states.py Tests/Watchlists/test_watchlists_artifacts_pane.py Tests/Watchlists/test_watchlists_artifacts_selection_in_place.py -k "empty or loading or refreshing or failure or mismatch or schedule"
```

### Step 5: Verify production geometry and generated CSS

Keep all new styles in `_watchlists.tcss`; do not add class-level
`DEFAULT_CSS`. Regenerate all checked-in bundles, then use
`ConsolidatedCSSApp` at 160x42 and the suite's normal size. Pin containment,
focus order, persistent labels, non-color selection/status markers, modal
scrolling, retained last-good content, and no clipped primary action.

Run:

```bash
python -m tldw_chatbook.css.build_css
python tldw_chatbook/css/check_bundle_sync.py
pytest -q Tests/Watchlists/test_watchlists_responsive_layout.py Tests/Watchlists/test_watchlists_bulk_source_authoring.py Tests/Watchlists/test_watchlists_artifacts_refresh_states.py
```

### Step 6: Verify and commit TASK-22866

```bash
pytest -q Tests/Watchlists/test_watchlists_sources_pane.py Tests/Watchlists/test_watchlists_bulk_source_authoring.py Tests/Watchlists/test_watchlists_artifacts_pane.py Tests/Watchlists/test_watchlists_artifacts_refresh_states.py Tests/Watchlists/test_watchlists_collections_screen.py Tests/Watchlists/test_watchlists_responsive_layout.py Tests/UI/test_watchlists_create_form_destination.py
ruff check tldw_chatbook/UI/Watchlists_Modules/bulk_sources_modal.py tldw_chatbook/UI/Watchlists_Modules/sources_pane.py tldw_chatbook/UI/Watchlists_Modules/table_selection.py tldw_chatbook/UI/Watchlists_Modules/artifacts_pane.py tldw_chatbook/UI/Screens/watchlists_collections_screen.py
python tldw_chatbook/css/check_bundle_sync.py
git diff --check
```

Commit boundary:

```bash
git add tldw_chatbook/UI/Watchlists_Modules/bulk_sources_modal.py tldw_chatbook/UI/Watchlists_Modules/sources_pane.py tldw_chatbook/UI/Watchlists_Modules/table_selection.py tldw_chatbook/UI/Watchlists_Modules/artifacts_pane.py tldw_chatbook/UI/Screens/watchlists_collections_screen.py tldw_chatbook/css/features/_watchlists.tcss tldw_chatbook/css/tldw_cli_modular.tcss Tests/Watchlists/test_watchlists_bulk_source_authoring.py Tests/Watchlists/test_watchlists_artifacts_refresh_states.py Tests/Watchlists/test_watchlists_sources_pane.py Tests/Watchlists/test_watchlists_artifacts_pane.py Tests/Watchlists/test_watchlists_collections_screen.py Tests/Watchlists/test_watchlists_responsive_layout.py Tests/UI/test_watchlists_create_form_destination.py Docs/User_Guide/watchlists.md backlog/tasks/task-22866\ -\ Remediate-Watchlists-bulk-authoring-and-Artifacts-workflow-UX.md
git commit -m "feat: streamline Watchlists authoring and Artifacts recovery"
```

## Plan-level self-review gate

- Failure categories are stable machine values; presentation copy is not an
  API and raw exceptions never escape.
- The User-Agent test exercises the generic transport path and adds no site
  special case.
- Multi-selection is keyed by canonical source ID and remains truthful across
  filtering, sorting, reload, and deletion.
- Partial source creation never silently changes collection membership.
- Artifacts never erase last-good content merely because a refresh starts or
  fails.
- Every shortcut is implemented, focus-scoped, non-conflicting, and visible in
  current-state help.
- 160x42 is tested with the production CSS bundle, not an unstyled harness.
