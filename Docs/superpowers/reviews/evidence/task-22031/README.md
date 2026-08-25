# TASK-22031 real Textual verification evidence

## Outcome

The production-shaped verification matrix passed on commit
`1a3ca3ad80e90f02fb39f09c27e8aced646d114d`. No product defect reproduced.

This rig mounts the real `LibraryScreen`, exact `TldwCli.CSS_PATH` production
stylesheet sequence, retained reader widgets, workers, and Textual compositor. It
uses bounded seeded scope services so the run is deterministic and makes no network
request. The executable driver is `task22031_live_matrix.py`; every state has a
compositor SVG, plain-text frame, and structured JSON alongside it. `summary.json`
is the machine-readable rollup.

## Isolation and provenance

- Effective package import:
  `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/library-media-reader/tldw_chatbook/__init__.py`
- Final scratch root: `/private/tmp/task22031-live-1a3ca.XGZpkQ`
- Effective config: `/private/tmp/task22031-live-1a3ca.XGZpkQ/config.toml`
- Effective data directory:
  `/private/tmp/task22031-live-1a3ca.XGZpkQ/data/verify_task22031`
- `HOME`, `XDG_CONFIG_HOME`, `XDG_DATA_HOME`, `XDG_CACHE_HOME`,
  `TLDW_CONFIG_PATH`, `TLDW_TEST_MODE`, and the data directory were set before
  importing `tldw_chatbook`; model-catalog network refresh was disabled.
- The driver refuses to import the app when any isolation boundary is absent or
  when config/data/package provenance falls outside the expected roots.
- Real config SHA-256 before and after:
  `edc7a780b61ae285b5082f1dfab1381981b7d7bcd9efb2b38f38e2fc4effbbd1`
  (53,496 bytes both times).
- Real `default_user` profile tree SHA-256 before and after:
  `7d6173b83a867f7ccf8a5520dbbc04fd50cf618ded95c8ff567912fcfdf393c0`
  (147 files both times).
- Git status before and after showed only the existing untracked
  `Docs/superpowers/reviews/evidence/` path; no tracked generated CSS or product
  file changed.

Final command (absolute venv interpreter and explicit worktree import order):

```bash
env HOME=/private/tmp/task22031-live-1a3ca.XGZpkQ \
  XDG_CONFIG_HOME=/private/tmp/task22031-live-1a3ca.XGZpkQ/xdg-config \
  XDG_DATA_HOME=/private/tmp/task22031-live-1a3ca.XGZpkQ/xdg-data \
  XDG_CACHE_HOME=/private/tmp/task22031-live-1a3ca.XGZpkQ/xdg-cache \
  TLDW_CONFIG_PATH=/private/tmp/task22031-live-1a3ca.XGZpkQ/config.toml \
  TLDW_TEST_MODE=1 TLDW_DISABLE_MODEL_CATALOG_NETWORK=1 \
  TASK22031_DATA_DIR=/private/tmp/task22031-live-1a3ca.XGZpkQ/data/verify_task22031 \
  PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/library-media-reader \
  ../../.venv/bin/python \
  Docs/superpowers/reviews/evidence/task-22031/task22031_live_matrix.py
```

Exit status: `0`.

## Exact geometry matrix

All measurements are compositor-settled widget regions. Each frame focuses the
Items grip, then proves its painted hit owner is that exact grip; this includes the
compact 80x24 state rather than relying on a resting frame.

| Destination | Terminal | Library | Items | Work | Effective open panes |
|---|---:|---:|---:|---:|---|
| Conversations | 160x50 | 28 | 40 | 78 | Library + Items |
| Conversations | 120x35 | 0 | 56 | 50 | Items |
| Conversations | 100x30 | 0 | 42 | 44 | Items |
| Conversations | 80x24 | 0 | 0 | 66 | Work protected |
| Media | 160x50 | 28 | 40 | 78 | Library + Items |
| Media | 120x35 | 0 | 56 | 50 | Items |
| Media | 100x30 | 0 | 44 | 46 | Items |
| Media | 80x24 | 0 | 0 | 70 | Work protected |

Both destinations retain two five-column grips at every size. At 80x24, both
content panes are collapsed, both restore grips remain focusable and painted, and
Work remains mounted with positive width.

Representative artifacts:

- Conversations: `conversations-160x50.svg`, `conversations-120x35.svg`,
  `conversations-100x30.svg`, `conversations-80x24.svg`
- Media: `media-160x50.svg`, `media-120x35.svg`, `media-100x30.svg`,
  `media-80x24.svg`

## Collapse, comfort expansion, restore, and shortcuts

At 160x50, collapsing only Library expands the mounted Conversations Items region
from 36 to 52 columns and its first row from 35 to 51 columns. The persisted Items
width remains exactly 40; responsive/effective comfort expansion did not write back
into the preference. The painted title improves from
`Alpha planning — an intentiona…` to
`Alpha planning — an intentionally long convers…`.

After collapsing Items too, Work expands to 146 columns and remains mounted. The
two focusable controls truthfully read `Expand Library pane` and `Expand Items pane`.
Both panes restore independently.

The corrected focus rig re-queries after restore and polls event plus compositor
readiness on a monotonic deadline. It proves the global pane cycle is
Library → Items → Work → Library, Escape from Work lands on visible Items, and `/`
lands on the visible Items filter. The live destination shortcuts are `/ focus
filter`, `F6 next pane`, and `esc focus Library`; each maps to an exercised action,
while global F1/Ctrl+P/Ctrl+Q remain app chrome.

Artifacts: `conversations-160x50-expanded.svg`,
`conversations-160x50-library-collapsed.svg`,
`conversations-160x50-both-collapsed.svg`, and
`conversations-160x50-focus-footer.svg`.

## Progressive transcript and complete-only Find

The Reader paints the first bounded page at 20 messages while `complete=false`.
Find for `needle` stays incomplete and reports that the complete transcript is being
searched. Releasing the second page yields exactly 21/21 messages, two off-loop
service calls at offsets 0 and 20 with limits 20 and `max_chars=8000`, the same
stable `message_epoch=epoch-chat-a` across both accepted pages, one exact match,
and focus on stable message ID `message-20`.

Artifacts: `conversations-progressive-first-page.svg` and
`conversations-progressive-find-complete.svg`.

## Bulk preview and identity truth

Entering Select mode with zero checked rows preserves one loaded transcript message
as a read-only preview, but explicitly excludes that preview from the bulk selection.
`Open in Console` is disabled, while Read/Info remain enabled as view selectors. The
asserted live status says: `Bulk selection: 0 conversations. The retained transcript
is not included and remains read-only.`

The A→B journey proves all four identity states through reachable public paths:

| Phase | Selected | Loaded/shown | Result |
|---|---|---|---|
| Loading B | `chat-b` v7 | `chat-a` v4 / `epoch-chat-a` | Status names both identities |
| Invalid B response | `chat-b` v7 | `chat-a` v4 / `epoch-chat-a` | Error names both; row action disabled |
| Retry succeeds | `chat-b` v7 | `chat-b` v7 / `epoch-chat-b` | Complete, error-free exact match |
| Authoritative refresh deletes B | `chat-b` v7 | none / no epoch | Exact locator called; `Conversation deleted.` |

Artifacts: `conversations-bulk-readonly-preview.svg`,
`conversations-a-to-b-loading.svg`, `conversations-a-to-b-error.svg`,
`conversations-a-to-b-retry.svg`, and `conversations-b-deleted.svg`.

## Review-blocker production-shaped probes

The evidence rig also drives the two final review blockers through the real mounted
`LibraryScreen` rather than relying only on unit state:

- **Cross-destination latest intent:** a delayed Conversations `library_open=false`
  persistence write is overtaken by a Media `library_open=true` intent. Both shared
  preference snapshots and the mounted Media shell finish open, the shared authority
  generation reaches 2, and disk finishes `true` even though both writes execute.
  Artifact: `shared-library-latest-intent.svg`.
- **Select + Find/Retry bulk fence:** with progressive page 2 still in flight, Select
  mode invalidates the reader. Find and Retry are each driven independently; neither
  starts another service call, neither restores loading or loaded-action eligibility,
  and bulk mode remains active after the released stale worker settles. Both live
  statuses retain the explicit `not included and remains read-only` copy. Artifacts:
  `conversations-select-find-fenced.svg` and
  `conversations-select-retry-fenced.svg`.
- **Concurrent double failure:** durable shared Library state starts `true`;
  overlapping Conversations `false` and Media `true` writes both execute and both
  fail. The current generation rolls back to durable `true`, leaving app config,
  both preference snapshots, and the mounted Media shell open. Artifact:
  `shared-library-double-failure.svg`.
- **Stale skip + newer failure:** the shared persistence lock holds both workers until
  the newer Media open is claimed. The stale Conversations close is skipped without a
  save; the only attempted newer `true` save fails and rolls back to durable `true`
  everywhere. Artifact: `shared-library-stale-skip-newer-failure.svg`.
- **Settings-refresh stale-write repair:** for Media Library, Conversations Library,
  Media Items, and Conversations Items, a stale `false` save is already executing
  when a newer Settings refresh supplies `true`. Every case observes exactly
  `[false, true]`; physical disk, durable authority, destination preference, app
  config, and the mounted shell all finish `true`. Both shared-Library cases also
  prove both Media and Conversations preference snapshots finish `true`. Artifacts:
  `settings-repair-media-library.svg`,
  `settings-repair-conversations-library.svg`,
  `settings-repair-media-items.svg`, and
  `settings-repair-conversations-items.svg`.
- **Failed repair truth:** when the stale `false` write physically succeeds but its
  repairing `true` write fails, disk is physically `false`; durable authority, app
  config, both shared preference snapshots, and the mounted Media shell all reconcile
  to `false`, with a warning notification. Artifact:
  `settings-repair-failure-truth.svg`.
- **Delayed Settings callback:** the missing production schedule is exercised for
  Media Library, Conversations Library, Media Items, and Conversations Items:
  Settings `true` commits first, the already-started stale grip `false` exits second,
  and only then does the Settings refresh callback run. Every authority observes
  exactly `[false, true]`; disk, durable state, preference snapshot, app config, and
  mounted shell finish `true`. Shared Library additionally proves both destination
  snapshots finish `true`. Artifacts are the four
  `delayed-settings-repair-<destination>-<pane>.svg` frames.
- **Delayed failed repair + exact TOML truth:** if the delayed repairing `true` write
  fails after stale `false` reached disk, the authoritative serialized TOML is
  `[library.reader] library_open = false`. Durable state, app config, both shared
  snapshots, and mounted Media all project that physical `false`, with a warning.
  Artifact: `delayed-settings-repair-failure-truth.svg`.
- **Persisted config semantics:** direct production-boundary reads record exact TOML
  facts for canonical-over-legacy precedence (`true`), real legacy shared Library
  fallback (`false`), and absent Items default (`true`) in `summary.json`.

## Evidence integrity

The refreshed candidate contains 105 files totaling 5,221,422 bytes. All JSON
artifacts and the isolated TOML config parse, the driver parses and passes Ruff,
`git diff --check` passes, and the summary records the exact four terminal sizes for
both Conversations and Media. Import provenance points at this worktree, and the
pre/post real config and profile fingerprints above are identical.

## Honest limits

- This is the real mounted Library surface and compositor with production CSS, but
  it is a deterministic headless Textual host, not a tmux-driven full `TldwCli`
  process. App chrome is rendered by the host; external terminal mouse protocol is
  outside this task's keyboard/collapse contract.
- Data authority is seeded local scope services; no personal Library data, remote
  server, or provider credentials are used. External-server Media identity is
  therefore covered by the automated regression suite, not this live matrix.
- Startup emits pre-existing optional-backend diagnostic logs for unavailable Study
  and Prompts stores in the lightweight test app. They do not affect the mounted
  Library destinations, and every asserted state settled before capture.
