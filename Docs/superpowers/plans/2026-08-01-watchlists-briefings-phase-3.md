# Watchlists Briefings Phase 3 — Markdown Export and the Podcast Feed Directory

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A briefing exports as markdown to a file the user picks, and a watchlist's audio episodes export as a self-contained podcast feed directory (`feed.xml` + the audio files) to a folder the user picks.

**Architecture:** Two independent user-initiated exports, both leaving the private-storage boundary **by intent**. A pure RSS builder (no I/O) sits behind a directory-writing service that copies audio out of the private dir and writes `feed.xml` atomically. UI is dispatch-only workers on the existing Artifacts toolbars — no new toolbar rows, because the pane's height budget is already pinned by two geometry tests.

**Tech Stack:** Python 3.11, Textual, SQLite, stdlib `xml.etree.ElementTree`, pytest (`.venv/bin/python -m pytest`, plain output).

## Global Constraints

- pytest is the ONLY python entry point for this repo's code. Never bare `python -c` importing `tldw_chatbook` (it loads the user's live config).
- Never `git stash`; never `git checkout --`/`git restore` to revert (Edit-tool reverts only); never any `git worktree` command; never `-q` with pytest.
- Never hand-edit `tldw_chatbook/css/tldw_cli_modular.tcss` — edit `css/features/_watchlists.tcss` and regenerate via `cd tldw_chatbook/css && ../../.venv/bin/python build_css.py`.
- All screen DB calls via `asyncio.to_thread`. Workers: `group=` always set; guard flags claimed at dispatch, cleared in `finally`.
- Toasts interpolating a value use `_notify_watchlists(..., markup=False)`. Any user-chosen path or provider text rendered in a widget is wrapped in `rich.text.Text` — never `escape_markup`.
- Exception logging is type-only (`type(exc).__name__`); never `logger.opt(exception=True)`; never log briefing bodies or file contents.
- No new `persist_event` event names (ADR-029 admits exactly six).
- `--strict-markers` is on: an unregistered marker is a **collection error**. Registered: `unit`, `integration`, `slow`, `requires_cleanup`, `asyncio`, `ui`. `Tests/Watchlists/` uses `pytestmark = pytest.mark.ui`.
- Every behavioural change gets a revert-confirm-RED-restore mutation check (Edit-tool reverts only).
- Spec: `Docs/superpowers/specs/2026-07-30-watchlists-briefings-design.md` §"Exports and feed (phase 3)".

## Decisions locked before implementation (do not relitigate; DO pin with tests)

1. **No localhost serving.** Owner decision, 2026-08-01, after a seam survey found the spec's premise false: `[web_server]` is textual-serve — a *mutually exclusive process mode* that serves the TUI itself to a browser (`app.py:9231` returns instead of running the TUI), its only static route is hardcoded to textual-serve's own assets (`Web_Server/serve.py:165`), and **`web_server.enabled` is read by no code at all**. There is no server alongside the TUI to toggle. Phase 3 ships the directory; serving is documented for the user's own static server. The spec's own framing already says "the directory is the deliverable."
2. **Never route a user-chosen destination through `private_paths`.** `secure_private_directory(..., application_owned=True)` chmods the target to `0o700` and raises `ValueError` when not application-owned; `create_private_text` opens `"xb"` and so refuses to overwrite. Applying either to the user's folder would chmod their directory and fail on re-export. User destinations use `validate_path_simple` + plain stdlib writes — the split the repo already makes (`library_screen.py:6486` and `STTS_Window.py:4040` are user-path; `Tools_Settings_Window.py:6313` is app-owned).
3. **`audio_file_path_is_safe` runs BEFORE any filesystem access on a stored path.** Established by phase 2b's Qodo round for playback (`artifacts_pane.py:359,381` — the order is load-bearing and commented as such). Export copies files *out*, so it must honour the same order or it becomes a wider hole than playback.
4. **RSS is written from scratch with stdlib `xml.etree.ElementTree`.** Nothing in the repo generates RSS (`rss_feed_generator.py` was deliberately deleted in `e74e37d07`, and the spec's Non-goals forbid reviving it; `feedparser` is declared but imported nowhere). The writing precedent to copy is `Web_Scraping/Article_Extractor_Lib.py:1141-1152` (`SubElement` + `tostring`). **Do not add `feedgen`** — a new dependency would inherit phase 2b's three-job CI edit for no benefit.

## File Structure

- `tldw_chatbook/DB/Subscriptions_DB.py` — one joined accessor for a watchlist's audio episodes (Task 2)
- `tldw_chatbook/Subscriptions/briefing_export.py` — NEW: markdown export helper + the feed-directory writer (Tasks 1, 5)
- `tldw_chatbook/Subscriptions/briefing_feed.py` — NEW: the pure RSS builder, no I/O (Task 3)
- `tldw_chatbook/UI/Watchlists_Modules/artifacts_pane.py` + `UI/Screens/watchlists_collections_screen.py` — buttons, messages, workers (Tasks 1, 6)
- Tests: `Tests/Subscriptions/test_briefing_export_markdown.py`, `test_briefing_feed.py`, `test_briefing_feed_export.py`; extensions to `Tests/Watchlists/test_watchlists_artifacts_pane.py`

Where a step shows a contract instead of a full body, the full body is mandatory and the named precedent must be read first (the phase-1/2a/2b convention).

---

### Task 1: Markdown export of a briefing

**Files:**
- Create: `tldw_chatbook/Subscriptions/briefing_export.py` (markdown half)
- Modify: `tldw_chatbook/UI/Watchlists_Modules/artifacts_pane.py` (`#artifacts-toolbar` at `:667`; `on_button_pressed` at `:1156`; new message beside the others at `:102-225`), `UI/Screens/watchlists_collections_screen.py` (import block `:75-90`, new handler)
- Test: `Tests/Subscriptions/test_briefing_export_markdown.py` (new), extend `Tests/Watchlists/test_watchlists_artifacts_pane.py`

**Interfaces — Produces:**
```python
# briefing_export.py
class BriefingExportError(RuntimeError): ...
def safe_export_stem(text: str, *, fallback: str) -> str      # alnum/space/-/_ only, stripped
def briefing_markdown_document(briefing: Mapping[str, Any]) -> str
def default_briefing_filename(briefing: Mapping[str, Any], *, watchlist_name: str) -> str  # "<stem>.md"
# artifacts_pane.py
class ExportBriefingRequested(Message): ...        # carries nothing; screen reads selected_briefing
```

- [ ] **Step 1: Read the precedent end to end.** `UI/Screens/library_screen.py:6428` (the `FileSave` push) and `_write_library_note_export_file:6444` (validate → write → honest toasts, plus an explicit "cancelled" toast on `None`). Import the **vendored** picker: `from ...Third_Party.textual_fspicker import FileSave` — its kwarg is `default_file`, **not** `default_filename` (that is the *enhanced* picker's, a different class). Do NOT copy `UI/Voice_Cloning_Window.py:594`, which passes a `show_dirs_only` kwarg no picker class accepts and would raise `TypeError`.

- [ ] **Step 2: Write the failing tests.** Service (`pytest.mark.unit`): `briefing_markdown_document` includes the body verbatim plus a front-matter header carrying watchlist name, status, coverage window and created time; a briefing whose `body_markdown` is NULL/empty raises `BriefingExportError` naming the briefing (an empty file is not an export); `safe_export_stem` strips path separators, `..`, and markup-shaped text, and falls back when nothing survives; `default_briefing_filename` ends in `.md` and never contains a separator. UI (`pytestmark = pytest.mark.ui`): the Export button renders in `#artifacts-toolbar` with `compact=True` and is **disabled when no briefing is selected or the selected briefing is not `complete`**; pressing it posts `ExportBriefingRequested`; the screen's handler pushes a `FileSave` (assert on a patched `push_screen`) seeded with the default filename; the callback writes the file and toasts success with `markup=False`; a `None` callback (user cancelled) writes nothing and toasts a cancellation; a write raising `OSError` toasts `type(exc).__name__` and writes nothing further.

- [ ] **Step 3: Run, confirm RED.**

- [ ] **Step 4: Implement.** Handler validates with `validate_path_simple(path, require_exists=False)` inside `try/except ValueError` → warn-and-return, then `validated.write_text(document, encoding="utf-8")` off the event loop via `asyncio.to_thread`. Button goes in the EXISTING `#artifacts-toolbar` — adding a new `Horizontal` costs a row the pane cannot spare (see Task 6's geometry note).

- [ ] **Step 5: Green + mutations.** (a) Drop the `validate_path_simple` call → the traversal test REDs; restore. (b) Let an empty body export → the empty-body test REDs; restore.

- [ ] **Step 6: Commit** `feat(briefings): export a briefing as markdown`.

---

### Task 2: One query for a watchlist's audio episodes

**Files:**
- Modify: `tldw_chatbook/DB/Subscriptions_DB.py` (beside `list_briefing_audio` at `:2241`)
- Test: `Tests/Subscriptions/test_briefing_feed_query.py` (new)

**Interfaces — Produces:**
```python
def list_watchlist_audio_episodes(self, watchlist_id: int, *, limit: int = 500,
                                  offset: int = 0) -> List[Dict[str, Any]]
    # One joined SELECT across briefing_audio -> briefing_scripts -> briefings, filtered to
    # audio.status='complete' AND audio.file_path IS NOT NULL, ordered briefings.created_at DESC,
    # audio.id DESC. Each row carries: audio_id, script_id, briefing_id, file_path,
    # duration_seconds, turn_count, preset_name, briefing_created_at, briefing_status,
    # covers_from_ts, model_used.
```

- [ ] **Step 1: Read the precedent.** `list_briefing_audio:2241` for the transaction/pagination shape; every DB op — reads included — goes inside `with self.transaction() as conn:` (Qodo rule 1011851), and collection queries take `limit`/`offset` as SQL `LIMIT ?/OFFSET ?` (a Qodo finding on 2a).

- [ ] **Step 2: Write the failing tests** (`pytest.mark.unit`, real `SubscriptionsDB` on `tmp_path`): a watchlist with two briefings, each with a script, each with one `complete` audio row → both returned, newest briefing first; a `failed` audio row is excluded; a `complete` row with NULL `file_path` is excluded; audio belonging to a *different* watchlist is excluded (the scoping test — assert by identity, not count); pagination walks without gaps or repeats and `limit` is honoured (spy the bound parameters, the shape used in `test_briefing_audio_db.py`); an empty watchlist returns `[]`.

- [ ] **Step 3: Run, confirm RED.**

- [ ] **Step 4: Implement** as a single `SELECT ... JOIN briefing_scripts ON ... JOIN briefings ON ...`. Column aliases must match the Interfaces block exactly — Tasks 3 and 5 quote those names.

- [ ] **Step 5: Green + mutation.** Drop the `status='complete'` predicate → the failed-row-excluded test REDs; restore. Drop the watchlist scoping → the cross-watchlist test REDs; restore.

- [ ] **Step 6: Commit** `feat(briefings): one query for a watchlist's audio episodes`.

---

### Task 3: The RSS feed builder (pure, no I/O)

**Files:**
- Create: `tldw_chatbook/Subscriptions/briefing_feed.py`
- Test: `Tests/Subscriptions/test_briefing_feed.py` (new)

**Interfaces — Consumes:** Task 2's row shape.
**Interfaces — Produces:**
```python
class FeedBuildError(RuntimeError): ...
@dataclass(frozen=True)
class FeedEpisode:
    title: str; filename: str; length_bytes: int; duration_seconds: float | None
    published: datetime; guid: str; description: str
def build_feed_xml(*, channel_title: str, channel_description: str,
                   episodes: Sequence[FeedEpisode], now: datetime) -> bytes
```

- [ ] **Step 1: Read the XML-writing precedent** — `Web_Scraping/Article_Extractor_Lib.py:1141-1152` (`ET.SubElement` + `ET.tostring(root, "utf-8")`). `defusedxml` is a *parsing* hardener and adds nothing here; do not import it for generation. **Do not add `feedgen`.**

- [ ] **Step 2: Write the failing tests** (`pytest.mark.unit`) — parse every result with `xml.etree.ElementTree` and assert on the tree, never on a substring of the raw bytes (a substring test passes on malformed XML): the document is well-formed and has `rss[@version='2.0']` with one `channel`; channel carries title, description and `lastBuildDate`; each episode yields an `item` with `title`, `guid`, `pubDate` in RFC-822, and an `enclosure` whose `url` is the **relative filename** (the directory is self-contained — an absolute local path would leak the user's home directory into a file they may share), `length` in bytes and `type="audio/wav"`; `itunes:duration` is emitted when `duration_seconds` is present and omitted when None; a title containing `<`, `&` and `]]>` round-trips exactly through parse (escaping is ElementTree's job — assert it, do not hand-escape); an empty `episodes` sequence still produces a valid channel with zero items; a filename containing a path separator raises `FeedBuildError` naming it (a feed must never reference outside its own directory).

- [ ] **Step 3: Run, confirm RED.**

- [ ] **Step 4: Implement.** `now` is injected, never `datetime.now()` inside — the tests must be deterministic and the repo forbids ambient clocks in pure builders.

- [ ] **Step 5: Green + mutations.** (a) Emit an absolute path in `enclosure/@url` → the relative-url test REDs; restore. (b) Drop the separator guard → that test REDs; restore.

- [ ] **Step 6: Commit** `feat(briefings): build a podcast RSS feed`.

---

### Task 4: Writing the feed directory

**Files:**
- Modify: `tldw_chatbook/Subscriptions/briefing_export.py` (feed half)
- Test: `Tests/Subscriptions/test_briefing_feed_export.py` (new)

**Interfaces — Consumes:** Task 2's accessor; Task 3's `build_feed_xml`/`FeedEpisode`; `artifacts_pane.audio_file_path_is_safe` (`:359`) — **import it, do not re-derive the check**.
**Interfaces — Produces:**
```python
@dataclass(frozen=True)
class FeedExportResult:
    directory: Path; episode_count: int; skipped: list[str]   # skipped names the reason per episode
def export_feed_directory(db, watchlist_id: int, *, destination: Path,
                          watchlist_name: str, now: datetime) -> FeedExportResult
```

- [ ] **Step 1: Read three things.** (a) The copy-out recipe — `UI/STTS_Window.py:4040-4049`: `validate_path_simple(dest, require_exists=False)` → `validate_path_simple(dest.parent, require_exists=True).resolve()` → `validate_filename(dest.name)` → `shutil.copy2`. (b) The atomic-write discipline — `Chatbooks/chatbook_creator.py:1506` `_create_zip_archive`: `os.open(partial, O_RDWR|O_CREAT|O_EXCL|O_NOFOLLOW, 0o600)` → write → `flush()` + `os.fsync()` → `os.replace(partial, final)`, with a `finally` that closes the fd and unlinks the partial on failure. (c) Decision 2 above — **never** call `secure_private_directory` on the destination.

- [ ] **Step 2: Write the failing tests** (`pytest.mark.unit`, `get_user_data_dir` patched to `tmp_path` in every test that touches storage — this repo has had three live-data incidents): a happy path writes `feed.xml` plus one file per episode and returns the count; **an episode whose `file_path` fails `audio_file_path_is_safe` is skipped with a reason and its file is never opened** (patch the safety helper to False and assert no read occurred — this is the load-bearing security test); an episode whose source file has vanished is skipped with a reason, and the rest still export; `feed.xml` is written atomically (assert no `.partial` remains, and that a failure mid-write leaves the previous `feed.xml` intact); re-exporting to the same directory overwrites cleanly (no `"xb"`-style refusal — the phase-2b private-write helpers must not be used here); the destination is never chmodded to `0o700` (assert the mode survives — this pins Decision 2); a destination that fails `validate_path_simple` raises before anything is written.

- [ ] **Step 3: Run, confirm RED.**

- [ ] **Step 4: Implement.** Episode filenames come from `safe_export_stem` + the audio id, so two briefings with the same title cannot collide. `length_bytes` is the real file size read after the copy. Never raise for one bad episode — collect into `skipped` so a partial export is honest rather than all-or-nothing.

- [ ] **Step 5: Green + mutations.** (a) Call the filesystem before the safety check → the load-bearing test REDs; restore. (b) Replace the atomic write with a plain `write_bytes` → the no-`.partial`/intact-previous test REDs; restore.

- [ ] **Step 6: Commit** `feat(briefings): write a self-contained podcast feed directory`.

---

### Task 5: The feed export action

**Files:**
- Modify: `tldw_chatbook/UI/Watchlists_Modules/artifacts_pane.py` (`#artifacts-audio-toolbar` at `:870`, `on_button_pressed` at `:1156`, new message), `UI/Screens/watchlists_collections_screen.py`
- Test: extend `Tests/Watchlists/test_watchlists_artifacts_pane.py`

**Interfaces — Consumes:** Task 4's `export_feed_directory`; `SelectDirectory` from `...Third_Party.textual_fspicker` (dismisses with `Path | None` — precedent `library_screen.py:11740`).
**Interfaces — Produces:** `class ExportFeedRequested(Message)`.

- [ ] **Step 1: Read the pane's two hard constraints.** (a) **Every reactive on this pane is `recompose=True`** (module docstring `:25-33`) — export-in-flight state must NOT live on the pane; keep the guard flag on the screen, claimed at dispatch, cleared in `finally`, `group="wl-feed-export"`, exclusive. (b) **The height budget is pinned**: `#artifacts-table` resolves to exactly `region.height == 4` (`_watchlists.tcss:1074-1088`), and Task 7 of phase 2b recorded that adding one fixed row plus a scrollable region squeezed the scripts table to its header. **Put the button in the existing `#artifacts-audio-toolbar`** (`compact=True`); do not add a `Horizontal`.

- [ ] **Step 2: Write the failing tests:** the Export-feed button renders in `#artifacts-audio-toolbar` and is **disabled when the watchlist has no complete audio episodes** (a dead control is a spec violation — phase 2b shipped a disabled Play for exactly this reason); pressing it pushes `SelectDirectory` (patched `push_screen`); the callback runs the export off the event loop and toasts the episode count with `markup=False`; a cancelled picker writes nothing and toasts a cancellation; a partial export (one skipped episode) toasts honestly that N of M exported rather than claiming success; a second press while in flight is refused naming the running export; an `OSError` from the service toasts `type(exc).__name__` and leaves the app running (`host.is_running` — the phase-2b app-death lesson, where an unwrapped worker with `exit_on_error=True` killed the app).
  Both pinned geometry tests (`test_the_list_the_button_and_the_body_are_all_on_screen`, `test_the_briefings_table_keeps_at_least_three_usable_rows`) must stay green; if the button squeezes them, stop and report rather than loosening a pinned test.

- [ ] **Step 3-4: Implement + green.** All DB and filesystem work inside one `asyncio.to_thread` hop; `is_attached` before any UI mutation after an await.

- [ ] **Step 5: Mutations.** (a) Claim the guard inside the worker body → the double-press test REDs; restore. (b) Report success on a partial export → the honest-count test REDs; restore.

- [ ] **Step 6: Commit** `feat(briefings): export a watchlist's podcast feed directory`.

---

### Task 6: Close-out

- [ ] Full sweep: `Tests/Subscriptions/ Tests/Watchlists/ Tests/UI/ -k watchlist`. Documented baselines (NOT ours): 2 tree-chevron failures in `test_destination_visual_parity_correction.py`; the TASK-1345 create-form mount race (rotating symptom — re-run alone before classifying).
- [ ] `backlog/tasks/task-1540`: check AC #3 (phase 3), leaving phase 4 unchecked.
- [ ] **Spec corrections** in `Docs/superpowers/specs/2026-07-30-watchlists-briefings-design.md` — a "Phase 3 delivery notes (2026-08-01)" block recording: (a) **localhost serving is cut**, with the reason (`[web_server]` is textual-serve, a mutually exclusive process mode; its only static route is textual's own assets; `web_server.enabled` is read by nothing) and the owner's decision to document a user-run static server instead; (b) the feed directory is self-contained and uses **relative** enclosure URLs deliberately, so sharing the folder never leaks a home path; (c) §Testing's "every new test `pytest.mark.unit` or it is invisible to CI" is **stale** — since task-1465 CI runs `pytest Tests --ignore=Tests/UI` plus `pytest Tests/UI` with no marker selection; what actually matters now is `--strict-markers` (an unregistered marker is a collection error).
- [ ] Cross-worktree ID scan (controller supplies IDs) → file a follow-up for in-app feed serving, carrying decision 1's evidence so a future implementer does not re-discover that `[web_server]` cannot do it.
- [ ] Add a short "Exporting a podcast feed" note to the user-facing docs directory if one covers Watchlists; if none exists, say so in the report rather than inventing a location.
- [ ] Commit `docs(briefings): phase 3 close-out`.

## Self-review

**Spec coverage (phase 3 scope):** markdown export via the file picker → T1; feed directory with `feed.xml` + audio files to a user-chosen folder → T2/T3/T4/T5; "export is deliberate egress outside the private-storage boundary" → Decision 2, pinned by T4's no-chmod test; localhost serving → cut by owner decision, recorded in T6. Phase 4 (scheduling) untouched.

**Placeholder scan:** none — every step names its precedent `file:line` or carries the contract inline.

**Type consistency:** Task 2's column aliases are quoted verbatim by Tasks 3-5; `FeedEpisode` fields match `build_feed_xml`'s use and Task 4's construction; `Path` is the currency at every picker boundary (both pickers dismiss with `Path | None`); `safe_export_stem` is defined in T1 and reused in T4.
