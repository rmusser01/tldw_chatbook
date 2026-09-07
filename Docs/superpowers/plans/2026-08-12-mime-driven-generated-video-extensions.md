# MIME-driven Generated-video Extensions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Store and resolve generated MP4 and WebM files under the one canonical extension proven by the requested format, adapter-observed container, and response MIME.

**Architecture:** Add one immutable MP4/WebM vocabulary at the `Video_Generation` boundary, require every adapter result to carry an observed canonical container, and validate request/result agreement in `worker.run_generation` before Console can persist or stage bytes. Migrate every production caller to explicit metadata-derived extensions before removing `VideoStore`'s MP4 defaults, then prove both database reload and Textual screen-state remount.

**Tech Stack:** Python 3.11+, frozen dataclasses, Textual 8.x, SQLite-backed Console persistence, pathlib/filesystem transactions, pytest, Ruff.

---

## Scope and sequencing

The API-breaking changes are sequenced so every commit remains runnable:

1. The result contract, both current result-producing adapters, worker validation, and the outer no-persistence regression change together.
2. Metadata and Console write/pending paths become container-aware while `VideoStore` temporarily retains MP4 defaults for untouched readers.
3. Every production read/action/reload/remount path becomes explicit.
4. Only after all production callers migrate do `VideoStore` defaults disappear; every direct focused test consumer is updated in that same commit.
5. ADR revision 3, exact verification, independent review, and Backlog closeout finish the task.

No provider registry, media probe, transcoder, new dependency, config field, command token, or UI format selector is added.

## File responsibilities

- Create `tldw_chatbook/Video_Generation/video_formats.py`: the only immutable MP4/WebM MIME/container/extension vocabulary and normalization functions.
- Modify `tldw_chatbook/Video_Generation/adapters/base.py`, `worker.py`, `adapters/minimax_video_adapter.py`, and `adapters/comfyui_video_adapter.py`: produce and validate exact result format evidence.
- Modify `tldw_chatbook/Video_Generation/video_metadata.py` and `video_store.py`: persist the container and enforce canonical extensions.
- Modify `tldw_chatbook/Chat/console_generate_video.py` and `tldw_chatbook/UI/Screens/chat_screen.py`: carry extensions through generation, pending recovery, playback, copy, regeneration, and cards.
- Modify `Tests/ProductionApp/test_chat_composition_retirement.py`: migrate its direct app-owned `VideoStore` calls when defaults are removed.
- Modify ADR-044 decisions 1 and 7 and maintain TASK-3403 through Backlog CLI.

Extension-sensitive boundaries are parameterized for MP4/WebM. Existing extension-independent lock, rollback, orphan-stage, symlink, cleanup, and capacity matrices remain single-source.

---

### Plan handoff: Commit the approved execution contract before implementation

**Files:**
- Create: `Docs/superpowers/plans/2026-08-12-mime-driven-generated-video-extensions.md`
- Modify: `backlog/tasks/task-3403 - MIME-driven-generated-video-file-extensions.md`

- [ ] **Step 1: Re-read the rendered task and check the plan diff**

```bash
backlog task 3403 --plain
git diff --check
```

Expected: TASK-3403 is In Progress, its summary reflects caller-first/default-removal-last sequencing, its ADR check names ADR-044, and the detailed-plan link is correct.

- [ ] **Step 2: Commit the plan and task bookkeeping**

```bash
git add Docs/superpowers/plans/2026-08-12-mime-driven-generated-video-extensions.md \
  'backlog/tasks/task-3403 - MIME-driven-generated-video-file-extensions.md'
git diff --cached --check
git commit -m "docs: plan MIME-driven video extensions"
```

- [ ] **Step 3: Record the implementation base**

Record the resulting plan commit as `implementation_base` in the execution evidence. Confirm `git status --short` is clean before Task 1 writes any production test or code.

---

### Task 1: Immutable format contract, strict adapters, and worker validation

**Files:**
- Create: `tldw_chatbook/Video_Generation/video_formats.py`
- Create: `Tests/Video_Generation/test_video_formats.py`
- Modify: `tldw_chatbook/Video_Generation/adapters/base.py:56-81`
- Modify: `tldw_chatbook/Video_Generation/worker.py:20-132`
- Modify: `tldw_chatbook/Video_Generation/adapters/minimax_video_adapter.py:130-170`
- Modify: `tldw_chatbook/Video_Generation/adapters/comfyui_video_adapter.py:110-150,766-1015`
- Modify: `Tests/Video_Generation/test_contracts.py`
- Modify: `Tests/Video_Generation/test_worker.py`
- Modify: `Tests/Video_Generation/test_minimax_adapter.py`
- Modify: `Tests/Video_Generation/test_comfyui_adapter.py`
- Modify: `Tests/Chat/test_console_generate_video.py`

- [ ] **Step 1: Write the immutable closed-vocabulary RED tests**

Expose a read-only record set, not a mutable dictionary:

```python
SUPPORTED_VIDEO_FORMATS = frozenset(
    {
        ("mp4", "video/mp4", "mp4"),
        ("webm", "video/webm", "webm"),
    }
)


def test_supported_video_format_records_are_exact():
    assert SUPPORTED_VIDEO_FORMATS == frozenset(
        {
            ("mp4", "video/mp4", "mp4"),
            ("webm", "video/webm", "webm"),
        }
    )
```

Parameterize exact helper behavior for both records. Reject unknown, alias, dotted, mixed-case, empty, and third-container values. Add malformed runtime MIME values (`None`, bytes, list, object) and require bounded `ValueError`, never `AttributeError`/`TypeError`. MIME normalization may trim/lowercase and remove `;` parameters only.

- [ ] **Step 2: Write worker and outer-boundary RED tests**

Add required `container` to every fake result. Prove unknown request format rejects before adapter dispatch. Prove mismatches among requested format, result container, and MIME-derived container reject after adapter return as sanitized `VideoGenerationError`.

In `Tests/Chat/test_console_generate_video.py`, enter `run_video_generation` through the real worker and a fake adapter returning unknown/contradictory facts. Spy on `VideoStore.save` and `tempfile.TemporaryFile`; assert adapter dispatch occurred, but managed save, pending staging, temporary stream creation, and managed-root creation stayed at zero. This belongs in Task 1 because the worker guard makes it GREEN.

- [ ] **Step 3: Write MiniMax and ComfyUI RED tests**

MiniMax must accept exact normalized `video/mp4` (including harmless parameters) and reject missing MIME, `application/octet-stream`, and `video/webm` rather than relabeling them.

ComfyUI must advertise `{"mp4", "webm"}` for generic workflows. Add an adverse-order history fixture containing animated WebP, MP4, WebM, MOV, and unrelated-node descriptors; request-aware selection returns only the canonical suffix matching `request.format`. Terminal success with no match fails. Descriptor suffix and response MIME must agree.

For shipped H3, add/reuse a generation-entry test that loads the local graph, requests WebM, and asserts rejection before `_validate_required_nodes`, upload, queue, or any other network seam. Do not duplicate the existing lower-level H3 parameterization test.

- [ ] **Step 4: Run the strict RED command before production edits**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest \
  Tests/Video_Generation/test_video_formats.py \
  Tests/Video_Generation/test_contracts.py \
  Tests/Video_Generation/test_worker.py \
  Tests/Video_Generation/test_minimax_adapter.py \
  Tests/Video_Generation/test_comfyui_adapter.py \
  Tests/Chat/test_console_generate_video.py -q
```

Expected: failures for the missing format module/container field, missing worker guards, MiniMax fallback, MP4-only generic ComfyUI, first-recognized descriptor selection, and outer persistence not being blocked. Record the exact count.

- [ ] **Step 5: Implement the minimal immutable mapping and worker guards**

Keep `SUPPORTED_VIDEO_FORMATS` as the exact public `frozenset` above. Functions scan its two records; no registry or mutable cache is needed:

```python
def normalize_video_mime(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("unsupported video MIME")
    return value.split(";", 1)[0].strip().lower()


def canonical_video_extension(container: object) -> str:
    for known_container, _mime, extension in SUPPORTED_VIDEO_FORMATS:
        if container == known_container:
            return extension
    raise ValueError("unsupported video container")
```

Add required `container: str` to `VideoGenResult`. Validate request format before dispatch and exact result agreement after dispatch; translate every format error to bounded `VideoGenerationError` without echoing raw values.

- [ ] **Step 6: Update both real adapters in the same commit**

MiniMax removes its generic-MIME fallback, requires normalized `video/mp4`, and returns `container="mp4"`.

ComfyUI loads the local workflow before remote checks; if it is H3 and the request is not MP4, fail immediately. Generic polling receives the requested canonical container and selects only a matching canonical descriptor. `_download_output` independently validates suffix/MIME agreement and returns that container. H3 remains MP4-only.

- [ ] **Step 7: Run GREEN and mutations**

Run Step 4; expected all pass. Mutate one at a time and restore after each:

1. add a third format record (exact-set test fails);
2. remove malformed MIME type guard;
3. remove worker post-adapter agreement (outer no-persistence test fails);
4. restore MiniMax MIME fallback;
5. choose first recognized ComfyUI descriptor;
6. move H3 WebM rejection after remote preflight.

Finish with the restored Step 4 GREEN run.

- [ ] **Step 8: Commit Task 1**

```bash
git add tldw_chatbook/Video_Generation/video_formats.py \
  tldw_chatbook/Video_Generation/adapters/base.py \
  tldw_chatbook/Video_Generation/worker.py \
  tldw_chatbook/Video_Generation/adapters/minimax_video_adapter.py \
  tldw_chatbook/Video_Generation/adapters/comfyui_video_adapter.py \
  Tests/Video_Generation/test_video_formats.py \
  Tests/Video_Generation/test_contracts.py \
  Tests/Video_Generation/test_worker.py \
  Tests/Video_Generation/test_minimax_adapter.py \
  Tests/Video_Generation/test_comfyui_adapter.py \
  Tests/Chat/test_console_generate_video.py
git diff --cached --check
git commit -m "feat: validate MP4 and WebM generation results"
```

---

### Task 2: Metadata, generation, pending recovery, and external-save boundary

**Files:**
- Modify: `tldw_chatbook/Video_Generation/video_metadata.py`
- Modify: `tldw_chatbook/Video_Generation/video_store.py:220-270` (exact validation while compatibility defaults remain temporarily)
- Modify: `tldw_chatbook/Chat/console_generate_video.py:45-320`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py:17870-18570,18710-18765`
- Modify: `Tests/Video_Generation/test_video_metadata.py`
- Modify: `Tests/Chat/test_console_generate_video.py`
- Modify: `Tests/Chat/test_console_video_capacity.py`

- [ ] **Step 1: Write metadata RED tests**

Add MP4/WebM construction and round-trip cases, explicit invalid-container construction/decode rejection, and a hand-built historical payload with no `container` key that decodes to `container == "mp4"`. Present-invalid must never receive the historical fallback.

- [ ] **Step 2: Write generation and pending RED tests**

Make the fake adapter echo a matching MIME/container for MP4 or WebM. Assert `run_video_generation(video_format="webm")` sends WebM, saves `.webm`, records metadata container `webm`, and returns pending artifacts with extension `webm`. The ordinary Console command remains explicitly MP4. Regeneration passes `meta.container` explicitly.

- [ ] **Step 3: Write external-save RED tests**

Cover picker default `slug.webm`, no-suffix normalization to `.webm`, exact `.webm` acceptance, and reprompt for `.mp4`, `.mov`, uppercase `.WEBM`, or unrelated suffixes. Directly call `_copy_pending_video_external` with a mismatched suffix and assert it rejects before capability probing, `mkdir`, staging, or target inspection; parent and target remain absent.

Reuse the existing identity/no-clobber harness. Do not duplicate its race matrix per container.

- [ ] **Step 4: Run RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest \
  Tests/Video_Generation/test_video_metadata.py \
  Tests/Chat/test_console_generate_video.py \
  Tests/Chat/test_console_video_capacity.py -q
```

Expected: failures at missing metadata container, MP4-hard-coded generation/pending paths, regeneration defaulting to MP4, picker naming, and the lower-level suffix boundary.

- [ ] **Step 5: Implement minimal metadata and write-path plumbing**

Add validated `container: str = "mp4"` to metadata, serialize it, and distinguish absent from present-invalid on decode:

```python
container = "mp4" if "container" not in payload else payload["container"]
```

`run_video_generation(video_format="mp4")` passes the format to `build_request`, derives one extension from the validated result container, writes metadata container, and reuses the extension for save/pending staging.

Keep `VideoStore` method defaults temporarily so untouched readers remain runnable, but make every explicit extension pass exact shared validation—remove regex sanitization/fallback for explicit bad values.

- [ ] **Step 6: Implement pending/external boundary**

Use `artifact.extension` for retry, adoption, and picker default. Add one small picker-target normalizer: no suffix appends the canonical suffix; exact lowercase match passes; any other suffix reprompts with generic guidance. `_copy_pending_video_external` independently checks exact suffix before any filesystem or capability action.

The ordinary command passes `video_format="mp4"`; regeneration passes `video_format=meta.container`.

- [ ] **Step 7: Run GREEN and mutations**

Run Step 4. Mutate one at a time: restore generation MP4 hard-code, omit regeneration format, treat present-invalid metadata as missing, trust picker-only validation, and restore `.mp4` picker default. Each named test must fail; restore and finish GREEN.

- [ ] **Step 8: Commit Task 2**

```bash
git add tldw_chatbook/Video_Generation/video_metadata.py \
  tldw_chatbook/Video_Generation/video_store.py \
  tldw_chatbook/Chat/console_generate_video.py \
  tldw_chatbook/UI/Screens/chat_screen.py \
  Tests/Video_Generation/test_video_metadata.py \
  Tests/Chat/test_console_generate_video.py \
  Tests/Chat/test_console_video_capacity.py
git diff --cached --check
git commit -m "feat: carry video containers through Console writes"
```

---

### Task 3: Explicit production reads, playback, reload, and remount

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py:17530-17555,18570-18715`
- Modify: `Tests/Chat/test_console_video_actions.py`
- Modify: `Tests/Chat/test_console_video_message.py`
- Modify: `Tests/ProductionApp/test_chat_composition_retirement.py`

- [ ] **Step 1: Write action-path RED tests**

Create WebM message/files and drive real Play and Save-copy button dispatch. Assert `resolve(..., extension="webm")`, `.webm` playback, and collision names `name.webm`, then `name_1.webm`. Keep MP4 coverage.

- [ ] **Step 2: Write distinct screen-state and DB reload RED tests**

1. Serialize a WebM message through the real controller payload, restore through `_restore_console_message`, build card specs against a real `.webm` file, and assert ready/resolved.
2. Persist a WebM message and `.webm` file, reconstruct a fresh Console path from SQLite, and assert resolution.
3. Persist raw historical metadata JSON with no container plus an `.mp4` file and assert the fresh path resolves MP4.

- [ ] **Step 3: Update ProductionApp behavior tests for explicit metadata-derived reads**

Add `extension="mp4"` to direct setup writes/resolves in `Tests/ProductionApp/test_chat_composition_retirement.py`, then add one WebM card-remount assertion through the existing real navigation harness. This file is directly affected and belongs in every focused gate from this task onward.

- [ ] **Step 4: Run RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest \
  Tests/Chat/test_console_video_actions.py \
  Tests/Chat/test_console_video_message.py \
  Tests/ProductionApp/test_chat_composition_retirement.py -q
```

Expected: WebM card/action/save-copy failures because production reads still omit the metadata-derived extension. Existing MP4 tests remain runnable because store defaults have not yet been removed.

- [ ] **Step 5: Implement explicit production reads**

Card specs, Play, and Save-copy derive `canonical_video_extension(meta.container)` and pass it to `resolve`. Save-copy uses that extension for collision naming. No filesystem probing or suffix guessing is introduced.

- [ ] **Step 6: Run GREEN and production mutations**

Run Step 4. Remove the metadata-derived extension from card, Play, and Save-copy one at a time; the corresponding WebM test must fail. Mutate `VideoGenerationMetadata.from_json` to drop the historical fallback and mutate screen-state serialization to omit container; the outer reload/remount tests must fail. Never prove load-bearing coverage by removing test assertions.

- [ ] **Step 7: Commit Task 3**

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py \
  Tests/Chat/test_console_video_actions.py \
  Tests/Chat/test_console_video_message.py \
  Tests/ProductionApp/test_chat_composition_retirement.py
git diff --cached --check
git commit -m "fix: resolve persisted videos by container"
```

---

### Task 4: Remove VideoStore defaults after caller migration

**Files:**
- Modify: `tldw_chatbook/Video_Generation/video_store.py:230-450`
- Modify: `Tests/Video_Generation/test_video_store.py`
- Modify: any focused direct caller found by the required inventory command

- [ ] **Step 1: Inventory every remaining direct caller before API removal**

Run:

```bash
rg -n '\.(save|resolve|adopt_oversized)\(' \
  tldw_chatbook Tests/Video_Generation Tests/Chat \
  Tests/ProductionApp/test_chat_composition_retirement.py
```

Classify every `VideoStore` call. Production calls must already be explicit from Tasks 2–3. Mechanically add `extension="mp4"` to legacy MP4 test setup calls in `Tests/Video_Generation/test_video_store.py` and any remaining focused consumer. Do not change unrelated stores that happen to expose `save`/`resolve`.

- [ ] **Step 2: Write strict API and extension-sensitive RED tests**

Parameterize MP4/WebM save/resolve and cross-extension slug collision behavior. Add calls to `save`, `adopt_oversized`, and `resolve` without extension and expect `TypeError` after the signature change. Explicit empty/dotted/unsupported extensions must fail before root creation.

- [ ] **Step 3: Add existing-behavior characterization for unknown files**

Plant safe `.mov`/unknown files inside a valid message directory. Confirm current `_snapshot` already includes them and session/TTL/capacity removes or accounts for them while canonical `resolve` cannot serve them. This test is expected GREEN before production edits; record it as characterization, not RED. Mutating `_snapshot` to filter unknown suffixes must make it fail.

- [ ] **Step 4: Run focused pre-change characterization and RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest \
  Tests/Video_Generation/test_video_store.py \
  Tests/Chat/test_console_generate_video.py \
  Tests/Chat/test_console_video_capacity.py \
  Tests/Chat/test_console_video_actions.py \
  Tests/Chat/test_console_video_message.py \
  Tests/ProductionApp/test_chat_composition_retirement.py -q
```

Expected: new exact API/default-removal assertions fail; unknown-file characterization passes; migrated production paths remain green.

- [ ] **Step 5: Remove defaults and make allocation cross-extension aware**

Remove extension defaults from `save`, `adopt_oversized`, and `resolve`. `_video_path` already uses exact shared validation from Task 2. `allocate_slug` checks both canonical extensions so one marker slug cannot name two live files. Leave `_snapshot`'s complete safe-file inventory intact.

- [ ] **Step 6: Run GREEN and mutations**

Run Step 4. Mutate one at a time: restore a default, omit cross-extension slug check, filter unknown suffixes from `_snapshot`, and restore permissive explicit-extension sanitization. Each named test must fail; restore and finish GREEN.

- [ ] **Step 7: Commit Task 4**

```bash
git add tldw_chatbook/Video_Generation/video_store.py \
  Tests/Video_Generation/test_video_store.py \
  Tests/Chat/test_console_generate_video.py \
  Tests/Chat/test_console_video_capacity.py \
  Tests/Chat/test_console_video_actions.py \
  Tests/Chat/test_console_video_message.py \
  Tests/ProductionApp/test_chat_composition_retirement.py
git diff --cached --check
git commit -m "fix: require explicit generated video extensions"
```

---

### Task 5: ADR revision 3, exact verification, and task closeout

**Files:**
- Modify: `backlog/decisions/044-ephemeral-generated-video-storage-playback-and-streaming.md`
- Modify: `backlog/tasks/task-3403 - MIME-driven-generated-video-file-extensions.md`
- Verify every production/test path changed in Tasks 1–4

- [ ] **Step 1: Amend ADR-044 as revision 3**

Update the ADR status line to revision 3 and describe revision 3 as the canonical generated-video container/extension amendment. Mark the changed Decision 1 clause explicitly as Revision 3: storage is `generated_videos/<message_id>/<slug>.<validated-ext>` from the exact MP4/WebM mapping. Mark Decision 7 explicitly as Revision 3: canonical container is persisted; historical missing-container metadata means MP4. Do not silently rewrite prior revision text and do not add a new ADR.

- [ ] **Step 2: Run the exact touched-file test gate**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B -m pytest \
  Tests/Video_Generation/test_video_formats.py \
  Tests/Video_Generation/test_contracts.py \
  Tests/Video_Generation/test_worker.py \
  Tests/Video_Generation/test_minimax_adapter.py \
  Tests/Video_Generation/test_comfyui_adapter.py \
  Tests/Video_Generation/test_video_metadata.py \
  Tests/Video_Generation/test_video_store.py \
  Tests/Chat/test_console_generate_video.py \
  Tests/Chat/test_console_video_capacity.py \
  Tests/Chat/test_console_video_actions.py \
  Tests/Chat/test_console_video_message.py \
  Tests/ProductionApp/test_chat_composition_retirement.py -q
```

Expected: all pass; only already-documented environment/platform warnings or skips may remain. Do not run the full collection or unrelated RuntimePolicy/UI suites.

- [ ] **Step 3: Run exact Ruff gates**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check \
  tldw_chatbook/Video_Generation/video_formats.py \
  tldw_chatbook/Video_Generation/adapters/base.py \
  tldw_chatbook/Video_Generation/worker.py \
  tldw_chatbook/Video_Generation/adapters/minimax_video_adapter.py \
  tldw_chatbook/Video_Generation/adapters/comfyui_video_adapter.py \
  tldw_chatbook/Video_Generation/video_metadata.py \
  tldw_chatbook/Video_Generation/video_store.py \
  tldw_chatbook/Chat/console_generate_video.py \
  Tests/Video_Generation/test_video_formats.py \
  Tests/Video_Generation/test_contracts.py \
  Tests/Video_Generation/test_worker.py \
  Tests/Video_Generation/test_minimax_adapter.py \
  Tests/Video_Generation/test_comfyui_adapter.py \
  Tests/Video_Generation/test_video_metadata.py \
  Tests/Video_Generation/test_video_store.py \
  Tests/Chat/test_console_generate_video.py \
  Tests/Chat/test_console_video_capacity.py \
  Tests/Chat/test_console_video_actions.py \
  Tests/Chat/test_console_video_message.py \
  Tests/ProductionApp/test_chat_composition_retirement.py

/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check \
  --select E9,F63,F7,F82 tldw_chatbook/UI/Screens/chat_screen.py
```

Expected: both commands exit 0.

- [ ] **Step 4: Run reproducible py_compile, diff, privacy, and artifact gates**

```bash
/bin/zsh -lc 'set -e; task3403_pycache=$(mktemp -d); test -n "$task3403_pycache"; PYTHONPYCACHEPREFIX="$task3403_pycache" /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m py_compile tldw_chatbook/Video_Generation/video_formats.py tldw_chatbook/Video_Generation/adapters/base.py tldw_chatbook/Video_Generation/worker.py tldw_chatbook/Video_Generation/adapters/minimax_video_adapter.py tldw_chatbook/Video_Generation/adapters/comfyui_video_adapter.py tldw_chatbook/Video_Generation/video_metadata.py tldw_chatbook/Video_Generation/video_store.py tldw_chatbook/Chat/console_generate_video.py tldw_chatbook/UI/Screens/chat_screen.py'

git diff --check

git diff --unified=0 origin/dev...HEAD -- \
  tldw_chatbook/Video_Generation \
  tldw_chatbook/Chat/console_generate_video.py \
  tldw_chatbook/UI/Screens/chat_screen.py | \
  rg -n 'https?://|/Users/|api[_-]?key|authorization|BEGIN [A-Z ]+PRIVATE KEY'

git diff --name-only origin/dev...HEAD | \
  rg -ni '\.(mp4|webm|mov|avi)$|(^|/)(dist|build)/|\.egg-info/'

git status --short --untracked-files=all | \
  rg -ni '\.(mp4|webm|mov|avi)$|(^|/)(dist|build)/|\.egg-info/'
```

Expected: py_compile and `git diff --check` exit 0. The privacy search and both committed/uncommitted artifact searches produce no output; because `rg` returns 1 for no match, record that as the expected clean result rather than a command failure. Confirm the diff contains no Image_Generation production changes.

- [ ] **Step 5: Review the complete implementation diff**

Compare the implementation base through HEAD against TASK-3403, the approved spec, and ADR-044. Verify the exact two-format set, explicit writes/reads, no-persistence invalid-result boundary, unknown-file accounting, pre-mutation external suffix validation, H3/MiniMax MP4 restriction, and real reload/remount evidence. Request independent code/spec review and resolve every Critical/Important finding.

- [ ] **Step 6: Update Backlog notes and mark Done only after evidence exists**

Use Backlog CLI to check all five ACs and add concise Implementation Notes covering approach, trade-offs, exact focused tests/static results, commits, ADR revision 3, deviations, and modified files. Set TASK-3403 to Done only after all Definition-of-Done items hold. If no generalizable incident surfaced, state that no lesson update was needed; do not invent one.

- [ ] **Step 7: Commit closeout documentation**

```bash
git add backlog/decisions/044-ephemeral-generated-video-storage-playback-and-streaming.md \
  'backlog/tasks/task-3403 - MIME-driven-generated-video-file-extensions.md'
git diff --cached --check
git commit -m "docs: complete MIME-driven video extensions"
```

- [ ] **Step 8: Final post-commit verification**

Confirm `git status --short` is clean, TASK-3403 is Done with all ACs checked, exact commit hashes are recorded, the committed plan still matches execution, and no generated media/build artifact exists.
