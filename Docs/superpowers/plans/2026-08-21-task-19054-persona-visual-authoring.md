# TASK-19054 Persona Visual Authoring and Import Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users review, edit, import, stage, Save, and Cancel profile-local Persona Visual packs from Personas Workbench without changing the active runtime until one explicit, authority-checked publication.

**Architecture:** Pure authoring helpers build an isolated immutable draft from an active Persona Visual graph and translate approved edits into the existing publication snapshot. A separate synchronous archive boundary validates the pinned server `.tldw-persona-vpack` layout, copies declared files into bounded profile-private staging, and returns a path-free review draft plus an opaque cleanup capability. A dedicated Textual widget renders the nine baseline states, bounded custom states, inventory, and one selected lazy preview; `PersonasScreen` owns all repository, filesystem, async, dialog, publication, cache-invalidation, and navigation fencing.

**Tech Stack:** Python 3.11, Textual 8, frozen/slotted dataclasses, SQLite, `zipfile`, Pillow, existing Persona Visual validation/assets/repository/publication boundaries, pytest, Ruff.

**ADR required:** no

**ADR path:** `backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md`

**Reason:** ADR-074 already fixes the separate Persona Visual runtime, local-Persona ownership, review-before-publication, archive trust boundary, immutable versions, and no-provider/no-Buddy scope. This task implements that decision without adding an architectural choice.

---

## File map

- Create `tldw_chatbook/Persona_Visual/authoring.py`: path-free immutable draft records, active-graph conversion, state add/replace/clear operations, validation inventory, authority snapshot, and publication-snapshot conversion.
- Create `tldw_chatbook/Persona_Visual/importer.py`: pinned server V1 archive inspection, bounded private staging, archive/source identity revalidation, review result, and identity-pinned cleanup.
- Modify `tldw_chatbook/Persona_Visual/__init__.py`: export only the small authoring/import surface intended for Workbench.
- Create `tldw_chatbook/Widgets/Persona_Widgets/personas_persona_visual_pack_widget.py`: baseline/custom state browser, path-free inventory, selected lazy preview, explicit Preparing/Busy/Saving states, and labelled authoring actions.
- Modify `tldw_chatbook/Widgets/Persona_Widgets/persona_profile_editor_widget.py`: mount the Persona Visual browser within the existing Persona editor and switch it between local/server/new-Persona availability states.
- Modify `tldw_chatbook/Widgets/Persona_Widgets/personas_pane_messages.py`: typed Persona Visual Replace/Clear/Add/Import/Save/Cancel/select messages.
- Modify `tldw_chatbook/UI/Screens/personas_screen.py`: active graph loading, draft lifecycle, one-operation admission, dialog/decode/import workers, publication, identity-scoped invalidation, authoritative refresh, dirty-navigation integration, and honest legacy import labels.
- Create `Tests/Persona_Visual/test_persona_visual_authoring.py`: pure draft and publication bridge tests.
- Create `Tests/Persona_Visual/test_persona_visual_importer.py`: golden server-compatible archive plus hostile archive/staging/cleanup tests.
- Create `Tests/UI/test_personas_persona_visual_pack.py`: real Textual widget layout, focus, copy, and message tests.
- Create `Tests/UI/test_personas_persona_visual_authoring.py`: screen orchestration, authority, cancellation, publication, invalidation, and server/local behavior tests.
- Modify existing focused Persona widget/workbench tests only where the new mounted section or corrected legacy label changes an incumbent contract.
- Modify `backlog/tasks/task-19054 - Author-and-import-Persona-Visual-packs.md`: status, checked acceptance criteria, and concise implementation notes after verification.

## Frozen archive boundary

The importer accepts only the pinned server V1 native archive:

```text
schema_version: tldw.persona_visual_pack.v1
manifest.json
metadata/pack.json
metadata/assets.json
checksums/sha256.json
assets/...                       # every present asset declared exactly once
README.md                        # optional, plain-text review only
signatures/README.md             # optional reserved metadata
```

It accepts only `sprite_frames` manifest version 1. Required JSON and present asset members must be declared by the outer manifest/checksum inventory and no undeclared regular file is accepted. The importer applies the stricter programme limits already owned by Persona Visual foundation: at most 256 assets, 100 MiB uncompressed asset bytes, 2 MiB per JSON document, safe canonical POSIX member names, supported raster MIME/decode/dimensions/frame counts, and no missing-byte archive import. Compressed bytes, members, compression ratio, central-directory identity, source inode/size/timestamps/digest, and free-space needs are bounded before extraction and revalidated before review is returned.

## Task 1: Add the isolated draft contract

**Files:**
- Create: `Tests/Persona_Visual/test_persona_visual_authoring.py`
- Create: `tldw_chatbook/Persona_Visual/authoring.py`
- Modify: `tldw_chatbook/Persona_Visual/__init__.py`

- [ ] Write behavioral RED tests using real validated manifests and repository graph records. Cover all nine baseline rows, bounded safe custom states, path-free inventory, exact authority capture, Replace/Clear/Add Custom immutability, invalid custom states, required-state validation, Cancel preservation, and deterministic publication-snapshot conversion.
- [ ] Run the focused file and confirm behavioral failures because the authoring API is absent.
- [ ] Implement frozen/slotted draft records that snapshot every scalar and container. Reuse `validate_persona_visual_manifest`; do not accept screen widgets, database handles, absolute paths, or mutable repository records.
- [ ] Make edits return a new draft. A cleared state removes only its direct mapping; validation reports whether required states still resolve through fallback. Save conversion refuses invalid/incomplete drafts and emits the exact existing `PersonaVisualPublicationSnapshot` contract.
- [ ] Run the focused file GREEN and mutate draft isolation/authority checks to prove the tests discriminate them.
- [ ] Commit the slice as `feat: add Persona Visual authoring drafts`.

## Task 2: Add review-first `.tldw-persona-vpack` import

**Files:**
- Create: `Tests/Persona_Visual/test_persona_visual_importer.py`
- Create: `tldw_chatbook/Persona_Visual/importer.py`

- [ ] Create a small golden fixture builder matching the pinned server member layout and metadata fields. Write RED tests for a valid full pack and for traversal, absolute/backslash/device/Unicode-or-case collisions, links, encryption, nesting, undeclared/external/missing files, duplicate keys/members, bad checksums, MIME/decode/dimension/frame mismatch, member/count/size/ratio/free-space budgets, source replacement, cancellation, and cleanup substitution.
- [ ] Implement synchronous import preview under a caller-supplied profile-private staging root. Open and attest the source, validate the central directory before extraction, stream declared bytes with hard limits and digests, use no-follow private staging, and validate the resulting manifest/assets through the foundation boundary.
- [ ] Return only an immutable path-free review result plus an opaque module-issued cleanup capability. Keep private source/staging paths inside the module.
- [ ] Revalidate source and staged identities before publication-snapshot handoff. Cleanup must reserve and verify the exact issued staging identity and delete only that tree; unverifiable cleanup fails closed.
- [ ] Run the importer file GREEN and perform targeted mutations for archive identity, declaration, checksum, cancellation, and cleanup guards.
- [ ] Commit the slice as `feat: import Persona Visual review drafts`.

## Task 3: Add the Persona Visual Workbench widget

**Files:**
- Create: `Tests/UI/test_personas_persona_visual_pack.py`
- Create: `tldw_chatbook/Widgets/Persona_Widgets/personas_persona_visual_pack_widget.py`
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_pane_messages.py`
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/persona_profile_editor_widget.py`

- [ ] Write real Textual RED tests at normal and 80x24 geometry. Assert nine labelled baseline states, bounded custom rows, path-free inventory, one selected preview, keyboard-operable labelled Replace/Clear/Add Custom/Import/Save/Cancel actions, focus preservation, honest local/server/new-Persona states, and typed messages.
- [ ] Implement one compact browser with metadata-only rows and a single selected preview node. Do not decode unselected assets or add key bindings.
- [ ] Use distinct idle-draft, Preparing/Importing/Previewing, and non-cancellable Saving presentations. Cancel is visible for an idle draft and cancellable work, but hidden during atomic publication.
- [ ] Mount it in the existing Persona editor scroll body. Local saved Personas can author; unsaved local Personas are prompted to Save first; server Personas show `Save Local Copy first` and no import/edit controls.
- [ ] Run widget/profile tests GREEN, then run the required Impeccable review after the final visible change and apply only in-scope findings.
- [ ] Commit the slice as `feat: add Persona Visual Workbench editor`.

## Task 4: Wire loading, preview, edits, and import into `PersonasScreen`

**Files:**
- Create: `Tests/UI/test_personas_persona_visual_authoring.py`
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py`

- [ ] Write RED screen tests with real frozen records and a real SQLite repository where authority matters. Cover eligible local load, inactive/no-pack empty draft, server Save Local Copy, selected-only preview, Replace/Clear/Add/import staging, first-operation cancellation, duplicate rejection, navigation decline/preserve, navigation approval/signal/drain/discard, stale session/Persona/revision/binding/draft/source/staging fences, and path-free failures.
- [ ] Add a separate Persona Visual author snapshot and one admitted task plus caller-owned cancellation event. Every filesystem/hash/decode/import operation runs through a named shield/drain helper off the event loop; same-owner operations reject instead of queueing.
- [ ] Reuse the existing repository/runtime/asset loader. Screen callbacks receive path-free records; only the selected asset is decoded, and every await is followed by full editor/session/Persona/binding/draft authority checks before widget mutation.
- [ ] Keep draft changes isolated in screen state. Cancel signals/drains active work, cleans only owned imported staging, drops the draft, and reloads the authoritative graph. Dirty navigation includes draft/inflight Persona Visual work.
- [ ] Correct legacy Character copy/filtering so `Import Expression Set` accepts only its legacy format; expose Persona Visual import only in the Persona Visual section; keep future Actor Pack import absent or separately labelled.
- [ ] Run the focused screen/widget/workbench cases GREEN and kill admission/cancellation/session/selected-preview/import-cleanup mutations.
- [ ] Commit the slice as `feat: author Persona Visual packs in Workbench`.

## Task 5: Publish once and invalidate exact identities

**Files:**
- Modify: `Tests/UI/test_personas_persona_visual_authoring.py`
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py`

- [ ] Extend RED tests for one Save admission, draft revalidation, current Persona revision, exact expected nine-field active identity, no-pack first activation, immutable next version, publication error cleanup, cancellation during thread publication, cancellation during post-commit reconciliation, stale editor after commit, and repeated Save rejection.
- [ ] Call existing synchronous `publish_persona_visual` off-loop with a caller authority guard that is valid both before and inside the repository transaction. Shield and drain irreversible work before releasing admission.
- [ ] On success invalidate affected runtime/cache entries by stable old and new complete identities only, preserving unrelated Persona/pack/version entries. Isolate mounted-consumer invalidation failures and always perform the current-fenced authoritative editor reload. Failed/cancelled precommit work invalidates nothing.
- [ ] Consume only opaque cleanup capabilities via the foundation cleanup API. Never log or display source paths, staging paths, archive members, provider text, bytes, or cleanup tokens.
- [ ] Run publication/invalidation tests GREEN and mutate publish admission, authority, cleanup, old/new invalidation, unrelated-cache preservation, and post-commit drain checks.
- [ ] Commit the slice as `feat: publish Persona Visual authoring drafts`.

## Task 6: Focused verification and closeout

**Files:**
- Modify: `backlog/tasks/task-19054 - Author-and-import-Persona-Visual-packs.md`
- Modify lessons/docs only if this task produced a genuinely reusable incident.

- [ ] Establish isolated `HOME`, XDG config/data/cache, explicit config, and data roots before app imports. Record the assigned worktree path and assert imported modules resolve inside it.
- [ ] Run all `Tests/Persona_Visual` plus the touched Persona profile widget, Persona Workbench, Persona Visual authoring, publication, and relevant architecture/privacy/diagnostic gates. Do not run the repository-wide full suite.
- [ ] Run scoped Ruff, format, compile, and `git diff --check`. Run real SQLite publication/repository tests and normal/80x24 Pilot layouts.
- [ ] Review the final diff against all eight ACs, ADR-074, the approved programme spec, the pinned server compatibility files, and the no-provider/no-SVI/no-Buddy/no-Actor-Pack boundaries.
- [ ] Mark all task ACs complete, set status Done, and add concise Implementation Notes including RED/GREEN/mutation/static/Impeccable evidence and any truthful baseline deviations.
- [ ] Commit closeout as `docs: complete Persona Visual authoring task`.

## Verification commands

Use the assigned worktree with the shared complete environment, while asserting import provenance:

```bash
TLDW_TEST_MODE=1 /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/Persona_Visual \
  Tests/UI/test_persona_profile_widgets.py \
  Tests/UI/test_personas_persona_visual_pack.py \
  Tests/UI/test_personas_persona_visual_authoring.py \
  Tests/UI/test_personas_workbench_foundation.py \
  Tests/UI/test_personas_workbench_state.py

/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check <touched-python-files>
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check <touched-python-files>
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m compileall -q <touched-python-packages>
git diff --check
```

The broader `Tests/UI/test_personas_workbench.py` file is added to the final touched gate because `PersonasScreen` and the Persona editor are modified. No full repository suite is authorized or claimed.
