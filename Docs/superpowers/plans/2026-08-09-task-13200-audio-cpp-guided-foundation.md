# TASK-13200 Guided audio.cpp Foundation Implementation Plan

> **For Codex:** Execute this plan test-first, one numbered task at a time. Do not
> pull generated `server.json`, process lifecycle, Settings widgets, Speech Lab,
> downloads, or later-family recipe work into this PR.

**Goal:** Add the typed guided-setup state, sealed release-0.5.1 recipes for all
Supertonic and PocketTTS packages, complete 21-family/67-package accounting, and
a bounded read-only local package scanner.

**Architecture:** Keep the existing `AudioCppConfig` runtime projection unchanged.
A new full-settings model owns Guided and dormant manual/External values. A frozen
recipe registry validates accepted package snapshots and exposes only allowlisted
model projections. A separate scanner receives that registry, inspects one explicit
root off-loop, and returns bounded evidence without launching or contacting anything.

**Tech Stack:** Python 3.11, Pydantic 2, stdlib `asyncio`, `dataclasses`, `enum`,
`hashlib`, `os`, `pathlib`, `stat`, `threading`, and pytest.

**ADR required:** no new ADR

**ADR path:** `backlog/decisions/050-audio-cpp-generated-model-setup-ownership.md`

**Reason:** This is a direct implementation of ADR-050's already-approved
structured-settings, sealed-recipe, and explicit-root scanner boundaries.

---

### Task 1: Typed full-settings and accepted-package snapshots

**Files:**
- Create: `tldw_chatbook/TTS/audio_cpp_guided_config.py`
- Create: `Tests/TTS/test_audio_cpp_guided_config.py`

- [x] Write failing tests for legacy External/manual defaults, full dormant-value
      round trips, bounded backend/device/thread/body/busy settings, immutable
      accepted projections, unique public model IDs, and rejected extra fields.
- [x] Run only the new test file and confirm the failures are missing APIs rather
      than broken fixtures.
- [x] Implement the smallest frozen Pydantic models and safe mapping projection
      that satisfy those tests while leaving `AudioCppConfig` untouched.
- [x] Re-run the new tests and existing `test_audio_cpp_config.py` plus
      `test_audio_cpp_managed_config.py`.

### Task 2: Sealed recipes, exact matching, and release accounting

**Files:**
- Create: `tldw_chatbook/TTS/audio_cpp_recipes.py`
- Create: `Tests/TTS/test_audio_cpp_recipes.py`

- [x] Write failing tests for all 4 Supertonic and 11 PocketTTS package IDs,
      immutable recipe records, exact required layouts and model projections,
      reviewed task/options/backend posture, traversal/absolute-path rejection,
      exact/ambiguous/unknown/incomplete/permission matching, accepted-snapshot
      validation, and the 21-family/67-package accounting totals.
- [x] Run the new recipe tests and confirm the expected missing-API failures.
- [x] Implement frozen recipe/evidence/accounting records, the production registry,
      pure matcher/validator, and explicit Open-gap entries for the remaining 52
      packages. Expose Verified claims only from exact evidenced tuples.
- [x] Re-run recipe and guided-config tests.

### Task 3: Bounded explicit-root package scanner

**Files:**
- Create: `tldw_chatbook/TTS/audio_cpp_package_scanner.py`
- Create: `Tests/TTS/test_audio_cpp_package_scanner.py`

- [x] Write failing tests for explicit-root recognition, canonical deduplication,
      multiple variants in one root, missing/bad GGUF evidence, entry/depth/time/
      candidate/result/metadata limits, cancellation, off-event-loop execution,
      permission isolation, nested symlink escape attempts, top-level symlink
      disclosure, and sanitized capped evidence.
- [x] Run the scanner tests and confirm the expected missing-API failures.
- [x] Implement one iterative no-follow scan, bounded header inspection, recipe-root
      derivation, candidate identity construction, cooperative cancellation, and an
      `asyncio.to_thread` wrapper. Never read tensor payloads or arbitrary file
      contents.
- [x] Re-run scanner and recipe tests.

### Task 4: Side-effect proof, documentation truth, and task closeout

**Files:**
- Modify: `Tests/TTS/test_audio_cpp_guided_config.py`
- Modify: `Tests/TTS/test_audio_cpp_recipes.py`
- Modify: `Tests/TTS/test_audio_cpp_package_scanner.py`
- Modify: `backlog/tasks/task-13200 - Build-guided-audio.cpp-package-recipes-and-bounded-scanner.md`

- [x] Add a joined pure-foundation regression proving configuration, matching, and
      scanning use no subprocess, socket, HTTP, download, or model-file write seam.
- [x] Run mutation-style negative fixtures to prove near-match and bound assertions
      fail when the safety condition is deliberately removed.
- [x] Run focused tests, Ruff on changed Python files, bundle-independent full TTS
      tests as proportionate verification, and `git diff --check`.
- [x] Self-review against every acceptance criterion and ADR-050; record only actual
      deviations or remaining evidence gaps.
- [x] Check all acceptance criteria, add concise Implementation Notes including the
      ADR decision and verification evidence, and mark TASK-13200 Done only after all
      checks succeed.
