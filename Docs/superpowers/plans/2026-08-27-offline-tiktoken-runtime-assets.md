# Offline tiktoken Runtime Assets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship Chatbook's supported tiktoken encodings as immutable package assets so standard token estimates and token chunking work on a cold offline installation without hidden downloads.

**Architecture:** Package tiktoken's native SHA-1-keyed cache files plus a reviewed JSON manifest and MIT notices. At the earliest package import, preserve either explicit cache environment variable; otherwise replace tiktoken 0.14.0's `read_file_cached` seam with a read-only manifest-checked loader. Reuse the canonical distribution checker and installed-wheel harness to prove exact artifact contents, read-only execution, and absence of fetch/mutation behavior.

**Tech Stack:** Python 3.11+, tiktoken 0.14.0, stdlib `hashlib`/`json`/`pathlib`, setuptools package data, pytest.

**Spec:** `Docs/superpowers/specs/2026-08-27-offline-tiktoken-runtime-assets-design.md`

**ADR:** `backlog/decisions/093-offline-tiktoken-runtime-assets.md`

**Closeout scope:** The implementation branch is committed and verified here.
Rebase, push, and PR creation are separate integration actions and are not part
of this execution.

---

### Task 1: Closed runtime bundle and early loader guard

**Files:**
- Create: `tldw_chatbook/Utils/tiktoken_runtime.py`
- Create: `tldw_chatbook/assets/tiktoken_cache/manifest.json`
- Create: `tldw_chatbook/assets/tiktoken_cache/LICENSE.txt`
- Create: `tldw_chatbook/assets/tiktoken_cache/NOTICE.txt`
- Move: `Tests/fixtures/tiktoken_cache/6d1cbeee0f20b3d9449abfede4726ed8212e3aee` to `tldw_chatbook/assets/tiktoken_cache/6d1cbeee0f20b3d9449abfede4726ed8212e3aee`
- Move: `Tests/fixtures/tiktoken_cache/6c7ea1a7e38e3a7f062df639a5b80947f075ffe6` to `tldw_chatbook/assets/tiktoken_cache/6c7ea1a7e38e3a7f062df639a5b80947f075ffe6`
- Move: `Tests/fixtures/tiktoken_cache/9b5ad71b2ce5302211f9c61530b329a4922fc6a4` to `tldw_chatbook/assets/tiktoken_cache/9b5ad71b2ce5302211f9c61530b329a4922fc6a4`
- Move: `Tests/fixtures/tiktoken_cache/fb374d419588a4632f3f557e76b4b70aebbca790` to `tldw_chatbook/assets/tiktoken_cache/fb374d419588a4632f3f557e76b4b70aebbca790`
- Add: `tldw_chatbook/assets/tiktoken_cache/0ea1e91bbb3a60f729a8dc8f777fd2fc07cd8df4`
- Add: `tldw_chatbook/assets/tiktoken_cache/ec7223a39ce59f226a68acc30dc1af2788490e15`
- Modify: `tldw_chatbook/__init__.py`
- Modify: `Tests/conftest.py`
- Modify: `Tests/test_tiktoken_vendored_cache.py`
- Modify: `Tests/Packaging/test_installed_distribution.py`
- Delete: `Tests/fixtures/tiktoken_cache/README.md`

- [x] **Step 1: Write failing source-runtime tests.** Point the cache inventory test at `tldw_chatbook/assets/tiktoken_cache`, require the exact six blob keys plus `manifest.json`, `LICENSE.txt`, and `NOTICE.txt`, load `gpt2`, `r50k_base`, `p50k_base`, `cl100k_base`, and `o200k_base`, and assert package import installs the guarded reader when neither override exists.

- [x] **Step 2: Write every failing runtime and installed behavior test before implementation.** In fresh subprocesses, prove pre-import `TIKTOKEN_CACHE_DIR` and `DATA_GYM_CACHE_DIR` values remain byte-for-byte unchanged and leave upstream `read_file_cached` installed. A separate subprocess must block imports of `tiktoken`, import Chatbook successfully, and exercise the character estimator. Unit-test missing, corrupt, URL/hash-mismatch, and unmanifested reads with `tiktoken.load.read_file` replaced by a sentinel that must never run. Also add the package-first and direct-engine-first installed probes plus missing/corrupt installed-tree probes described below, parameterized across source-built and sdist-rebuilt wheels.

- [x] **Step 3: Run RED.** Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
    Tests/test_tiktoken_vendored_cache.py \
    Tests/Packaging/test_installed_distribution.py::test_installed_tiktoken_bundle_is_offline_and_immutable \
    Tests/Packaging/test_installed_distribution.py::test_installed_tiktoken_bundle_missing_or_corrupt_falls_back_without_writes
  ```

  Expected: FAIL because the runtime asset directory and guard do not exist.

- [x] **Step 4: Add the minimal guarded reader.** Implement a private cached manifest loader and a reader with this contract:

  ```python
  def _read_bundled_file(blobpath: str, expected_hash: str | None = None) -> bytes:
      try:
          entry = _manifest_by_url()[blobpath]
      except KeyError as error:
          raise BundledTiktokenAssetError(...) from error
      cache_key = hashlib.sha1(blobpath.encode()).hexdigest()  # nosec B324
      if entry["cache_key"] != cache_key:
          raise BundledTiktokenAssetError(...)
      if expected_hash != entry["sha256"]:
          raise BundledTiktokenAssetError(...)
      try:
          data = (_ASSET_DIR / cache_key).read_bytes()
      except OSError as error:
          raise BundledTiktokenAssetError(...) from error
      if hashlib.sha256(data).hexdigest() != expected_hash:
          raise BundledTiktokenAssetError(...)
      return data
  ```

  `install_tiktoken_runtime()` must return before importing tiktoken when either override variable is present. Otherwise it must catch `ImportError` so package import preserves the character fallback; when tiktoken is available, verify the 0.14.0 seam parameters, set the default package cache path, and replace only `tiktoken.load.read_file_cached`.

- [x] **Step 5: Install the guard at package import.** Call `install_tiktoken_runtime()` from `tldw_chatbook/__init__.py` before any Chatbook submodule can resolve an encoding. Remove the test-only cache override from `Tests/conftest.py` so tests exercise production ownership.

- [x] **Step 6: Build the reviewed asset directory.** Move the four existing verified blobs, acquire r50k/p50k from the exact tiktoken 0.14.0 constructor URLs, verify all six SHA-256 hashes, copy tiktoken's MIT license, and record encoding name, URL, full cache key, expected hash, tiktoken version, reviewed constructor module/path, `read_file_cached` signature, cache-key algorithm, model-to-encoding coverage, license source, collaborator clarification URL, and repeatable update procedure in the manifest/notice.

- [x] **Step 7: Run source GREEN and commit.** Run the focused source runtime test plus `Tests/Chunking/test_tokens_offsets.py`; the installed probes remain intentionally red until Task 3 declares the package data. Commit only after the source behavior is green.

### Task 2: Deterministic real-token and character-fallback tests

**Files:**
- Modify: `Tests/Chat/test_token_counter.py`

- [x] **Step 1: Reproduce the baseline RED.** Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Chat/test_token_counter.py Tests/test_tiktoken_vendored_cache.py
  ```

  Expected on the untouched test: the claimed fallback case uses tiktoken, caches its result, and contaminates the later character-band assertion.

- [x] **Step 2: Isolate tokenizer tiers.** Remove the `skipif` from the mandatory real-tokenizer test. Make fallback tests explicitly disable tiktoken and custom tokenizers, calling `clear_estimate_cache()` before and after each tier override:

  ```python
  tc.clear_estimate_cache()
  monkeypatch.setattr(tc, "TIKTOKEN_AVAILABLE", False)
  monkeypatch.setattr(tc, "custom_tokenizers_available", lambda: False)
  yield
  tc.clear_estimate_cache()
  ```

- [x] **Step 3: Run GREEN and commit.** Rerun the exact command and confirm real-token and fallback paths pass in one process without modifying production cache behavior.

### Task 3: Exact package and release-checker contract

**Files:**
- Modify: `pyproject.toml`
- Modify: `MANIFEST.in`
- Modify: `Packaging/check_manifest.py`
- Modify: `Tests/Packaging/test_installed_distribution.py`

- [x] **Step 1: Write failing packaging tests.** Extend artifact assertions to require the exact nine-entry cache inventory in both sdist and wheel, exact `Requires-Dist: tiktoken==0.14.0` in wheel METADATA and sdist PKG-INFO, and release-checker failures for each missing required cache record plus an unexpected cache-prefix member.

- [x] **Step 2: Run RED.** Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
    Tests/Packaging/test_installed_distribution.py::test_built_artifacts_match_distribution_contract \
    Tests/Packaging/test_installed_distribution.py::test_release_checker_accepts_fresh_artifacts \
    Tests/Packaging/test_installed_distribution.py::test_release_checker_rejects_missing_tiktoken_asset \
    Tests/Packaging/test_installed_distribution.py::test_release_checker_rejects_unexpected_tiktoken_asset
  ```

  Expected: FAIL because assets are not declared, dependency is unpinned, and the checker does not own the prefix.

- [x] **Step 3: Implement minimal package declarations.** Pin `tiktoken==0.14.0`; enumerate the nine package-data entries and matching `MANIFEST.in` paths. Do not add a second manifest system.

- [x] **Step 4: Extend the canonical checker.** Add one `TIKTOKEN_RESOURCE_PATHS` set to both required artifact sets, compare every member under the cache prefix for exact equality, and reject metadata whose tiktoken requirements are anything other than `tiktoken==0.14.0`.

- [x] **Step 5: Run packaging and installed GREEN.** Build source wheel/sdist once through the existing module-scoped fixture. Run the release-checker nodes plus the prewritten installed probes. The probes must assert their package root is below `EXPECTED_TARGET`, use separate package-first and direct-engine-first subprocesses, begin without either cache override, point the default cache inside the installed package, prohibit tiktoken's upstream network reader, tokenize all five encodings, and execute inside `_read_only_installed_tree` for both build paths.

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
    Tests/Packaging/test_installed_distribution.py::test_built_artifacts_match_distribution_contract \
    Tests/Packaging/test_installed_distribution.py::test_release_checker_accepts_fresh_artifacts \
    Tests/Packaging/test_installed_distribution.py::test_release_checker_rejects_missing_tiktoken_asset \
    Tests/Packaging/test_installed_distribution.py::test_release_checker_rejects_unexpected_tiktoken_asset \
    Tests/Packaging/test_installed_distribution.py::test_installed_tiktoken_bundle_is_offline_and_immutable \
    Tests/Packaging/test_installed_distribution.py::test_installed_tiktoken_bundle_missing_or_corrupt_falls_back_without_writes
  ```

  The missing/corrupt probe must prove route-specific outcomes with no upstream fetch and an unchanged installed snapshot: direct `_read_bundled_file` raises `BundledTiktokenAssetError`; token estimation logs and returns its current OpenAI character approximation; and `Chunk_Lib`, with Transformers deterministically disabled, raises `ChunkingError` before word-approximate chunks. Commit when both build paths, both import paths, and both mutation modes pass.

### Task 4: Documentation, verification, and task closeout

**Files:**
- Modify: `Docs/Design/Packaging.md`
- Modify: `Docs/User_Guide/library/import-and-export.md`
- Modify: `Docs/superpowers/specs/2026-08-27-offline-tiktoken-runtime-assets-design.md`
- Modify: `Docs/superpowers/plans/2026-08-27-offline-tiktoken-runtime-assets.md`
- Modify: `backlog/decisions/093-offline-tiktoken-runtime-assets.md`
- Modify: `backlog/tasks/task-2526 - Ship-tiktoken-and-its-encoding-tables-for-offline-token-estimates.md`

- [x] **Step 1: Document the runtime contract.** Explain the standard offline bundle, explicit pre-import cache override behavior, closed inventory for new encodings, immutable package ownership, update/hash process, and accepted MIT clarification evidence.

- [x] **Step 2: Run focused verification.** Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
    Tests/Chat/test_token_counter.py \
    Tests/test_tiktoken_vendored_cache.py \
    Tests/Chunking/test_tokens_offsets.py \
    Tests/Chunking/test_chunk_lib_shim.py
  ```

  Run packaging selection separately so its `-k` filter cannot deselect the
  token/runtime/chunking files:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
    Tests/Packaging/test_installed_distribution.py \
    -k 'tiktoken or built_artifacts_match_distribution_contract or release_checker'
  ```

  The packaging selection covers every tiktoken probe, the built-artifact
  contract, release-checker acceptance, and the complete release-checker
  hardening matrix against the module-scoped fresh distribution directory. Run
  Ruff on changed Python files, `python -m py_compile` on the loader/checker,
  and `git diff --check`. Do not run the full repository suite without separate
  owner opt-in.

- [x] **Step 3: Review requirements and diff.** Map every TASK-2526 acceptance criterion to fresh evidence, run independent spec and code-quality reviews, and fix/re-review every valid finding.

- [x] **Step 4: Close the task.** Check every acceptance criterion, add concise Implementation Notes with verification evidence and ADR links, then use Backlog CLI to set TASK-2526 to Done and verify the rendered task/file path.

- [x] **Step 5: Commit the documentation and closeout.** Commit the owned
  documentation, accepted ADR record, checked acceptance criteria, and final
  evidence. Do not rebase, push, or open a PR during this closeout execution.
