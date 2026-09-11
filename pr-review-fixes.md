# PR 2484 Qodo Review Fixes — Report

Branch `codex/artifact-share-web-export`, base HEAD `9ef873ac5`. All 16 findings
addressed as specified; no new dependencies. Line numbers below refer to the
post-fix files.

## HIGH

### #8 — Unvalidated chatbook_id in staged filenames
- `tldw_chatbook/Web_Server/artifact_share_manifest.py:205-221` (`stage_share`):
  the identifier is interpolated only after
  `safe_id = slugify_share_name(str(chatbook_id))` (empty falls back to
  "artifact"), plus a defensive containment check
  (`(share_dir / candidate).resolve().parent == share_dir.resolve()`, else
  `ArtifactShareStagingError`) after the dedup suffix is settled, before
  `atomic_copy`. The raw id is still recorded as manifest data
  (`source_chatbook_id`); only the filename is sanitized.
- Test: `Tests/Web_Server/test_artifact_share_manifest.py::test_stage_share_sanitizes_untrusted_chatbook_ids`
  — `../../evil` and `/abs/path` ids stage as plain contained names; staging
  dir contains exactly the staged files + bundle + manifest.

### #13 — Closed screen can push dialog
- `tldw_chatbook/UI/Screens/artifacts_screen.py:150` (`__init__`:
  `_share_dialog_generation = 0`), `:205` (`on_unmount` bumps it next to the
  worker cancel), `:1231-1241` (helper
  `_share_dialog_publish_allowed(generation) -> bool` = `not
  _chatbook_unmounted and generation == current`), `:1245-1264`
  (`_run_share_dialog_open` captures the generation at start and guards BOTH
  error-notify `call_from_thread` sites and the `push_screen` publication).
- Test: `Tests/UI/test_artifacts_screen_share.py::test_share_dialog_publication_guard_invalidated_on_unmount`
  — real `pop_screen()` unmount: generation bumps, stale generation blocked,
  and the unmounted flag dominates even a hypothetical current generation.

### #15 — CR/LF header injection in Content-Disposition
- `tldw_chatbook/Web_Server/artifact_share_server.py:500-512` (`_ascii_fallback`):
  strips every ASCII control char (`ord(ch) < 0x20` and `DEL 0x7f`) from the
  fallback token. `filename*=UTF-8''…` was already safe (`urllib.parse.quote`
  percent-encodes controls).
- Test: `Tests/Web_Server/test_artifact_share_server.py::test_content_disposition_strips_crlf`
  — artifact name `evil\r\nX-Injected: 1` and share name with
  `\r\nSet-Cookie: pwn=1` produce headers with no `\r`/`\n` on both the
  artifact and bundle downloads.

## MEDIUM

### #1 — Unvalidated source path
- `artifact_share_manifest.py:174-201` (`stage_share`): after `expanduser()`,
  a symlinked source raises `ArtifactShareStagingError` (checked on the
  pre-resolve path), then `.resolve()` is required to be `is_file()`. A
  comment documents the trust model: sources are user-chosen export
  locations and are deliberately NOT confined to the private chatbooks dir;
  staging containment is enforced server-side (`_contained_file`).
- Test: `test_stage_share_rejects_symlinked_source` — a symlink to a real
  bundle is refused and staging is cleaned.

### #2 — Unbounded dialog fields
- `tldw_chatbook/UI/Screens/artifact_share_dialog.py:26-28,184-197`: bounds
  share_name ≤ 200, username ≤ 64, password ≤ 256 (port already bounded),
  rejected via the existing `_status(...)` pattern. The raw-dict dismiss
  result stays — a pydantic model is overkill for an internal
  `ModalScreen[dict | None]` result consumed by one caller
  (`_on_share_dialog_result`).
- Test: `Tests/UI/test_artifact_share_dialog.py::test_oversized_fields_are_rejected`
  (all three over-bound values blocked; values at the bound accepted).

### #3 — aiohttp import guidance
- `artifact_share_server.py:81-99` (`_require_aiohttp` → `require_dependency("aiohttp", "web")`;
  aiohttp is a real module name, so the gate import works as-is). Called at
  the top of `build_app()` (:135) and `run()` (:382); in `main()` (:561) it
  sits after argparse (so `--help` works without the extra) and before any
  server work — documented in the helper docstring (handlers are unreachable
  before `build_app`).
- Test: `test_build_app_gates_aiohttp_through_optional_deps` — a raising
  `require_dependency` makes `build_app()` surface the ImportError guidance;
  a pass-through mock proves the `("aiohttp", "web")` delegation and that the
  app still builds.

### #4 — Missing docstrings
- Concise Google-style docstrings (summary + Args/Returns/Raises where
  applicable) added to every public class and function/method across all four
  files: manifest (`build_share_auth`, `verify_share_auth`, `SharedArtifact`,
  `ArtifactShareManifest`, `share_root_dir`, `stage_share`, `load_manifest`,
  `pid_alive`, `sweep_stale_shares`), server (class `__init__`, `build_app`,
  `run`, `main`, all `handle_*`, `_file_response`), controller (`ShareStatus`,
  `__init__`, `status`, `startup_sweep`, `start_share`, `stop_share`,
  `_emit_status`), dialog (`action_cancel`). Kept to summary + sections.

### #5 — Handler type annotations
- `artifact_share_server.py:25-30`: `if TYPE_CHECKING: from aiohttp import web`
  (file already has `from __future__ import annotations`). All handlers
  annotated `async def handle_x(self, request: "web.Request") -> "web.StreamResponse"`
  (:214, :220, :246, :265), `build_app(self) -> "web.Application"` (:126),
  `_file_response(...) -> "web.StreamResponse"`, and both middlewares
  (:148-152, :170-174) with
  `handler: "Callable[[web.Request], Awaitable[web.StreamResponse]]"`.

### #6 — Sanitizer policy in `_render_index` — BRANCH TAKEN: no repo sanitizer
- Verified `tldw_chatbook/Utils/input_validation.py`: the only HTML-adjacent
  helpers are `sanitize_string` (control-char filter — not a tag allow-lister)
  and `validate_text_input(allow_html=False)` (validator, not a sanitizer);
  no bleach/nh3/lxml-clean exists as an approved HTML allow-list sanitizer.
  Per the agreed resolution, NO new dependency was added. Instead
  `artifact_share_server.py:477-497` adds `_escape_fragment(value)`: `html.escape(quote=True)`
  (the existing defense, kept) with an explicit final invariant check that the
  escaped fragment carries no raw `<`/`>` (re-escaping as the fail-safe if
  that impossible state ever occurred), documented in the docstring. Used for
  name, description, kind, and share_name (:344-361).
- Test: `test_escape_fragment_guarantees_no_raw_markup` plus the pre-existing
  escaped-page integration test
  (`test_index_page_lists_artifacts_escaped_no_auth`).

### #7 — Callback log context
- `tldw_chatbook/Web_Server/artifact_share.py:243-253` (`_emit_status`): the
  failure log now reads
  `Artifact share status callback failed (share='<name>')` with
  `"<stopped>"` when clearing — share name only, never credentials.

### #9 — Popen-raise leaves staging
- `artifact_share.py:142-163` (`start_share`): the `subprocess.Popen` call is
  wrapped; on `OSError` it runs `self._cleanup_staging()` and re-raises as
  `ArtifactShareError(f"Could not start share server: {exc}")`.
- Test: `Tests/Web_Server/test_artifact_share_controller.py::test_start_share_popen_failure_cleans_staging`
  (monkeypatched raising Popen → error raised, staging dir empty).

### #10 — PBKDF2 on event loop
- `artifact_share_server.py:122,188-197`: `verify_share_auth` runs via
  `await loop.run_in_executor(None, verify_share_auth, auth, username, password)`,
  bounded by `self._verify_semaphore = asyncio.Semaphore(2)` created in
  `__init__` (lazy loop binding on 3.10+, so construction without a running
  loop is safe — repo floor is 3.11, venv 3.12). The `(user,pass)` digest
  fast path stays synchronous BEFORE offloading (:185-187); failure/lockout
  bookkeeping stays on the loop after the await. Existing auth + lockout
  integration tests pass unchanged (10 failures → 429, then good creds still
  locked).

### #11 — Bundle symlink escape
- `artifact_share_server.py:287-306` (`_contained_file(path) -> Path | None`):
  requires regular non-symlink file whose `.resolve()` stays inside
  `staging_dir`. Used by both `handle_bundle` (:269-274; None → 410 Gone) and
  `_resolve_staged` (:330-336). Behavior note: for artifacts, a staged name
  with separators still 404s via the pre-existing `Path(staged).name != staged`
  check; a symlinked staged file now surfaces 410 (gone) instead of 404 —
  cosmetic-only difference for a corrupted manifest, no test relied on it.
- Test: `test_bundle_symlink_escape_is_gone_never_external_bytes` — bundle.zip
  swapped for a symlink to an external file → 410 and the external bytes are
  never served.

### #12 — PID reuse in sweep
- `artifact_share_manifest.py:309-336` (`_pid_is_share_server`): after the
  cheap `pid_alive` pre-check, `ps -p <pid> -o command=` (timeout 5s) must
  contain `artifact_share_server`; on ps failure/unavailable (non-posix) it
  fails OPEN (True) preserving the old pid-exists behavior. Wired at :371.
- Tests: `test_sweep_removes_dead_pid_and_live_non_share_pid` (real `ps`
  against a live non-share pid — a real `sleep 30` child, deliberately NOT
  `os.getpid()`, because this suite's own argv contains
  `test_artifact_share_server.py`, which contains the `artifact_share_server`
  marker substring and would make the assertion invocation-dependent);
  `test_sweep_keeps_live_pid_confirmed_as_share_server` (positive case via
  monkeypatched helper, per the agreed allowance; real-child evidence stays
  in the controller lifecycle integration test);
  `test_pid_is_share_server_requires_share_server_command` (the ps gate
  itself: match / mismatch / fail-open).
- The previous `test_sweep_removes_dead_pid_and_keeps_live` was superseded by
  the two-step policy (its "keep live" half asserted exactly the behavior
  this finding removes).

### #14 — Colon usernames
- Dialog: `artifact_share_dialog.py:184-188` (rejects `:` when auth enabled,
  actionable `_status` copy). Controller: `artifact_share.py:112-116`
  (`start_share` raises `ArtifactShareError("Username cannot contain ':'.")`
  before staging). Rationale in both: Basic auth splits on the first `:`.
- Tests: `test_username_with_colon_is_rejected` (dialog stays open),
  `test_start_share_rejects_colon_username` (raises, nothing staged).

### #16 — Plaintext cache
- `artifact_share_server.py:105-106,185-199,461-474`: `_verified` is now a
  `set[str]` of `hashlib.sha256(f"{username}\x00{password}".encode()).hexdigest()`
  digests (`_verified_digest` helper); membership compares the digest of the
  presented pair; the pair is never stored.
- Test: `test_verified_cache_stores_digests_not_plaintext`; existing
  challenge/admit and lockout tests pass unchanged (behavior identical).

## Test evidence

Command (the agreed 7-file targeted set):

```
.venv/bin/python -m pytest \
  Tests/Web_Server/test_artifact_share_manifest.py \
  Tests/Web_Server/test_artifact_share_server.py \
  Tests/Web_Server/test_artifact_share_controller.py \
  Tests/Web_Server/test_web_server_dependency_gate.py \
  Tests/UI/test_artifact_share_dialog.py \
  Tests/UI/test_artifacts_screen_share.py \
  Tests/UI/test_artifacts_screen_reports.py -q --timeout=120
```

Output tail:

```
......................................................................... [ 98%]
.                                                                        [100%]
73 passed in 22.14s
```

(One iteration earlier in the loop: the new colon-username controller test
asserted against a never-created staging root — fixed by using the repo's
`not root.exists() or not any(root.iterdir())` pattern; everything else was
green on the first run.)

Static analysis: no linter (ruff/flake8/mypy) is installed in this venv and
none is configured in pyproject; all changed files were verified with
`python -m py_compile` and the suite above.

## Deviations / notes

- #3: `_require_aiohttp()` in `main()` sits after `argparse` rather than as
  the literal first statement, so `--help` works on installs without the
  `[web]` extra; it still precedes every code path that imports aiohttp.
- #6 branch: NO repo HTML allow-list sanitizer exists → belt-and-braces
  fragment check in `_escape_fragment`, no new dependency (as pre-agreed).
- #11: symlinked STAGED artifact files now return 410 instead of 404 (was an
  unreachable-without-symlink branch); bundle symlink is 410 by choice.
- #12: the positive "share server pid preserved" unit case monkeypatches the
  ps helper (spec-sanctioned); the mismatch case uses a real `ps` against a
  real non-share process.
