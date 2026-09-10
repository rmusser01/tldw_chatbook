# Post-rebase SQLite / live TTS harness audit

Read-only audit on 2026-09-09. Incoming comparison is pinned to
`2e3389e694e93592a1c66e5c3416bf29a1057d6c..a36fc6133c69f77b8b14596a59918de34261f8ef`.
Root subsequently rebased the TTS work to `1edcbb1aee` and froze source for live
validation. No application imports, tests, inference, ASR, playback, server,
package changes, or repository edits were performed by this audit.

## Result

No required harness compatibility change found for the current isolated Kokoro
CPU/MPS/ONNX language scenarios. This is a static compatibility review, not new
qualification of the incoming SQLite lifecycle.

## Runtime and installed artifact prerequisites

- Incoming `pyproject.toml` raises the application minimum from Python 3.11 to
  Python 3.12. ADR-125 requires the interpreter's actual `sqlite3` implementation
  to expose `Connection.setconfig`, `Connection.getconfig`, and
  `SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE`, and to successfully set/read back `True` on
  an in-memory connection. The admission check runs before opening a profile
  store and returns typed `runtime_unsupported` if unavailable. Python version
  alone is not the capability proof.
- Root independently reported a passing Python 3.12.11 / SQLite 3.49.1 capability
  probe in `post-rebase-sqlite-capability.json`. This audit did not rerun it.
- The helper is **not a compiled native extension**. `DB/private_sqlite_process.py`
  launches the same `sys.executable` with `-I -S` and the resolved absolute sibling
  `DB/private_sqlite_helper_entry.py`. The entry bootstraps only fixed package
  namespaces and stdlib-only helper leaves, then runs private framed pipe IPC.
  There is no separate SQLite SDK, compiler, helper executable build, or third
  party helper dependency added by this change. The app's build requirement is
  still `setuptools>=77.0`; a `python -m build` workflow also needs its existing
  `build` frontend. Do not replace the interpreter's sqlite module with an
  unverified shim.
- A newly built/installed application wheel must include the helper entry and
  its fixed 15-file Python closure. The exact inventory is
  `Tests/Packaging/test_private_sqlite_helper_distribution.py::PRIVATE_HELPER_CLOSURE`.
  That test checks byte-exact wheel/install content and execution from hostile
  working-directory/PYTHONPATH roots. Since launch uses `-I -S`, missing installed
  leaves cannot be repaired by the harness's `PYTHONPATH` or a source checkout.
  A source-root run instead uses its source-root sibling helper naturally.
- Helper processes need a usable POSIX exec/pipe environment and the existing
  private filesystem/no-follow checks. They do not need a network connection,
  listener, credentials, model assets, or an audio device.
- No files changed under `packages/` in this pinned upstream diff.
  `packages/tldw_profile_core` remains the separate pure-Python contract package
  with its existing pydantic/rfc8785 dependencies and setuptools packaging.

## Current harness startup and cleanup

`scripts/validate_live_tts.py` imports `TldwCli` only to reuse its completion
handler. It constructs its own `Host(App)`, not a `TldwCli` instance. The Host
creates a Speech Playground pane without a profile preset. Its Console handler
uses `default_profile_id_reader=lambda: None` and leaves
`profile_service_loader=None`; its `ConsoleChatStore()` has no durable
persistence adapter. `build_default_tts_service` constructs lazy adapter specs,
preferences and a private reference-file materializer, not a profile repository.
The Speech pane's normal catalog mount path does not open a profile repository.

Upstream `Tests/Packaging/test_tts_profile_repository_import_closure.py` explicitly
asserts that app/personas import leaves the repository, schema, policy,
validation, migration and store-lock closure deferred until explicit repository
first use. The new `TTS.__getattr__` preserves the public repository export while
making its import lazy. The incoming TTS diff contains no backend generation or
adapter-service lifecycle changes.

Therefore the current mounted generation/playback paths have no newly owned
profile SQLite handle or retained proof helper to add to their cleanup. Their
existing handler/service/player joined cleanup remains the relevant ownership
check. The harness hashes every application-package `.py` file before/after;
this already covers the new helper files. It does not currently record a SQLite
capability result or helper process census, and its passing `final_resources`
must not be presented as qualification of full-app profile-store shutdown.

## If profile-store scenarios are added later

Use a fresh isolated application data directory and the real profile owner;
record the loaded helper path/hash and runtime capability along with actual
owned-child reaping. Await the repository's own close. Healthy cleanup performs
a guarded PASSIVE checkpoint, closes SQLite, then releases proof/store ownership.
Lost proof is deliberately different: `restart_required` retains the live
SQLite handle, store lease and worker until process exit and latches new TTS
profile admission. Do not interpret that state as a leak-free close, repeatedly
retry it as normal cleanup, or force-close the retained SQLite handle. Current
live Kokoro scenarios do not exercise this terminal policy.

References inspected: ADR-125; incoming `pyproject.toml`; the two Packaging tests
named above; `DB/private_sqlite_process.py` and helper entry; TTS package lazy
exports, `profile_sqlite_policy.py`, repository/schema/proof/errors;
`adapter_bootstrap.py`; current live harness Host/handler construction and cleanup;
STTS/TTS event handlers; Speech Playground mount; in-memory Console store.
