# Writing Suite Structural Parity

Date: 2026-04-22

## Scope Summary

This tranche adds Chatbook's first source-explicit Writing destination for structural authoring parity with the current `tldw_server` manuscript contract.

Implemented scope:

- Local project, manuscript, chapter, and scene CRUD.
- Local unassigned-chapter bucket.
- Local Markdown scene working drafts.
- Local manuscript and chapter metadata/structure working state without authored body drafts.
- Local manual versions for manuscripts, chapters, and scenes.
- Local soft-delete, trash listing, and restore.
- Local reorder and scene move support across chapter and direct manuscript parents.
- Server project, part-as-manuscript, chapter, and scene CRUD through `WritingScopeService`.
- Server structure browse with unassigned chapters rendered in an explicit project-level bucket.
- Server search and reorder through the current server manuscript routes.
- Source-specific Local/Server mode selection with no mixed view and no local fallback for server actions.

Out of scope for this tranche:

- LLM writing generation, outline generation, revision helpers, export, publishing, collaboration, sync/mirroring, and prose-IDE features.
- Server direct manuscript-level scenes, server manual version history, server trash restore, and server scene reparenting until verified server endpoints exist.

## Capability Table

| Capability | Local | Server |
| --- | --- | --- |
| Project CRUD | Supported | Supported through manuscript projects |
| Manuscript CRUD | Supported | Supported through server parts |
| Chapter CRUD | Supported | Supported |
| Unassigned chapters | Supported | Rendered from server structure |
| Scene CRUD under chapter | Supported | Supported |
| Direct manuscript-level scenes | Supported | Blocked |
| Scene Markdown drafts | Native Markdown | Preserved through adapter wrapper/plain fallback |
| Manual versions | Supported for manuscript/chapter/scene | Blocked |
| Restore historical versions | Supported locally | Blocked |
| Soft-delete | Supported | Supported where server delete routes soft-delete |
| Trash listing and restore | Supported locally | Blocked |
| Reorder manuscripts/chapters/scenes | Supported | Supported for parts/chapters/scenes |
| Chapter assign/unassign | Supported | Supported through verified part update/reorder shape |
| Scene move/reparent | Supported locally | Blocked |
| Search | Source-specific local search | Source-specific server search |

## Unsupported Server Capabilities

The following server actions are blocked centrally and surfaced as disabled/unavailable states:

- `server_direct_manuscript_scene_unavailable`
- `server_version_history_unavailable`
- `server_trash_restore_unavailable`
- `server_scene_reparent_unavailable`

These are intentionally not emulated locally in server mode. Chatbook does not invent hidden chapters, write local mirrors, or silently fall back to local state when the user selects server mode.

## Verification

Verification was run with the bundled Python 3.12 runtime:

```bash
/Users/macbook-dev/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest Tests/tldw_api/test_writing_manuscripts_client.py Tests/Writing_Interop Tests/DB/test_writing_db.py Tests/UI/test_writing_screen.py Tests/UI/test_screen_navigation.py -q
```

Result: `119 passed`

```bash
/Users/macbook-dev/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest Tests/Writing_Interop/test_local_writing_service.py Tests/Writing_Interop/test_server_writing_service.py Tests/Writing_Interop/test_writing_scope_service.py Tests/UI/test_writing_screen.py -q
```

Result: `70 passed`

```bash
/Users/macbook-dev/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest Tests/UI/test_screen_navigation.py Tests/UI/test_writing_screen.py -q
```

Result: `36 passed`

```bash
/Users/macbook-dev/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest Tests/Writing_Interop -q
```

Result: `62 passed`

```bash
/Users/macbook-dev/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest Tests/tldw_api Tests/RuntimePolicy Tests/UI/test_screen_navigation.py -q
```

Result: `227 passed`, `2 failed`

Known unrelated failures from the dirty RuntimePolicy/API worktree state:

- `Tests/RuntimePolicy/test_runtime_policy_core.py::test_backend_exception_classifier_handles_session_invalid_authentication_errors`
- `Tests/RuntimePolicy/test_runtime_policy_core.py::test_backend_exception_classifier_handles_session_invalid_401s`

Both failures return `None` from `classify_backend_exception(...)` where the RuntimePolicy tests expect `server_session_invalid`.

```bash
PYTHONPYCACHEPREFIX=/tmp/tldw-writing-pycache /Users/macbook-dev/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m compileall tldw_chatbook/Writing_Interop tldw_chatbook/DB/Writing_DB.py tldw_chatbook/UI/Screens/writing_screen.py tldw_chatbook/UI/Writing_Window.py tldw_chatbook/Widgets/Writing
```

Result: no syntax errors.

```bash
git diff --check
```

Result: clean.

## Manual Smoke Test

1. Start Chatbook.
2. Open `Writing`.
3. In `Local` mode, create project -> manuscript -> chapter -> scene.
4. Save Markdown in the scene and create a manual version.
5. Delete the scene and restore it from local trash.
6. Switch to `Server` mode with a configured server.
7. Create project -> manuscript(part) -> chapter -> scene under chapter.
8. Confirm an unassigned server chapter appears in `Unassigned Chapters`.
9. Confirm direct manuscript-level scene and server version actions are disabled with reason codes.
10. Switch back to `Local` mode and confirm local records are unchanged.
