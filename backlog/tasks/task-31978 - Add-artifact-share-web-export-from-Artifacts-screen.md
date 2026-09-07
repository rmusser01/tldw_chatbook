---
id: TASK-31978
title: Add artifact share web export from Artifacts screen
status: To Do
assignee: []
created_date: '2026-09-06 02:18'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let a user select local Chatbook artifacts on the Artifacts screen and temporarily host them as a small web page (aiohttp child process, staged snapshot) so recipients can browse and download exactly those bundles, with optional single shared username/password (HTTP Basic). Recipients import via the existing Chatbook Import Wizard. Design: Docs/superpowers/specs/2026-09-05-artifact-share-web-export-design.md
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Artifacts screen Share action opens a multi-select dialog over local chatbook records (records without on-disk bundles disabled) — `Tests/UI/test_artifacts_screen_share.py::test_share_button_and_binding_open_dialog`, `Tests/UI/test_artifact_share_dialog.py` (disabled options carry the no-bundle reason; staging fail-closed for bundle-less records in `Tests/Web_Server/test_artifact_share_manifest.py::test_stage_share_rejects_record_without_bundle`)
- [x] #2 Starting a share stages an immutable copy plus bundle.zip and spawns a child server exposing only the staged files via HTML page, per-artifact download, bundle download, and index.json — `Tests/Web_Server/test_artifact_share_manifest.py` (staging copies + sha256 + 0600 manifest + bundle), `Tests/Web_Server/test_artifact_share_server.py` (index page escaped, index.json/artifact download match staged bytes, traversal containment, subprocess ready-line + download)
- [x] #3 When auth is configured every route requires Basic auth (401 + WWW-Authenticate, per-IP lockout 10 fails/30s); non-loopback bind without password requires typed confirmation — `Tests/Web_Server/test_artifact_share_server.py::test_auth_challenges_and_admits`, `::test_auth_lockout_after_ten_failures`, `Tests/UI/test_artifact_share_dialog.py::test_lan_without_password_requires_typed_confirmation`
- [x] #4 Share session is app-owned: survives navigation, single active share, stops cleanly via UI and on app exit; stale share dirs are swept at startup — `Tests/Web_Server/test_artifact_share_controller.py` (lifecycle, second start replaces first, startup sweep removes stale dirs), `Tests/UI/test_artifacts_screen_share.py::test_banner_reflects_active_share`, `::test_stop_button_calls_controller`, `::test_app_shutdown_stops_share`
- [x] #5 Targeted tests pass (manifest/staging, server routes+auth, controller lifecycle, dialog, screen wiring) without a full-suite run — verified on branch `codex/artifact-share-web-export` at `0f62ae58e`: 51 passed across the seven targeted files (see task-7 report in `.superpowers/sdd/2026-09-05-artifact-share-web-export/task-7-report.md`)
<!-- AC:END -->

## Implementation Plan

Executed via the SDD plan `Docs/superpowers/plans/2026-09-05-artifact-share-web-export.md` (spec: `Docs/superpowers/specs/2026-09-05-artifact-share-web-export-design.md`), tasks 1–7:

1. Spec + ADR-123 + backlog scaffolding.
2. Staging/manifest/auth-verifier module (`Web_Server/artifact_share_manifest.py`).
3. Child aiohttp server (`Web_Server/artifact_share_server.py`).
4. App-side controller with child supervision (`Web_Server/artifact_share.py`).
5. Share dialog with guardrails (`UI/Screens/artifact_share_dialog.py`).
6. Screen/app wiring: `s` keybinding, Share/Stop buttons, banner, app-owned controller lifecycle.
7. User guide (`Docs/User_Guide/artifacts.md`), spec closeout appendix, this task's notes, final targeted verification.

ADR check:

```text
ADR required: yes
ADR path: backlog/decisions/123-artifact-share-web-export.md
```

## Implementation Notes

- **Approach.** Per ADR-123 (Approach A of the spec): the TUI stages
  immutable copies of the selected Chatbook zips (plus a pre-built
  `bundle.zip`) into `<user_data>/share/<uuid>/` with a `0600` manifest
  (schema 1, Pydantic, staging-relative paths, sha256 per staged file),
  then spawns `python -m tldw_chatbook.Web_Server.artifact_share_server`
  in its own process group. The child serves ONLY the staging directory:
  plain-HTML index (no JS, escaped names, `no-store`), per-artifact
  download via opaque 128-bit keys, `bundle.zip`, `index.json`, optional
  HTTP Basic (PBKDF2 verifier in the manifest, constant-time compare,
  per-IP lockout 10 fails / 30 s, verified-credential cache), uniform
  security headers incl. RFC 5987 filenames. Stop = SIGTERM→SIGKILL
  escalation + staging deletion; the child PPID-polls and exits when
  orphaned; a dead-PID startup sweep clears crash residue. The
  controller is app-owned (`_wire_prompt_chatbook_services` creates it;
  `on_unmount` stops the share), so the session survives navigation;
  the Artifacts screen holds only dialog + banner wiring.
- **Files added:** `tldw_chatbook/Web_Server/artifact_share_manifest.py`,
  `tldw_chatbook/Web_Server/artifact_share_server.py`,
  `tldw_chatbook/Web_Server/artifact_share.py`,
  `tldw_chatbook/UI/Screens/artifact_share_dialog.py`,
  `Tests/Web_Server/test_artifact_share_manifest.py`,
  `Tests/Web_Server/test_artifact_share_server.py`,
  `Tests/Web_Server/test_artifact_share_controller.py`,
  `Tests/UI/test_artifact_share_dialog.py`,
  `Tests/UI/test_artifacts_screen_share.py`,
  `Docs/superpowers/specs/2026-09-05-artifact-share-web-export-design.md`,
  `Docs/superpowers/plans/2026-09-05-artifact-share-web-export.md`,
  `backlog/decisions/123-artifact-share-web-export.md`.
- **Files modified:** `tldw_chatbook/UI/Screens/artifacts_screen.py`
  (`s` binding, Share/Stop buttons, status banner, dialog open/start/stop
  workers, footer hint), `tldw_chatbook/app.py` (app-owned
  `artifact_share_controller`, startup sweep, `_shutdown_artifact_share`
  in `on_unmount`), `Docs/User_Guide/artifacts.md` (Sharing artifacts
  section).
- **Deviations from the spec** (recorded in the spec's post-implementation
  appendix): LAN-IP discovery resolves the hostname first with the UDP
  probe as fallback (test-suite network guard compatibility); aiohttp
  3.14 requires `__middleware_version__ = 1` markers, set directly to
  keep aiohttp lazily importable; `_shutdown_artifact_share` nulls the
  controller reference in `finally` for strictly-once shutdown; the
  dialog uses a `SelectionList` (not the sketched `DataTable`) with
  disabled options carrying the no-bundle reason and no separate
  select-all control; manifest auth fields are
  `pbkdf2_salt_hex`/`pbkdf2_hash_hex` and the child PID lives in
  `status.json`. Known gap recorded honestly: no mid-share child-crash
  watcher or "restart" banner action (next stop/start or the startup
  sweep cleans up); future work.
- **Plain-HTTP caveat** documented in the user guide: Basic auth is an
  access gate, not encryption; hostile networks need a TLS reverse proxy.
