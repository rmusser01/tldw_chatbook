# Server Sharing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose tldw_server's non-admin sharing contract in Chatbook as a remote-only surface.

**Architecture:** Add typed sharing schemas and `TLDWAPIClient` methods for workspace shares, shared-with-me discovery, shared workspace proxy operations, and share tokens. Wrap those endpoints in a remote-only service plus runtime-policy-aware scope seam, then mount a lightweight `Sharing` panel under `Tools & Settings` with explicit local-mode unavailable guidance and JSON-oriented controls.

**Tech Stack:** Python 3, Pydantic v2, Textual, pytest/pytest-asyncio, existing `runtime_policy` action IDs.

---

### Task 1: Typed Sharing API Contract

**Files:**
- Create: `tldw_chatbook/tldw_api/sharing_schemas.py`
- Modify: `tldw_chatbook/tldw_api/client.py`
- Modify: `tldw_chatbook/tldw_api/__init__.py`
- Test: `Tests/tldw_api/test_sharing_client.py`

- [x] **Step 1: Write failing client tests**

Cover workspace share create/list/update/revoke, shared-with-me list/detail/clone/proxy operations, token create/list/revoke, and public token preview/verify/import.

- [x] **Step 2: Run the new API test**

Run: `/Users/macbook-dev/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest Tests/tldw_api/test_sharing_client.py -q`

Expected: fail because the sharing schemas and client methods do not exist.

- [x] **Step 3: Add schemas and client methods**

Mirror the non-admin server schema from `tldw_server2/tldw_Server_API/app/api/v1/schemas/sharing_schemas.py`. Do not add admin config/audit methods.

- [x] **Step 4: Re-run the API test**

Expected: pass.

### Task 2: Remote-Only Sharing Service And Scope Seam

**Files:**
- Create: `tldw_chatbook/Sharing/server_sharing_service.py`
- Create: `tldw_chatbook/Sharing/server_sharing_scope_service.py`
- Create: `tldw_chatbook/Sharing/__init__.py`
- Modify: `tldw_chatbook/app.py`
- Test: `Tests/Sharing/test_server_sharing_service.py`

- [x] **Step 1: Write failing service/scope tests**

Cover normalized server IDs, local-mode rejection before policy dispatch, action IDs for list/inspect/create/configure/revoke/launch, and app bootstrap service construction.

- [x] **Step 2: Run the service test**

Run: `/Users/macbook-dev/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest Tests/Sharing/test_server_sharing_service.py -q`

Expected: fail because the Sharing package does not exist.

- [x] **Step 3: Implement service and scope seam**

Use fixed runtime-policy actions:
- `sharing.links.list.server` for listing shares, shared-with-me, tokens, and shared sources.
- `sharing.links.inspect.server` for share details, shared media, public preview, password verify, and public import.
- `sharing.links.create.server` for share-token creation.
- `sharing.permissions.configure.server` for workspace share create/update.
- `sharing.links.revoke.server` for share and token revocation.
- `sharing.links.launch.server` for shared workspace clone and shared workspace chat.

- [x] **Step 4: Wire app bootstrap**

Attach `server_sharing_service` and `server_sharing_scope_service` beside other remote-only server services.

- [x] **Step 5: Re-run the service test**

Expected: pass.

### Task 3: Tools & Settings Sharing Panel

**Files:**
- Create: `tldw_chatbook/UI/Sharing_Panel.py`
- Modify: `tldw_chatbook/UI/Tools_Settings_Window.py`
- Test: `Tests/UI/test_tools_settings_window.py`

- [x] **Step 1: Write failing UI tests**

Cover a `Sharing` navigation button, local-mode unavailable guidance, server-mode enabled controls, workspace share creation/listing, token creation/listing, shared-with-me listing, clone/chat launch, and JSON status rendering.

- [x] **Step 2: Run the UI test**

Run: `/Users/macbook-dev/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest Tests/UI/test_tools_settings_window.py -q`

Expected: fail because the panel and nav target do not exist.

- [x] **Step 3: Add lightweight panel and navigation**

Use explicit inputs for common IDs and enums, JSON text areas for payload-heavy operations, and route every operation through `server_sharing_scope_service`. Do not invent local sharing storage.

- [x] **Step 4: Re-run the UI test**

Expected: pass.

### Task 4: Docs And Verification

**Files:**
- Modify: `Docs/Parity/2026-04-21-gap-ledger.md`
- Modify: `Docs/Parity/2026-04-21-execution-roadmap.md`
- Modify: `Docs/Parity/2026-04-21-capability-matrix.md`

- [x] **Step 1: Update parity status**

Record Sharing as a first remote-only control surfaced in Chatbook, with remaining gaps limited to richer UX, deeper shared-resource rendering, and any future local import/sync design.

- [x] **Step 2: Run focused verification**

Run:

```bash
/Users/macbook-dev/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest Tests/tldw_api/test_sharing_client.py Tests/Sharing/test_server_sharing_service.py Tests/UI/test_tools_settings_window.py Tests/UI/test_screen_navigation.py Tests/RuntimePolicy/test_runtime_policy_core.py -q
git diff --check
/Users/macbook-dev/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m compileall tldw_chatbook/Sharing tldw_chatbook/UI/Sharing_Panel.py tldw_chatbook/UI/Tools_Settings_Window.py tldw_chatbook/tldw_api Tests/Sharing Tests/UI/test_tools_settings_window.py Tests/tldw_api/test_sharing_client.py -q
```

Expected: all commands pass before committing.
