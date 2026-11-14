## Assessment

**Ready to merge: With fixes.**
**Spec compliance: Issues found.**

Reviewed `9ba96ebb626dd010f14d093b0e8d40f37c715d32..459e666970ef9e7aa4e148705cddafb9621a72d1`. Three Important issues and one Minor issue remain. No Critical issue found.

## Strengths

- The two-store acceptance design is implemented coherently: exact attempt identity, ledger ownership cutoff, conversation receipt, and revision-specific draft consumption precede dispatch.
- Automatic descendants use the original allowance root while retaining conversation-local run parents. The new native-start → child → wake test exercises the combined runtime path and shared generation, call, token, and deadline limits.
- Physical worker ownership survives cancellation until the underlying execution drains. Manual Send can withdraw prepared work, while accepted targets remain independent of the source.
- Machine-origin requests preserve provenance, literal slash/`@` text, and profile-authority exclusions. Durable draft edits and clears are revision-fenced.
- Saved launch history is separated from current activity. The final manual-recovery fix preserves historical refusal information without leaving successfully recovered chats falsely blocked.

## Issues

### Critical

None found.

### Important

#### 1. Reject archived destinations at creation and launch boundaries

**Locations:** [creation record validation](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/tldw_chatbook/Chat/console_chat_controller.py:16940), [workspace validation](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/tldw_chatbook/Chat/chat_persistence_service.py:2367), [start acceptance](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/tldw_chatbook/Chat/console_chat_start.py:403).

`validate_workspace_target()` only checks whether the workspace exists. Archived records still satisfy that check. The approved-token revalidation checks source ownership and source workspace identity, but does not reject an archived destination. Native acceptance also lacks a destination-availability check.

A real SQLite probe prepared and approved a same-workspace request, archived that workspace, then executed the approved request:

```text
archived_after_approval {"ok": true, "launch_status": "draft", "destination_archived": true, "saved_scope": "workspace", "saved_workspace": "archivable"}
```

The destination is therefore mutated after becoming unavailable. The same missing availability check exists before native launch.

**Required fix:** Validate the exact captured destination as available and unarchived immediately before creation and again before native acceptance. Preserve the approved destination; do not retarget. Add focused cases for archival during approval and during asynchronous start preparation.

#### 2. Disabling the runtime during preparation does not prevent acceptance or dispatch

**Locations:** [initial gate](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/tldw_chatbook/Chat/console_chat_start.py:202), [acceptance boundary](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/tldw_chatbook/Chat/console_chat_start.py:403), [frozen dispatch setting](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/tldw_chatbook/Chat/console_chat_controller.py:24965).

`start()` checks `_agent_runtime_enabled` before asynchronous preparation. `accept()` never checks it again. Dispatch subsequently uses the earlier `turn_context.tool_configuration` value.

The focused barrier probe paused readiness, called the actual `update_agent_runtime(enabled=False, bridge=...)`, then resumed preparation:

```text
runtime_disabled_before_acceptance {"status": "started", "reason": null, "runtime_enabled": false, "provider_calls": 1, "handoff_state": "consumed", "generation_used": 1}
```

The live gate was disabled before the ownership cutoff, yet the draft was consumed and provider work ran. This violates the requirement to recheck current policies at acceptance and preserve the draft when the destination runtime is disabled.

**Required fix:** Recheck current runtime/start eligibility at the acceptance boundary. A disabled gate should refuse before acceptance, preserve the draft, and settle the uncommitted reservation. Cover the configuration change with a readiness-barrier regression.

#### 3. The approval card does not disclose remembered-body authority or identify an instructions override

**Locations:** [remember button](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/tldw_chatbook/Widgets/Chat_Widgets/chat_create_confirm_card.py:48), [approval body](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/tldw_chatbook/Widgets/Chat_Widgets/chat_create_confirm_card.py:128), [instructions label](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/tldw_chatbook/Widgets/Chat_Widgets/chat_create_confirm_card.py:155).

The card says “Allow for this session” without explaining that the grant covers later requests in the same mode and destination, including later supplied opening prompts and instructions. It also renders explicit instructions as generic “System prompt,” without naming the override.

The backend grant scoping is sound, but the required disclosure is absent. With `mode=start`, remembering approval permits later supplied bodies to run without another card; that makes the omission material.

**Required fix:** State the exact remembered scope and coverage of later supplied bodies. Label nonblank tool instructions as an explicit override, while retaining the complete body and markup-disabled rendering. Verify the actual card text.

### Minor

#### 4. Unavailable-Persona fallback notices are discarded

**Locations:** [prepared payload](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/tldw_chatbook/Chat/console_chat_controller.py:16902), [restored assistant selection](/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/tldw_chatbook/Chat/console_chat_controller.py:17082).

The canonical resolver returns `startup.notice`, but creation omits it from both the approval payload and restored session. Explicitly supplying the resolved assistant also bypasses the ordinary notice-producing defaults path.

The corrected probe installed the ordinary notice callback and inspected the actual session field:

```text
degraded_default {"resolver_notice": "Workspace default Persona unavailable (persona_deleted). Started with None.", "approval_has_notice": false, "notice_callbacks": [], "target_assistant_default_notice": "", "assistant_id": "console"}
```

The plain fallback is correct, but the user receives no explanation that the workspace Persona was unavailable.

**Required fix:** Carry the canonical notice through the approval/creation presentation path and ordinary session notice mechanism. Add a missing-Persona case.

## Deferred observations

| Observation | Evidence and disposition |
|---|---|
| Invalid-escape and unrelated temporary-directory cleanup warnings | **Inherited Minor debt.** The baseline logs identify existing `patch_tool_impls.py:32` and splash-source warnings. Owned final runs avoid unrelated global cleanup directories. No new warning-source regression established; do not claim those sources were fixed. |
| FD growth | **Unresolved test-resource debt; not an established feature regression.** Baseline recorded growth of 1,080 descriptors; later recovery runs also recorded growth. Per-test GC suppressing the final warning does not prove general cleanup. Preserve the logs and investigate separately. |
| Consolidated test-owner coverage | **Accept the documented deviation.** Important cases were consolidated into `test_console_chat_start.py` rather than every originally named test file. The initial recursive integration gap was subsequently filled with the actual native-start → child → wake path. File placement is not a remaining defect; the newly identified boundary cases still need coverage. |
| Two `reasoning_replay` token-budget mock failures | **Inherited Minor fixture defect.** Supplied isolated FIX_BASE reproduction reports `2 failed, 38 deselected`; its diagnostics identify lambdas rejecting the existing keyword. The relevant test file and `agent_service.py` also have identical blobs at overall BASE and HEAD. This is stronger evidence than merely observing unchanged files. The affected tests remain failing; do not describe the related group as entirely green. |
| Ctrl+Q inside the switcher | **Existing modal-routing behavior; separate usability follow-up.** The branch changes only four metadata-rendering lines in the switcher. Its bindings, dismissal mixin, and app binding are unchanged. The app’s Ctrl+Q binding is non-priority; installed Textual excludes app bindings from a modal’s non-priority binding chain. This explains the captured Escape-then-Ctrl+Q behavior. No baseline live replay was run. |
| User-guide conflict markers | **Inherited Minor documentation defect.** The three markers are present at immutable overall BASE lines 470, 1249, and 1455. The new section does not resolve them. Separate cleanup is appropriate; the guide should not be described as wholly clean. |
| Git GC/unreachable-object warnings | **Repository housekeeping, outside feature scope.** No evidence of a feature regression; no pruning or shared Git mutation warranted. |
| Disk-full interruption during a verification run | **Invalid infrastructure run, superseded by recorded passing reruns.** It should remain qualified as an interrupted run rather than product failure or passing evidence. |
| PTY evidence, streaming usage uncertainty, clean restarts | **Valid limitations.** The record contains real PTY/local-provider evidence, not native screenshots. Unknown streaming usage is conservatively retained; confirmed non-streaming settlement is separately evidenced. Clean shutdown/reopen does not establish OS hard-kill or power-loss behavior. |

## Checks performed

- Reviewed the complete package in passes across production changes, tests, migrations, documentation, and combined control flow.
- Verified the package’s diff body exactly matches `git diff -U10 BASE HEAD`—528,650 bytes beneath its wrapper.
- Audited all 29 verification-log manifest entries against recorded sizes and SHA-256 hashes.
- Inspected the actual final `163 passed in 127.97s` log and postcommit lint, formatting, ratchet, and diff-check output.
- Inspected representative PTY frames and database receipts, including literal machine text, edited/cleared drafts, Stop, refusal without replay, and successful manual recovery.
- Ran only the isolated probes below. No supplied suite was rerun, no helper agent was spawned, and no checkout/index/HEAD mutation was made.

The controller’s reopened Backlog task and untracked QA directory are external to this review’s mutations.

## Exact focused probes

All commands ran from:

```text
/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook
```

### A. Archive after approval

This command also contained the initial fallback diagnostic. Its `assistant_startup_notice` lookup was not the correct session field; probe C below corrects that diagnostic.

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python - <<'PY'
import os, json, tempfile
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
from loguru import logger
logger.remove()
with tempfile.TemporaryDirectory(prefix='final-chat-start-review-') as directory:
    root = Path(directory)
    (root/'data').mkdir()
    config = root/'config.toml'
    config.write_text('[paths]\ndata_dir = '+json.dumps(str(root/'data'))+'\n')
    os.environ.update(TLDW_CONFIG_PATH=str(config), XDG_CONFIG_HOME=str(root/'config'), XDG_DATA_HOME=str(root/'data'), TLDW_TEST_MODE='1', PYTHON_KEYRING_BACKEND='keyring.backends.null.Keyring', HF_HUB_OFFLINE='1', TLDW_AGENTS_RUN_LOG_ENABLED='false')
    from Tests.Chat.test_console_chat_start import handoff_store, creation_controller, _prepared_creation
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService
    from tldw_chatbook.Workspaces.models import WorkspaceAssistantDefaults
    handoff = handoff_store.__wrapped__(root)
    store_db = next(handoff)
    fixture = creation_controller.__wrapped__(store_db)
    controller, source, db = next(fixture)
    workspaces = WorkspaceDB(root/'workspaces.sqlite', client_id='review')
    registry = LocalWorkspaceRegistryService(workspaces)
    registry.create_workspace(workspace_id='archivable', name='Archivable')
    controller.store.persistence.workspace_registry = registry
    controller.app.workspace_registry_service = registry
    source.workspace_id='archivable'
    notices=[]
    controller.app.notify=lambda *a,**k: notices.append(a)
    try:
        prepared=_prepared_creation(controller, source)
        controller._chat_create_session_grants[source.id]={prepared['_grant_scope']}
        assert controller.request_chat_create_confirm(prepared,session_id=source.id)['allow']
        registry.archive_workspace('archivable')
        result=controller.execute_agent_chat_create(prepared)
        saved=db.get_conversation_by_id(result['conversation_id']) if result.get('conversation_id') else None
        print('archived_after_approval', json.dumps({'ok':result['ok'],'launch_status':result.get('launch_status'),'destination_archived':registry.get_workspace('archivable').archived,'saved_scope':saved['scope_type'] if saved else None,'saved_workspace':saved['workspace_id'] if saved else None}))
        registry.create_workspace(workspace_id='degraded',name='Degraded',assistant_defaults=WorkspaceAssistantDefaults(assistant_id='missing-persona'))
        source.workspace_id='degraded'
        settings=replace(controller._default_session_settings(),system_prompt=None)
        controller._default_session_settings=lambda:settings
        controller.app.local_character_persona_service=SimpleNamespace(get_persona_profile=lambda _:None)
        prepared=_prepared_creation(controller,source)
        notice=controller._chat_creation_records[prepared['_creation_token']]['startup'].notice
        controller._chat_create_session_grants[source.id]={prepared['_grant_scope']}
        assert controller.request_chat_create_confirm(prepared,session_id=source.id)['allow']
        result=controller.execute_agent_chat_create(prepared)
        target=next(s for s in controller.store.sessions() if s.persisted_conversation_id==result['conversation_id'])
        print('degraded_default',json.dumps({'resolver_notice':notice,'approval_has_notice':any('unavailable' in str(v) for v in prepared.values()),'notifications':notices,'target_startup_notice':getattr(target,'assistant_startup_notice',None),'assistant_id':target.assistant_id}))
    finally:
        fixture.close()
        handoff.close()
        workspaces.close()
PY
```

Exit 0:

```text
archived_after_approval {"ok": true, "launch_status": "draft", "destination_archived": true, "saved_scope": "workspace", "saved_workspace": "archivable"}
degraded_default {"resolver_notice": "Workspace default Persona unavailable (persona_deleted). Started with None.", "approval_has_notice": false, "notifications": [], "target_startup_notice": null, "assistant_id": "console"}
```

### B. Runtime disable before acceptance

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python - <<'PY'
import asyncio, os, json, tempfile
from pathlib import Path
from loguru import logger
logger.remove()
with tempfile.TemporaryDirectory(prefix='final-native-policy-review-') as directory:
    root=Path(directory); (root/'data').mkdir()
    config=root/'config.toml'; config.write_text('[paths]\ndata_dir = '+json.dumps(str(root/'data'))+'\n')
    os.environ.update(TLDW_CONFIG_PATH=str(config),XDG_CONFIG_HOME=str(root/'config'),XDG_DATA_HOME=str(root/'data'),TLDW_TEST_MODE='1',PYTHON_KEYRING_BACKEND='keyring.backends.null.Keyring',HF_HUB_OFFLINE='1',TLDW_AGENTS_RUN_LOG_ENABLED='false')
    from Tests.Chat.test_console_chat_start import _native_start_rig
    async def probe():
        c,s,runs,source,target,chain,request=await _native_start_rig(root)
        entered=asyncio.Event(); release=asyncio.Event(); resolve=c._resolve_for_send_bounded
        async def hold(selection):
            entered.set(); await release.wait(); return await resolve(selection)
        c._resolve_for_send_bounded=hold
        task=asyncio.create_task(c._chat_start.start(request))
        try:
            await asyncio.wait_for(entered.wait(),5)
            c.update_agent_runtime(enabled=False,bridge=c._agent_bridge)
            release.set()
            result=await asyncio.wait_for(task,10)
            await asyncio.gather(*c._chat_start.tasks())
            print('runtime_disabled_before_acceptance',json.dumps({'status':result.launch_status,'reason':result.reason,'runtime_enabled':c._agent_runtime_enabled,'provider_calls':c.provider_gateway.parent_calls,'handoff_state':target.agent_handoff_state,'generation_used':runs.automatic_work.snapshot(chain).used['generation']}))
        finally:
            release.set(); await c.shutdown();runs.close();s.persistence.db.close_connection()
    asyncio.run(probe())
PY
```

Exit 0:

```text
runtime_disabled_before_acceptance {"status": "started", "reason": null, "runtime_enabled": false, "provider_calls": 1, "handoff_state": "consumed", "generation_used": 1}
```

An earlier setup attempt called a nonexistent setter and was discarded. The result above uses the actual runtime-update method.

### C. Correct notice field and approval rendering

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python - <<'PY'
import json, os, tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from loguru import logger
logger.remove()
with tempfile.TemporaryDirectory(prefix='final-notice-review-') as directory:
    root = Path(directory)
    (root / 'data').mkdir()
    config = root / 'config.toml'
    config.write_text('[paths]\ndata_dir = ' + json.dumps(str(root / 'data')) + '\n')
    os.environ.update(TLDW_CONFIG_PATH=str(config), XDG_CONFIG_HOME=str(root / 'config'), XDG_DATA_HOME=str(root / 'data'), TLDW_TEST_MODE='1', PYTHON_KEYRING_BACKEND='keyring.backends.null.Keyring', HF_HUB_OFFLINE='1', TLDW_AGENTS_RUN_LOG_ENABLED='false')
    from Tests.Chat.test_console_chat_start import handoff_store, creation_controller, _prepared_creation
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService
    from tldw_chatbook.Workspaces.models import WorkspaceAssistantDefaults
    from tldw_chatbook.Widgets.Chat_Widgets.chat_create_confirm_card import ChatCreateConfirmCard
    handoff = handoff_store.__wrapped__(root)
    fixture = creation_controller.__wrapped__(next(handoff))
    controller, source, db = next(fixture)
    workspaces = WorkspaceDB(root / 'workspaces.sqlite', client_id='review')
    registry = LocalWorkspaceRegistryService(workspaces)
    controller.store.persistence.workspace_registry = registry
    controller.app.workspace_registry_service = registry
    notices = []
    controller.store._on_assistant_default_notice = notices.append
    try:
        registry.create_workspace(workspace_id='degraded', name='Degraded', assistant_defaults=WorkspaceAssistantDefaults(assistant_id='missing-persona'))
        source.workspace_id = 'degraded'
        settings = replace(controller._default_session_settings(), system_prompt=None)
        controller._default_session_settings = lambda: settings
        controller.app.local_character_persona_service = SimpleNamespace(get_persona_profile=lambda _: None)
        prepared = _prepared_creation(controller, source)
        resolver_notice = controller._chat_creation_records[prepared['_creation_token']]['startup'].notice
        controller._chat_create_session_grants[source.id] = {prepared['_grant_scope']}
        assert controller.request_chat_create_confirm(prepared, session_id=source.id)['allow']
        result = controller.execute_agent_chat_create(prepared)
        target = next(s for s in controller.store.sessions() if s.persisted_conversation_id == result['conversation_id'])
        print('degraded_default', json.dumps({'resolver_notice': resolver_notice, 'approval_has_notice': any('unavailable' in str(v) for v in prepared.values()), 'notice_callbacks': notices, 'target_assistant_default_notice': target.assistant_default_notice, 'assistant_id': target.assistant_id}))
        card = ChatCreateConfirmCard()
        card._payload = {'tool': 'new_chat', 'title': 'Review', 'mode': 'start', 'scope_type': 'workspace', 'workspace_id': 'degraded', 'assistant': 'console', 'model': 'local-model', 'opening_prompt': 'go', 'instructions': 'custom instructions', 'resolved_instructions': 'custom instructions'}
        print('approval_body', json.dumps(card._body_text()))
    finally:
        fixture.close()
        handoff.close()
        workspaces.close()
PY
```

Exit 0:

```text
degraded_default {"resolver_notice": "Workspace default Persona unavailable (persona_deleted). Started with None.", "approval_has_notice": false, "notice_callbacks": [], "target_assistant_default_notice": "", "assistant_id": "console"}
approval_body "Destination: Workspace: degraded\n\nMode: start one bounded turn in the background\n\nAssistant: console \u00b7 Model: local-model\n\nOpening prompt (one background turn):\ngo\n\nSystem prompt:\ncustom instructions"
```

## Recommendation

Use the single final fix wave for these four findings, with focused regressions at the affected boundaries and approval surface. The underlying ownership, allowance, persistence, and recovery architecture does not need redesign. Re-review the resulting fix diff before integration.
