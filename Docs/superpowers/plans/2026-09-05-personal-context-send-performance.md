# Personal Context send performance

Goal: complete TASK-31504 without weakening per-request authorization.

ADR required: yes (amend existing decision)
ADR path: backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md
Reason: record connection lifetime and negative-cache invalidation boundaries.

## Implementation

1. Add failing regressions in `Tests/Personal_Context/` for configured operation connection counts, unchanged absent status, local/external setup invalidation (including WAL commits), failure cleanup, nested calls, and concurrent thread isolation. Preserve existing authority-race and export-concurrency tests. Use actual SQLite with call counters, not mocked authority behavior.
2. In `tldw_chatbook/Personal_Context/repository.py`, introduce bounded operation-scoped autocommit connection reuse. Keep the existing short export transaction and all live checks after it; never surround authorization with one read transaction or a write lock. Close the connection on success and exceptions, and never share it across threads. Retain hardened opens and reject path/owner replacement between operations; assess replacements during reuse so stale file identity cannot authorize context.
3. Scope reuse around authorized view construction and the Console provider-composition operation in `tldw_chatbook/Personal_Context/context_service.py` and `tldw_chatbook/Chat/console_chat_controller.py` as appropriate. Keep the double authorized-view fence because separately read identity and authority must be revalidated; document this owner note instead of changing the authorized-view contract. Live agent calls remain freshly checked.
4. In the Personal Context service, cache only proven absent state using content-free database/WAL identity and change metadata. Any signature change, error, setup/start-fresh, service replacement, or uncertain validity requires a hardened read. Never cache READY authority. Existing locked facades need no SQLite; unlock replaces the facade. Test symlink/owner replacement rejection and failed setup.
5. Amend ADR-102 and task notes with the bounded lifetime, invalidation contract and retained double-build rationale. Run targeted context/repository/export/agent/Console tests, provenance checks and affected-file static checks. Compare configured global/workspace hardened-open counts against measured 44/68 baseline. Do not claim latency gains from instrumented counts. Request spec then quality review and address findings before completion.

Verification uses the repository root `.venv/bin/python` from this isolated worktree, confirming imports resolve here. No full-suite sweep, new dependency, persistent ready-state cache, schema change, or new authorization contract.
