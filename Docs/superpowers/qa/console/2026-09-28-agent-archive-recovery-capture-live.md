# Capture-On native archive recovery: bounded live qualification

Status: **RAG-only passed; fresh Direct restore follow-up remains unqualified.** The reviewed shared-factory correction is committed as `e4a2717b4158d178d2ec1616d9a8ebd983982c08`. TASK-33164 remains In Progress with AC1 unchecked. The original feature and Task1 reviews remain completed.

## Native results

| Phase / source | Actual localhost model requests | Outcome |
| --- | --- | --- |
| RAG-only, `75456b7a2d3f8ea4f711f41c061d212abdd03add` | 1–4 | Search, natural second-result selection, exact workspace confirmation, confirmed restore and explicit Open succeeded. Four COMPLETE captured calls. |
| Initial Direct, same source | 5–6 | Search completed; natural follow-up blocked before model contact. Actual constructor traceback rejects unsupported `collections_service`; native Cancel cleared preparation. |
| Fresh Direct, `e4a2717b4158d178d2ec1616d9a8ebd983982c08` | 7–10 | Shared provider constructs successfully. Model invokes find_tools twice, load_tools, then search_archived_conversations. Native archive results appear; the next post-tool request stops at the launcher's ten-attempt ceiling. Four COMPLETE calls plus one ERROR call; no restore follow-up sent. |

All ten actual HTTP requests returned 200 and SSE DONE without JSON/SSE/transport errors or retries. The capped next call never reached the server; its generic RuntimeError is consistent with the explicit observer guard and exhausted counter, rather than an HTTP failure. No cap was reset or extended. All three apps used native Ctrl+Q and their attached PTYs exited0; final health/slots show the existing server healthy and idle.

## Verified RAG-only behavior

The native prompts were “Find recently archived chats across all workspaces.” and “Restore the second one.” The fixture contains chat-only, workspace-only and chat+workspace archival cases. Search order is Birch, **Linden planning conversation · Atlas Research**, then Maple. [Exact confirmation](2026-09-28-agent-archive-recovery-capture-live/rag/restore-confirm.txt) names Linden/Atlas, offers Restore workspace and discloses that other active chats become visible while individually archived chats remain archived.

Before explicit confirmation, all fixture rows/messages/workspace records remain unchanged. Confirmation changes only Atlas's archived/archive timestamp/revision/update timestamp fields; revision increments exactly once. Default/Vault workspace records remain exact before Open. All three original conversation rows, six saved messages, active leaves and message IDs remain exact; conversation count remains4. Restore leaves the source Default chat selected until explicit Open.

[Open](2026-09-28-agent-archive-recovery-capture-live/rag/original-opened.txt) displays the original two saved messages in Linden. Its Library agent access is Blocked; no target send occurs. [Returning to the source](2026-09-28-agent-archive-recovery-capture-live/rag/source-draft-after-open.txt) retains the unsent “Keep this source draft.”

The real read-only ConsoleTraceNativeReader reconstructs all four captured requests with no omissions. Each call_boundary event pins the exact current saved user revision, including the derived metadata rider. The two original call rows and reconstructed request hashes remain exact through pending/restore/Open. The metadata rider is present only on the two follow-up requests: one observed page/three rows/zero omitted pages. Saved source user texts remain exactly the two prompts. A pending confirmation has response_started, not COMPLETE; settlement is checked after confirmation.

[Runnable observation assertions](2026-09-28-agent-archive-recovery-capture-live/qualify.py) ran against the synthetic snapshots with the installed Python3.12 interpreter and exited0; [qualification result](2026-09-28-agent-archive-recovery-capture-live/rag/qualification.json) records the assertions. It does not execute tools or restore data.

## Environment and evidence

The app used the real isolated worktree/native runtime/SQLite/gateway/confirmation UI and the existing Gemma4 server at localhost:9099, context64000. All HOME/XDG/data/config/database paths are disposable per-case scratch. Null Keyring is selected and verified before import; HF/Transformers offline, automatic retrieval/model refresh/local filesystem tools disabled. A real deny-only PreToolUse hook blocks search_library_rag while archive tools retain their ordinary gates. MCP server configuration is disabled; existing MCP/meta schemas may still be advertised, and none was selected. No embedding/index/Chroma artifact path was found.

[Config controls](2026-09-28-agent-archive-recovery-capture-live/config-controls.json) retain selected settings and digests, without copying full config or normal-profile content. [Closure](2026-09-28-agent-archive-recovery-capture-live/closure.json) records equal normal config/chat/workspace DB/WAL/SHM fingerprints. [Raw artifact manifest](2026-09-28-agent-archive-recovery-capture-live/manifest.json) preserves54 synthetic/structural/native observations byte-for-byte, with clearly labeled selected app-log excerpts. No databases or raw HTTP message payloads were copied.

Native tmux input required an attached client. The app's server pane was pinned160×48; the host PTY client remained80×24. Text and ANSI captures are native server-pane evidence, not full physical-client screenshots. Initial setup mistakes (TERM/detached input, wrong workspace table, premature COMPLETE/source-surface assumptions) are harness failures, separated from the real constructor failure. Read-only checks use workspace_records and call_boundary source pins; derived current text need not appear as a saved revision in direct surface projection. Additional preparatory config checks initially used the system interpreter without tomllib and an incorrect mcp.server key; the installed interpreter and actual mcp.enabled setting were then verified. No model request resulted from those setup mistakes.

## Source correction, review and remaining boundary

The sole production change deletes the unsupported constructor argument. Real-service RED:2failed/1passed; covering GREEN:15passed14.92s. Test Ruff/format and source/authored whitespace pass; runtime retains its34 baseline Ruff signatures and existing formatting debt. The exact14 raw factory RED whitespace lines remain preserved. [Focused QA](2026-09-28-console-direct-library-factory.md) and [independent review](2026-09-28-console-direct-library-factory/review.md) document the contract and approve the source range with no actionable findings. No new ADR is needed; ADR097/166 remain governing contracts.

Fresh Direct's discovery consumed the four remaining requests rather than the expected two search requests. The constructor repair is verified; the full live Direct follow-up is not. The ten-attempt budget is exhausted, and a separate user request to extend it is pending. No fresh Direct restore, transcript immutability after that restore, or full Direct completion is claimed. Earlier dictionary/checkpoint/FD/provider/Next Send limits remain unchanged. No full sweep, normal-profile edit, server configuration change, download, push, PR, merge or worktree cleanup occurred.
