# Task5 fenced archive follow-up evidence

Base: `be3bd821b2936d15ea9c44f1b806289fa2a90326`. Approved requirements: `.superpowers/sdd/2026-09-27-agent-archive-recovery-capture-on/task-5-brief.md`, including root's subsequent dispatch-correlation ruling; ADR166/design metadata remains root-owned.

The controller uses the existing supported execution-key set for its private archive-reference rider. The only runtime change passes its existing per-call dispatch correlation to restore. The ordinary fence still has no call_id; native capability remains false/server_default. Parser, correlation generation, bridge native/fenced selection, wire schema and nonempty recovery-key guard remain unchanged.

`captured_followup` now has fenced-structured Direct/RAG cases through real gateway/parser/controller/runtime/capture/SQLite, with only the documented injected HTTPX client transport replaced. Every external request asserts no native tools/tool_choice schema; every response has ordinary reasoning_content and fenced content with no native tool_calls. DNS is denied/recorded and conftest's import-time socket guard remains enabled. Final cases report no DNS or socket attempts. Config/home/data/keyring/offline isolation loads from Tests/conftest before production imports; no direct production probe imports occurred.

## TDD phases

Each phase's `*-command.json` retains raw argv, cwd, environment, interpreter, exit, source hashes and uncompressed log hash. Large logs/diagnostic JSON are gzip-compressed byte-exact; inventory.json carries compressed and decompressed hashes. Invalid phases are preserved and do not establish production correctness.

| Phase | Exit/outcome | Meaning |
|---|---|---|
| red-fenced | 1; 2 failed | Invalid external seam: no-schema llama.cpp bypasses chat_api_call_fn for native HTTPX completion, which network denial refused. The default pytest root also emitted unrelated historical garbage-directory cleanup warnings; no cleanup of those directories was performed. |
| red-fenced-diagnosis | 1; 2 failed | Same invalid seam, with fresh named basetemp and diagnostic assertion. |
| red-fenced-client | 1; 2 failed,57 deselected | Valid RED: injected HTTPX MockTransport, real fenced search and terminal revision/verified_equal, then natural follow-up lacks confirmation; no trace failures. |
| green-fenced | 1; 2 failed | Predicate repair exposes nonempty identity guard: metadata is carried, restore still lacks a raw native call ID. |
| green-fenced-diagnosis | 1; 1 failed | Same sufficient blocker with context; selected read-only SQLite evidence is in identity-refusal-evidence.json. Root ruled before runtime code expansion. |
| green-correlation | 1; 2 failed | Exact confirmation reached after approved dispatch fix. Old current-mode assertion rejects actual server_default retained history; old test cleanup uses nonexistent async stop_active_run(session_id=...). |
| green-correlation-policy | 0; 2 passed,57 deselected in13.03s | Exact retained fenced historical blocks asserted; old current-mode omission assertions retained. Test uses actual synchronous Stop and drains its accepted task. |
| green-followup-file | 0; 59 passed in116.21s | Approved maximum targeted selection; no pytest warnings. |

The actual frozen replay semantics are `console_prepared_request.py`'s keep_old = mode not in {current,off}; server_default retains groups. The natural-flow test now checks that exact one historical group and literal blocks, and empty active groups. Its finally uses synchronous `stop_active_run(record_user_stop=True)` (controller API), followed by gather of held tasks. No production cleanup or replay change.

## Verified boundaries

Natural second-result metadata reaches exact target/chat_only confirmation; only that conversation restores after decide(allow=True), with current session/workspace preserved. RecoveryCallKey is nonempty and equals actual SQLite tool_proposed/tool_execution_started identity: turn-2-call-0 for no-ID fences, restore-1 for prior native responses. First/follow-up terminal links are revision/verified_equal; exact prior call/header/link/request/response reconstruction, saved text/thinking and current-user source pins remain. Ordinary following capture completes.

The full follow-up file preserves native/plain/tagged/structured positives and LibraryBlocked/frozenBlocked, PersonaDenied/ask, kill/unreadable switch, global tool flag false, unsupported/unknown destination, temporary/remote/legacy, changed destination, expired/manual/imported/foreign references and refusal/cancellation controls.

`scope-final` exits0: controller191/runtime20 existing Ruff diagnostic signatures are unchanged; both retain whole-file formatting debt. Changed predicate/dispatch snippets format; owned test and QA scripts lint/format clean; scoped git whitespace passes. scope-static-summary.json records exact unchanged gateway/bridge/native_tools/Console runtime/factory/service/profile/archive-runtime bytes and five strict service-guard hashes. Agent runtime equals its baseline with only the authorized restore dispatch argument/format replacement. No other source changes.

Earlier static/stdin indentation probes and QA-script authoring lint/syntax diagnostics were corrected without production semantics changes. Raw static failure phases and qa-lint are retained; the intermediate shell-only QA authoring diagnostics remain in the tool transcript. Final qa-lint-final/qa-format-final/whitespace/scope-final exit0.

No app, model request, live server, normal profile, native capability/replay override, full sweep, push, PR, merge or original-checkout write. Root retains native qualification and budget22consumed/30cap/8remaining; these deterministic transport tests do not qualify actual model behavior or resolve optional staged-RAG capture and earlier excluded live issues.
