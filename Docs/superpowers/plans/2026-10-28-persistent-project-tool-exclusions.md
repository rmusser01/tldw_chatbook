# Persistent project tool exclusions investigation

Task: TASK-37234; continuation of the Fedora report and PR #3284.

ADR required: no for investigation and interpreter qualification.
ADR path: backlog/decisions/104-token-budgeted-agent-tool-disclosure.md; backlog/decisions/032-local-agent-tool-permission-boundary.md; backlog/decisions/069-console-project-instruction-local-state-and-preflight.md
Reason: preserve current disclosure, permission and folder-authority boundaries. Any new boundary decision requires an ADR before implementation.

1. Confirm the reporter's tested commit and exact current error. GitHub still shows PR #3284 open; a local integration is possible but not yet identified.
2. Distinguish excluded/unknown tools from unloaded schemas using the latest transcript and per-run discovery metadata, then trace configuration, provider composition, persona narrowing and admitted folders.
3. Run the existing recovery and real-command controls under Linux Python 3.14 as well as 3.12. The original log records Python 3.14.8, while earlier Linux evidence was 3.12.15.
4. Reproduce any capture-default mismatch through a real captured configuration and provider consumer. Three capture sites use default false while the catalog/provider default constant is true. This is a candidate defect, not a confirmed incident cause: normal CLI bootstrap also merges defaults.
5. Repair only verified causes, preserve explicit disable/denial/folder restrictions, and record targeted evidence and remaining uncertainty. Do not claim the reporter's project works from scripted discovery tests alone.

Reporter commit, new refusal text, Context Next Send folder/tool metadata and local-tools master state have been requested; investigation continues independently.

Latest report: Local tools are on and the error is now Tool schema not loaded.
That guard category means the named tool is allowed and registered, so the
capture-default mismatch does not explain that case. The new supplied logs are
ordinary application logs and still omit the four agent steps' tool names and
results. A read-only saved-run SQLite extraction was validated against a local
recovery run and supplied to the reporter; actual find/load results remain needed.

A real Console project planner probe (reported unknown qwen model, llama_cpp,
native/fenced calls, verified/unverified 32k window) exposed find_tools and
load_tools in all four cases. That minimal planner scenario does not reproduce
the failure. Its disposable probe was removed; no permission/prompt fix is inferred.

Linux run 40746120405 passed 142 cases on Python 3.12. Python 3.14.9 passed 141
and failed the POSIX exited-leader descendant control before its PID marker
appeared; the control allowed only 600 ms for complete isolated leader startup.
The qualification fixture now waits for actual descendant readiness (and leader
exit when requested), cancels through the public execution, checks empty custody
and budgets, and independently checks absence of the OS process group. A deliberate
800 ms startup control exceeds the old allowance. Existing timeout-specific
controls remain in the execution suite. No production cleanup rule is changed.

Matching-family qualification is now green on source head 94f11ed026: run
40748792976 retained 144 tests with zero failures/errors/skips on each of Python
3.12.15 (45.478s XML) and Python 3.14.9 (49.296s XML). The reporter has become
unavailable again; the saved tool sequence and tested git line remain pending.
The original incident is open.

Added ordinary-message diagnostics so the next normal app log can distinguish
missing discovery, no load attempt, invalid IDs, budget omissions and replacement.
Five message-only/privacy tests first failed solely for absent diagnostics. The
initial diagnostic + recovery selection then passed 16 cases (160.50s); a private
query/catalog variant passed five (101.68s), with final explicit privacy sentinels
also selected in the two-version CI job. No execution or permission rule changed.
Independent review found no source privacy/bounds/semantic issue. Its inventory
check detected exactly the expected three added AgentService calls and changed
bridge call; no unrelated owner or sink-topology drift. Regenerate the canonical
inventory before publishing these source edits. TASK-37234 remains In Progress.

Canonical inventory regeneration completed; the resulting diff matches exactly the two reviewed owners and aggregate call count, with no sink-topology changes.
