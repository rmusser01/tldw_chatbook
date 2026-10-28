# Persistent project tool exclusions investigation

Task: TASK-37234; continuation of the Fedora report and PR #3284.

ADR required: no for investigation and interpreter qualification.
ADR path: backlog/decisions/104-token-budgeted-agent-tool-disclosure.md; backlog/decisions/032-local-tool-permission-adapter.md; backlog/decisions/069-console-project-instruction-local-state-and-preflight.md
Reason: preserve current disclosure, permission and folder-authority boundaries. Any new boundary decision requires an ADR before implementation.

1. Confirm the reporter's tested commit and exact current error. GitHub still shows PR #3284 open; a local integration is possible but not yet identified.
2. Distinguish excluded/unknown tools from unloaded schemas using the latest transcript and per-run discovery metadata, then trace configuration, provider composition, persona narrowing and admitted folders.
3. Run the existing recovery and real-command controls under Linux Python 3.14 as well as 3.12. The original log records Python 3.14.8, while earlier Linux evidence was 3.12.15.
4. Reproduce any capture-default mismatch through a real captured configuration and provider consumer. Three capture sites use default false while the catalog/provider default constant is true. This is a candidate defect, not a confirmed incident cause: normal CLI bootstrap also merges defaults.
5. Repair only verified causes, preserve explicit disable/denial/folder restrictions, and record targeted evidence and remaining uncertainty. Do not claim the reporter's project works from scripted discovery tests alone.

Reporter commit, new refusal text, Context Next Send folder/tool metadata and local-tools master state have been requested; investigation continues independently.
