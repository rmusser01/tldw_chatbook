# MCP root draft and save lifetime

TASK-32791 repairs the existing explicit Save root workflow. The root applies to
local MCP serving and Hub tool tests; blank uses that serving process's current
folder. Console file tools remain governed by Chat scratch and admitted Workspace
folders. UI copy, generated config guidance and the resolver docstring must agree.

The application owns a narrow FIFO root-save coordinator. Capture the selected
configuration path, current working directory and exact draft identity/revision
before dispatch. Validate with the existing shared validator, then call the atomic
configuration mutation under its identity precondition. UI cancellation cancels
only observation. The latest compact receipt survives MCP screen recreation;
normal shutdown closes admission and drains writes before dependent teardown.
No runtime permission, folder-binding, exposure or provider authority changes.

Keep editing available during a save. Read-only refresh and source/mode changes
preserve dirty input and its receipt. A completed save canonicalizes only its
unchanged originating draft; newer edits, including A→B→A, remain untouched.
Validation and persistence failure keep retryable drafts. Root and master-toggle
receipts are separate. Saved-to-file/cache-refresh-failed remains a warning,
including after a no-op Save. Scope receipts and known committed values to the
selected config path, publication generation and file revision; later independent settings
publication supersedes an old root override and canonicalization. Recreated-view
receipts must not claim to retain a discarded draft. Never report an old-profile
receipt under a new profile.
Publish persistence before best-effort catalog refresh.

Use existing token-backed text/height/error classes and the scrollable Tools
canvas. No fixed visual values or new dependencies are needed. Targeted tests must
hold real observer/thread boundaries, cover overlap, newer drafts, recreation,
shutdown, cache publication and config selection, and verify native compact/wide
dark/light validation, retry and real private persistence. No full-suite or
provider request is authorized or required for this slice. Master-toggle write
ordering remains a separate existing defect to review next.

Governance: [ADR-168](../../../backlog/decisions/168-mcp-root-save-lifetime.md),
ADR-033, ADR-082/102, ADR-150/161. The wider migration remains open.
