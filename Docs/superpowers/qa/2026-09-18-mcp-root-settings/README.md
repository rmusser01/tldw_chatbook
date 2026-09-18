# TASK-32791 — MCP root drafts and ordered saves

Root edits now survive read-only refresh and mode/source projection. Explicit
saves finish in submission order, outlive disposable UI observers, and drain at
app shutdown. Older receipts cannot overwrite newer drafts or later configuration
publications/external TOML edits. Failed drafts remain retryable while mounted;
recreated screens show saved configuration and accurately describe older failures.
Root guidance now names local MCP serving and Hub tests, with Console scratch and
admitted Workspace folders remaining separate. Blank uses the serving process cwd.

## Targeted evidence

| Scope | Evidence |
| --- | --- |
| Root owner, mounted draft/ABA/order/recreation/shutdown/copy, real atomic writes, external edits and existing round trips: 26 passed | [Root checks](root-settings.txt) |
| Adjacent app-owned lifecycle admission/watchdog: 4 passed | [Ownership](ownership.txt) |
| Tools/compact layout plus token governance: 56 passed in the earlier run | [Related checks](related-tools-and-tokens.txt) |
| Config save semantics, component governance and Tool/State column readability: 45 passed | [Config and components](config-and-components.txt) |
| Final Tools/compact rerun: 47 passed, one existing Permissions resize failure | [Final layout run](final-layout.txt) |

The Permissions failure is retained, not erased by its [passing isolated retry](permissions-recheck.txt).
A fresh detached checkout of pre-change HEAD `6c0e317ab75638cc34343b6170c7cba9c7cd1b29`
failed the identical light-theme case on its first run: a 7-cell State region was
clipped to 6 cells. [Baseline evidence](permissions-baseline.txt) proves this
predates the root changes. It remains an open Permissions reflow repair in the
[MCP ledger](../../reports/2026-09-18-mcp-review.md). The 131 distinct selected cases
have passing evidence across these runs; this is not an all-green final sweep or
whole-destination completion claim. No full repository suite was run.

[Independent review](independent-review.txt) covers ordering, cancellation,
shutdown, draft ownership, stale receipt projection and external edits.
[Red evidence](red.txt) records the original four draft/write regressions.
[Preflight](preflight.txt) records clean new-file lint/format, no new inherited-file
lint findings, scoped formatting, backlog IDs and diagnostic inventory.
[Diagnostic review](diagnostic-review.txt) records removal of the obsolete root
exception logger; bounded receipts retain no exception bodies.

## Native visual review

Final run003 used a fresh private profile, real LinuxDriver/TTY, real navigation,
keyboard editing, shared validation and atomic config persistence. Each dark/light
80x24 and 170x48 journey rejected a missing directory, preserved it through refresh,
retried a valid root and saved blank. A held real write then completed beneath a
newer draft; both the draft and truthful receipt survived mode navigation.
No tool or provider request was executed.

| Theme / size | Validation | Saved |
| --- | --- | --- |
| Dark 80x24 | [View](textual-dark-80x24-invalid.svg) | [View](textual-dark-80x24-saved.svg) |
| Dark 170x48 | [View](textual-dark-170x48-invalid.svg) | [View](textual-dark-170x48-saved.svg) |
| Light 80x24 | [View](textual-light-80x24-invalid.svg) | [View](textual-light-80x24-saved.svg) |
| Light 170x48 | [View](textual-light-170x48-invalid.svg) | [View](textual-light-170x48-saved.svg) |

[Pending newer draft](pending-newer-draft.svg) and [completed older save](saved-newer-draft.svg)
show the separate draft and save state. All ten final SVG captures were rendered
and visually inspected. Receipts fit beneath Save at compact size; the scrollable
help remains available below them. The first native inspection found receipts
below the fold, prompting this placement repair and a compositor assertion.

[Native source receipt](native-result.json) pins the runner and 32 production files;
[capture hashes](capture-manifest.json) track the original and normalized screenshots.
[Lifecycle](lifecycle.json) proves app.run returned, exit0, PID gone before terminal
closure, lock reacquisition, ten healthy databases, zero conversations/messages,
unchanged default-profile fingerprints, and no error/faulthandler output.
Runs001/002 passed earlier functional journeys and exited cleanly; their retained
pre-final receipts are historical and do not qualify final source.

[ADR-168](../../../../backlog/decisions/168-mcp-root-save-lifetime.md) records the
narrow app-owned lifetime and config identity fences. Existing ADR-033/082/102/150/161
retain authority. Draft PR2707 still requires separate visual review and merge
approval. Permissions resize, the local-tools master toggle, remaining MCP
runtime workflows and the broader component migration remain open. The pre-save
PR head also has an unresolved Windows GGUF SelectOverlay failure; no CI-green
claim is made here.
