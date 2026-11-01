# Independent retained interpretation witnesses

These four package inputs were authored separately from the parsers. Each
`expected.json` contains observations from the literal public `inspect_package`
implementation at **both** source commits:

- `1fac8f83291f5b48c3daa7b9095253a23c05d44d` — pinned current-dev history.
- `7b4d5c501ba8500267d8f9f84f524742c33ec882` — published original PR history.

Only `package/` is captured. Expected output and provenance are its siblings,
so expected bytes cannot recursively enter their own content digest. Every
package member is a regular non-executable file with mode `0644`. `guard.py`
raises if executed; inspection must never launch it.

| Package | Observed historical distinction |
| --- | --- |
| `portable_common` | Both `chatbook-portable/1` implementations produce identical complete selected semantics and coalesce. |
| `portable_declared_value` | Dev retains an invalid typed hook environment plus required-variable/constraint blockers. The published implementation supports that declaration and leaves the independent command clear. Both meanings keep the same `/1` label. |
| `openai_guard` | Dev's dated OpenAI adapter keeps the hook unsupported. Published `chatbook-codex/1` adapts it and retains its dependent skill's guard blocker. |
| `cursor_guard` | Dev's dated Cursor adapter keeps the hook unsupported. Published `chatbook-cursor/1` adapts it and retains its dependent skill's guard blocker. |

The portable declared-value package uses the existing contained `working/`
directory as its cwd. Workspace placeholders occur in argv/environment only.
The negative observation changes only cwd to `${CHATBOOK_WORKSPACE_DATA}`;
both original parsers retain an invalid required hook and constraint blockers.
It is recorded under that case's `templated_cwd_refusal` key.

## Capture provenance

Captures ran serially on actual macOS, fixed CPython 3.12.11, in separate pipless
virtual environments with `python -I -B -u`. Each source tree was extracted by
`git archive --format=tar.gz <exact-commit> tldw_chatbook packages`. Every file's
Git blob identity and executable mode were verified against `git ls-tree -rz`;
source files were then made read-only without changing executable bits.

Only that literal tree, its literal `packages/tldw_profile_core/src`, and the
already installed dependency sites entered the child through one plain-path
`.pth`. No editable-site hooks were processed. **Every loaded product module**
was required to originate in the literal tree and match its pinned blob, SHA256,
and executable mode. Neither current product code nor a current selector or
historical-recipe implementation generated these expectations.

Before any product import, the actual profile/network guard files frozen from
integration source `18db258722fc135f590b5ade80f3bfd8200f903e` were installed by
file path. A deliberate blocked connection proved refusal before DNS. Each
child used a private HOME/XDG/temp/data profile, a complete configuration from
its literal source, offline flags, a null keyring, and a small explicit
environment without inherited API/auth tokens, SSH-agent or user Git settings.
No host/network exemption, package execution, dependency installation, or
fabricated trust record was used.

Raw commands, stdout/stderr, exit codes, complete observations, all module
origins and hashes, dependency metadata, fixture/source manifests, guard and
interpreter hashes are retained in the integration evidence workspace under
`qualification-repair01/<exact-commit>/capture01/`. Both captures exited zero,
with all inputs and dependency metadata unchanged and no unexpected guard
attempts. Each golden carries the raw artifact digests and defining module
hashes so that its provenance remains explicit outside that private workspace.

The selected-semantics projection was computed directly from the original
model JSON: remove acquisition path/evidence fields (`source_identity`,
`source_digest`, `materialized_identity`, `link_targets`, `diagnostics`,
`candidates`) and per-component `evidence`/`availability`. All other fields,
including complete definitions, constraints, dependencies, support, selections,
variable declarations and digests remain exact. Definition and variable hashes
use the literal original trust module's canonical JSON functions. Expected
files were generated only after both children exited. The sole replacement in
the full original inspection is `source_identity: "@PACKAGE_ROOT@"`; tests
replace that pathname with their copied package root.

## Current qualification

`test_retained_inspection.py` compares current retained candidates and exact
selection against these independent projections, then uses the real private
runtime owner, SQLite registry, authenticated authority store, materialization,
registry-loss recovery, trust/configuration and update/rollback paths. Negative
witnesses cover changed bytes/definitions, incomplete or duplicate component
sets, unknown/wrong identities, genuinely different structural interpretations,
and templated cwd. No test manufactures a signed authority grant. Fresh `/2`
observations remain separate from these expected historical meanings.
