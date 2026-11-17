# Task62: original backend tool-count presentation

Artifact-only implementation of `task62-backend-tool-counts-contract.md`
(SHA256 `0ca57db588c212ec3231282f2b1e2d4ad652bbac8e80f3d90dfb63da74b6c17e`).
No live product/test changes, imports, Python execution, native work or tests.
Central owns endpoint initialization, composition, closed wire schema and adoption;
root owns execution.

ADR required: no new ADR. This completes the existing original presentation
boundary under ADR126's complete Console service owner and the approved Task62
contract. It adds no tool policy, permission or command authority.

## Proposed targets and integration

- New `tldw_chatbook/Chat/console_backend_tool_counts.py`.
- Exactly `_publish_mcp_inspector_counts` and `_publish_selected_tool_count` in
  `tldw_chatbook/Chat/console_chat_controller.py`. Existing local-App branches are
  unchanged; only the previously empty `App is None` paths select the backend.
- One new focused test: `Tests/Chat/test_console_backend_tool_counts.py`.

All are under `proposed/` in this draft. The existing controller has a byte-exact
preimage; both new live targets were absent. Nine implementation/test dependency
snapshots plus the contract are retained locally and pinned in the manifest.

Central should import the helper **inside** `ConsoleBackendPresentation.__init__`
and set `self.tool_counts = ConsoleBackendToolCounts(self)` once. Construction is
inert and needs no completed runtime graph. `ConsoleExecutionBackend.start`
already imports the controller before creating the endpoint, so this does not
need lazy first-publication source capture. Avoid importing the helper at the
endpoint module's top level, since the helper names the endpoint's actual class.

At composition, qualify the helper module's `_RESOLVER` definition, obtain
`backend_tool_counts(controller)`, then call `snapshot(controller, session_id)`.
The two original controller call sites use this same resolver identity/code
qualification before any returned object's method can be invoked. The resolver
checks the exact concrete endpoint and installed helper defining methods.
Central's independent endpoint/protocol files are not edited by this artifact.

## Behavior

`ConsoleBackendToolCountSnapshot` is a frozen named record containing exactly
`published`, `builtin_count`, `mcp_count`, `not_connected_count`, and
`selected_count`. Before any MCP publication, `published=False` and the three
MCP/builtin fields are None. A later real `(None, None)` MCP publication has
`published=True`; it is not confused with absence. Actual selected zero remains
zero. Parent presentation can apply its unchanged absent-attribute conventions.

`publish_mcp` calls the original `offered_builtin_tool_count` once with the exact
actual controller bridge/session/composed-switch inputs. It computes before
mutating retained presentation values, rechecks the original graph, bridge and
session/incarnation, removes that session's selected count, then stores the
original builtin/MCP/not-connected values together. It does not compose a new
catalog or call a second count producer. Any work performed by the original
count function remains its original work. `snapshot` never calls it.

`publish_selected` stores the actual original selected count with its session
incarnation. Snapshots prune entries only when their session is absent from the
original store, under its existing preparation lock. A mismatching incarnation
cannot publish an old selected count into a replacement session. Only bounded
presentation values/IDs are retained; no plan, native owner or session graph is
cached in that map. MCP counts preserve the original App-global semantics.

Every publication/capture checks the exact helper/controller/endpoint/backend
defining methods, original surface functions, actual endpoint association,
owner loop, original backend source checks, controller/runtime/domain/store and
bridge association, and actual session membership. MCP computation holds no store
lock across the original callback. Publication checks membership again under the
original lock. Foreign owners, changed source bodies and absent sessions refuse
without changing retained values. Unsupported headless/legacy constructions keep
their original no-publication behavior; a missing `domain_services` attribute
does not turn their former no-op into an exception.

## Authored verification and limits

The single test constructs the actual settled private profile and backend graph,
then calls the original controller publication methods. A passive original-code
observer confirms the exact builtin-count calls and that snapshots do not
recompute them. It checks unpublished versus published-None, selected reset and
zero, frozen output, foreign owner/session rejection, replaced helper/resolver
refusal before callback invocation, and pruning after the original store close.
It supplies no fake App, controller, source guard or positive retirement proof.
Until central installs the constructor line, the test binds the same inert helper
to the actual endpoint explicitly.

Reviewed source and the two-method controller diff only. This is not a passing
test, lint, syntax or timing result. Root must execute the focused test after
review/adoption. Original README/edit journey and three-pair acceptance remain
unchanged; this presentation completion establishes no speedup.

Separate portable `product.patch` and `tests-only.patch` are generated against
the exact preimage/absence, with repository-relative targets and `/dev/null` new
file hunks. Only patch text line endings are normalized. Their SHA256 pins and
read-only application checks are recorded in the manifest.
