# PR3346 fenced project-chat CI diagnosis

## Finding and approved correction

The project-chat UAT expects the full discovery/workspace-file tool journey but enables only console.local_tools_enabled. It never opts into tools.full_agent_tools_enabled. The current production light-surface policy defaults that second gate off and intentionally removes fs_* from ordinary sends. The workspace note correctly reports bound folders without available path tools and omits their unusable aliases. The strict scripted model then fails its one-advertised-binding assertion.

Applied minimal fix, approved by root under TASK38544 AC16: explicitly preconfigure tools.full_agent_tools_enabled=True in this UAT's existing tools configuration, alongside its existing preconfigured read permissions. Preserve every actual binding/disclosure/read/persistence/painted-evidence assertion and all production gates. This selects the intended user configuration; it does not make the default light surface advertise workspace authority.

Root was informed of this cause and recommendation before the fixture edit and authorized this correction. Tests/QA/test_project_chat_uat.py now explicitly opts into the full tool surface and discloses that opt-in in its receipt substitutions. No production gate, assertion, budget, manifest or other source file was changed by this investigation. No commit or push was made.

## Evidence

Rebased HEAD 79e659dcde6f5360edb0070f9d184713bd8584d1 and pinned dev 62a169f906508ed02bd7546ab278d916638d1607 both fail the exact original node:
Tests/QA/test_project_chat_uat.py::test_project_chat_uat[fenced].

- Rebased: 1 failed in 18.30s; private child 1 failed in 15.09s.
- Pinned-dev archive: 1 failed in 20.87s; private child 1 failed in 17.09s.
- Both: expected one advertised project binding, got [].
- Both original UAT flows already created the named workspace, added its one real folder binding, verified its resolved locator, activated the workspace chat and sent via the actual composer.
- Both saved private configs show console.local_tools_enabled=True and tools.full_agent_tools_enabled absent.
- Both actual outbound request receipts contain the named UAT workspace and the bounded message: “This workspace has bound folders, but no fs_*, git_* or virtual_cli tools are available in this run.”
- The UAT, scripted-model oracle, agent bridge, light-surface owner and workspace note owner are byte-identical to pinned dev. SHA256 proofs and selected config/receipt facts are in cause-proof.json.

The original hosted log is copied exactly as admission2-ci.log from /private/tmp/mixins-pr3346-admission2.log, with SHA256 retained. Its fenced failure matches both reproductions. The later artifact checker reports missing/duplicate UAT cases because the failing UAT never produced its complete expected case set; it is downstream evidence, not a separate root cause established here.

## Ownership trace

1. Tests/QA/test_project_chat_uat.py saves console.local_tools_enabled=True but its tools section configures only list_directory_enabled.
2. Agents/plain_chat_tool_surface.py defines FULL_AGENT_TOOLS_GATE_KEY=full_agent_tools_enabled and FULL_AGENT_TOOLS_DEFAULT_ENABLED=False. Its production reader light_surface_configured returns the inverse of that configured gate.
3. ConsoleAgentBridge.light_surface_for keeps this ordinary first send light: no explicitly bound skill, narrow Goal launch or in-flight agent work applies.
4. build_console_first_request_plan applies light_surface_allowed_tools before producing the model-facing workspace note. The narrowing retains its safe set, independently enabled builtins and admitted user sources; fs_list/fs_read belong to the local full surface and are removed.
5. _advertised_path_root_aliases derives aliases only from providers whose path tools survive that final allowed list. workspace_context_note therefore omits aliases rather than advertising inaccessible folders.
6. ProjectChatModel._reply correctly insists on one alias from actual messages/schemas before issuing its discovery/list/read script. It receives none and fails, as designed.

ADR104's 2026-10-31 light-surface amendment explicitly documents this default-off behavior for upgraded profiles too. Existing light-surface tests explicitly enable this same gate when testing full agent behavior. No production permission widening is indicated.

## Isolation and limitations

The original repros ran only the exact failing fenced node. After the approved correction, all four existing variants ran using their unchanged private_profile_test child, real Textual controls, real workspace registry/filesystem, normal admission and loopback model server. The UAT scrubbed credentials/environment and asserted owned network destinations. Repository Python 3.12, no pytest cache provider, PYTHONDONTWRITEBYTECODE=1, distinct owned basetemps/artifact directories, unchanged test-specific 240s timeout and lane --timeout=300 were used. Exact argv/cwd/results are current.json and baseline.json. The previously owned pinned-dev git archive was reused without modifying it. No helpers or full suite.

This is a reproducible baseline fixture/configuration mismatch, not a mix-in-only authority regression. The explicit UAT opt-in selects the full-tool configuration prescribed by ADR104 while the production default-off policy remains unchanged.

## Fixed verification

- Exact four UAT nodes (fenced, native, compact-recovery, wrong-family): **4 passed in 107.35s**, exit 0. Exact argv, cwd and wall time are in fixed-all.json; full output in fixed-all.log and fixed-all.xml. The runner is run-fixed.py.
- `scripts/check_project_chat_uat.py --junit <owned>/fixed-all.xml --artifacts <owned>/fixed-all-artifacts`: **exit 0**, “Project-chat UAT: all four cases passed with complete receipts.” Exact argv/output is validator.json.
- All four actual private-profile receipts are status=passed, phase=complete, model_errors=[], blocked_destinations=[]; their rendered viewport text/SVG and request/turn evidence remain attached. The native and fenced paths, compaction recovery and wrong-family refusal assertions are unchanged.
- `git diff --check -- Tests/QA/test_project_chat_uat.py`: exit 0.
- Per-file Ruff compared to FIX_BASE 79e659dcde6f5360edb0070f9d184713bd8584d1 using identical stdin filename/config: existing lint JSON is byte-identical; both formatter checks exit 1 on existing formatting debt, and the exact before/after formatting edit inventory is identical. See lint-comparison.json and format-delta.json. An initial normal Ruff invocation could not write the managed checkout cache; the recorded comparison uses --no-cache. No unrelated formatting cleanup.

The fixed run uses a private local scripted HTTP model with the existing real Textual/bootstrap, folder authority, read/discovery, transcript, persistence, network-block and rendered-evidence oracles. It does not establish live cloud-model behavior. Root is concurrently fixing separate CI findings; this report owns only the explicit UAT fixture correction. No full suite, additional agents, commits, pushes or repository evidence manifests were used.

## Exact owned evidence

report-files.json lists explicit absolute paths, byte counts and SHA256 hashes for every selected report, command, log, junit, original red receipt and fixed complete UAT artifact. This inventory is scoped to this uniquely created private directory. It excludes private profile databases/config files and does not collect unrelated /tmp task globs.
