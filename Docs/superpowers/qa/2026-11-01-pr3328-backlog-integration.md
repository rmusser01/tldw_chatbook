# PR3328 integration and repair evidence

Behavior candidate: `6357516f55daef96133e4e401f958c78f9064725`, based on merged PR3323/dev `4a38b3d2dba872e46ca8f9569908d9ccfd7aa2e9`. Independent review continues from immutable `fcc069896880d842e3838054336ed64cfcd7132b`. Following evidence candidate `a214e3806180acd933f3d9177264246de6488544`, the final review delta changes three user-facing strings and their existing assertions/documentation only; no admission or runtime behavior changes. No remote operation was performed by the implementer.

I1 captures slash/palette origin before scheduling, then applies the existing chat/composer guard before compaction. I3 forwards exact skill bindings through the shared precommit Ask gate from all four turn actions. Its real accounting/SQLite control proves no accepted attempt, sibling or provider call on Ask.

I2 estimates current static mandatory inputs and uses current-owner locally published provider input/output caps with the shared capacity solver. All agent-runtime/bridge-present chats, including light agent chat, have **unavailable next-send forecasts**. High displayed estimated history retains an explicitly UNKNOWN advisory; it never establishes Ready/ASK or definite next-send pausing. Blocking readiness, dispatch/emergency, acknowledgement and failure statuses keep priority. Broad AC3 remains unchecked and TASK33621.4 In Progress: qualified current tool/schema provenance/invalidation and broader dynamic-input fidelity remain unfinished. This is coordinator-selected partial implementation, not user acceptance of narrowed AC3.

Do not call tool preview for forecast. Its no-I/O docstring does not cover the transitive light path: `preview_tool_surface -> light_surface_local_tools_available -> tool_kill_switch_engaged -> UnifiedControlPlane.get_kill_switch -> PermissionStore.get_kill_switch -> load/mutation_fence`, including corruption backup/reset. This repair adds no parallel send assembler, planner-input owner/cache or permission I/O.

| Receipt | Outcome | Meaning |
| --- | --- | --- |
| r-02ea0a5f | RED, 2 failures, 248.45s | Actual delayed slash/palette workers wrongly admit later chat |
| r-4fa5818a | GREEN, 5 passed, 224.66s | I1 ownership and direct command success/refusal/error |
| r-5a8176ff | RED, 56.16s | Real prepared M5000/H5000/budget5000/trigger4000 exposes omitted overhead |
| r-b98769b6 | I3 RED, 71.42s | Admitted skill overhead wrongly passes precommit gate; concurrent I2 failure was fixture-only |
| r-090af2c8 | GREEN, 6 passed, 271.56s | Combined I1/I2/I3/cold/renderer controls |
| r-3818a982 | RED, 81.88s | Disabled high-context advisory fails intended assertion |
| r-26fd8ae4 | RED, 44.62s | Unavailable copy wrongly masks Send acknowledgement |
| r-6805ddf0 | GREEN, 2 passed, 115.06s | Final mounted static capacity/advisory and status priority |
| r-f1a69890 | 7 passed / 3 failed, 414.56s | Cap currentness, old revision, three summary fences, recovery fallback and light skill pass; three mounted hold journeys time out before hold assertions |
| r-29ace5d6 | GREEN, 1 passed, 16.27s | Existing tooltip control expects explicitly estimated wording |
| r-cc2390f0 | diagnostic failure, 98.25s | Source-only sampling shows progressing native authority reads, not a fixed-lock deadlock |
| r-e99598da | 3 failed, 272.94s | Supported persisted nonstream fixture trial still times out; incidental fixture change reverted |
| r-cc804daa | base comparison failure, 165.58s | Exact merged base reproduces the shared early Continue timeout |
| r-d9e4aa47 | 3 selected nodes: 2 pass / 1 F1 teardown error, 49.33s | F1 body assertions complete; native-worker retirement errors (XML aggregate: 4 / 0 failures / 1 error, including teardown accounting) |

Every receipt above has unchanged paired before/after source hashes. The final GREEN mounted test changes current prompt/catalog and compares mandatory/history/cap/output accounting against real prepared/preflight values, then checks agent low-unknown/high-advisory without tool preview/provider calls. Earlier prompt-capture, margin, framing and warmup fixture failures are not causal RED evidence. Retained I1 action/I3/cold function hashes were reverified after the narrow copy adjustment.

SHA256 identities (only the final review copy changes production after the behavior candidate):

| File | SHA256 |
| --- | --- |
| `UI/Console_Modules/compact_command.py` (r-d9e4aa47) | `d3ed729e6dd80320494e1fb409588c610df3026f01cb4afc8e753ee4f63ad349` |
| `Chat/console_context_budget_copy.py` (r-d9e4aa47) | `4594f4c138da8c0abfa4d9626de9572cecb7b3d8f021862b9fd4f36372080520` |
| `Tests/Chat/test_console_compact_command.py` (r-d9e4aa47) | `d05240ff33cc404f3df6c1a13fd29728ccc832c4e5eeddfcd7a43db38bd95f6b` |
| `Tests/UI/test_console_compaction_hold_status.py` (r-d9e4aa47) | `163f3bfc9ddf775a11deca2a22a8f639750a5f609543ded9808f1cf6889d6862` |
| `Chat/console_compaction_gate.py` | `e70e6e685c1eeed18d575a723ba2a6930a1463567ea8417827ff3c7b40d8b568` |
| `UI/Console_Modules/context_spend.py` | `21b55771f78c0716046a8f7e4351f6026ee47e7ad3ea859a15a5ced1058c3691` |
| `Tests/UI/test_console_compaction_forecast.py` (r-29ace5d6) | `096439ea491fdc351eec199176133d98835012147af5c61b7f99dd2211ba206e` |
| `Tests/UI/test_console_compaction_hold_flow.py` (r-6805ddf0; trial reverted) | `fc37dcd17eb710231c7b68c4caa1ffd6661aeabe3c3bf321098836a05147bc68` |
| `Tests/UI/test_console_readiness_config_projection.py` (r-f1a69890) | `228cb04d1f4333a2b9884f98621705fdf7a4912ff06b107c8f8ffc3f7a09a2a5` |

The three unqualified mounted journeys are hold-card copy, held rewind and Continue. Their existing 10s `wait_for` expires in visible Send -> `capture_turn_configuration_snapshot` -> `console_configuration_preparation.py:1503` finite `capture_authority` -> `console_preparation_reads.py:226/246`, before hold assertions. Sampling observes Windows native admission advancing through MCP, workspace bindings, skill/plugin config and RAG definitions. The exact-base comparison uses production archive SHA256 `e9df865b61d44d4b35270cf0d7dafdfcb460fc93331c25990d92f12fd96a4e6a` with only candidate hold-flow test overlay SHA256 `b8ccc3f7d5ec8ea4b568aed27b1f8032083956486690ff17ee2965c30f03af07`. It reproduces the **shared inherited pre-hold precondition timeout** for Continue. It does not prove those three hold-feature assertions pass or establish resolver-wrapper causation. No timeout/admission guard was weakened; no fallback repair is claimed.

Static lint differential against the reviewed baseline: 125 inherited diagnostics before/after, zero new (line-number-only F811 text normalized). Whitespace diff clean. Inventory rebuild: 721 owners, 1069 TASK492 calls, 56 TASK31551 calls, 7795 TASK494 calls, 16 sinks; base720/current721 adds only compact-command owner, with no changed/removed prior owner, unchanged privacy116/sinks16. The warning interpolates exception type only. Inventory SHA256 `5d594d1660953dc5b136e8a3e0d7763a0665933d28cc15b18621f307c32a9046`; merged JSON delimiter corrected before validation. Four pure size commits omitted; 28 behavioral source commits retained; dev ratchet/budget files unchanged. CI/size waiver is not a passing check claim.

Census reconciliation appends seven locally passing node selectors/eight cases, preserves all 1315 prior entries and floor, and validates 1322 entries. Failed mounted journeys are not newly enrolled as passing evidence. No full suite, live provider, physical small-screen layout or performance qualification is claimed. Known unrelated base trace/Stop and shutdown-layer failures remain untouched.

.102 remains Minor: billed compaction can finish without its command-result report when the outer worker is cancelled; accepted-turn custody does not own this command. Other existing follow-ups, including .100/.80/.90/.95/.96/.103/.105/.92/.93/.99/.101/.104, remain unresolved. Capture-on held rewind provenance and broader dynamic forecast semantics are not certified here. Historical task implementation/live/all-green/raised-ratchet paragraphs describe earlier source heads and are explicitly superseded by this integration qualification.

Canonical plan: [2026-11-01-pr3328-backlog-integration.md](../plans/2026-11-01-pr3328-backlog-integration.md). ADR required: no; existing ADR052/230 apply. Full content-free receipt hash maps, lint comparison and inventory comparison are retained in the ignored integration evidence package for the same independent reviewer.

## Final review copy correction

The `/compact` exception catch now says only “Conversation compaction failed.” A postcommit finish/flatten failure cannot justify the former claim that memory was not updated; this change makes no persisted-state claim. Logging still includes exception type only, with the identical diagnostic statement. F1 limits hold-card choices to prepared composer sends, names queued/Buddy/unprepared over-trigger cardless refusals, and qualifies static forecast estimates versus unavailable agent/tool forecasts and possible high-context advisory.

Only the directly affected compact-error, hold-detail copy and existing F1 help controls were selected in r-d9e4aa47. The first two pass cleanly. F1 completed its body assertions, but teardown hit `Tests/conftest.py:487` executor shutdown grace and `Tests/conftest.py:1869` refused config retirement while native workers remained uncertain. This receipt is not all-GREEN; no comparative evidence establishes this teardown error as inherited. The guard and existing timeouts remain intact, with no unrelated shutdown/admission repair. No new tests or census changes are needed for these existing controls. I1/I2/I3 guards, partial AC3, .102 and all prior qualification limits remain unchanged.
