# Single-readiness guidance implementation review

**Final approval: all seven exact Python files reviewed; no remaining finding.** Exact reviewed product/new-test hashes are in `/private/tmp/uat-guidance-single-readiness-implementation-hashes.json`. No repository edits were made by this reviewer.

## Original finding — now resolved

[P2, test compatibility] `Tests/UI/test_console_fleet_wake_ui_freshness.py:285` assigns `_console_provider_blocker_copy = lambda: ""`, then explicitly invokes `_sync_native_console_chat_ui` at303. The real sync reaches `_sync_console_control_bar_under_config`→`_sync_console_transcript_guidance`, which now passes `settings_readiness=`. The zero-argument lambda cannot accept it. Give that test double the optional keyword (preserving its constant return and all assertions); similarly inspect other mounted helper stubs that can receive an ordinary sync. Do not add production TypeError swallowing. This incompatibility follows the exact caller graph; the attempted existing test execution failed earlier at profile setup and therefore did not exercise it.

## Product behavior

The first full `_active_console_settings_readiness()` call remains at the original first-acquisition point. All actual native checks, cold session creation and task177 convergence are completed inside that unchanged method. Its returned frozen settings/readiness pair is a local variable passed to the three synchronous projection helpers. Default helper calls still acquire independently. No attribute memo, config operation lifetime, await, UI effect, native permission check or send admission is added/removed by this plumbing.

Independent AST normalization proved all four modified methods unchanged after removing only the optional pair arguments, their None/fresh branch and the one guidance acquisition/keyword forwarding. In particular first-send and message predicates remain ordered after readiness; surface and modal calls remain later. Error propagation has no new handler, and no pair survives on the screen. Sharing one display snapshot intentionally avoids mixed values from three redundant derives; it does not promise observing every hypothetical external mutation between the old reads.

## Executed evidence

Root genuine RED `/private/tmp/uat-guidance-red2.log` reaches all state/output assertions and reports derives `[3,3,3,3]` rather than ones. Earlier invalid fixture/API attempts are not product RED evidence.

Independent new native case **PASS8.389s** in a unique fixed-profile child, using the actual TldwCli Settings boot and an unmounted ChatScreen. It delegates real readiness/config/session/helper calls. Only the two downstream UI consumers are replaced with observation sinks, avoiding a second mounted Console or discovery. It covers cold blocked session creation; real Settings save followed by local-provider convergence; explicit user OpenAI settings retained; credential save; three independent helpers each fresh; repeated guidance counts1; predicate/consumer ordering; acquisition-boundary and first-send projection exceptions retained by identity; and a successful fresh pass after each exception. Session convergence here is **in-memory session-store state**, not proof of a durable conversation write. Real config public save/read APIs are exercised; no separate fresh-process config readback was added.

The combined independent selection finished **1PASS/5FAIL in11.89s** (`/private/tmp/uat-guidance-independent.log`, `.xml`). All five legacy compatibility tests failed before the changed helper use with `raw_source_selection_changed` while constructing their test app. These are unqualified compatibility results, not passes and not a new baseline-equivalence claim. No guard was bypassed or profile authority reset. Root independently reports the new case plus existing four provider-derivation cases5PASS43.67s.

The new test's count assertion is meaningful because the original producer actually runs and the same state/output assertions succeeded in the baseline RED. It also requires real guarded config loads to remain. UI sinks check no derivation memo is left, matching the narrow design. These results do not qualify native Windows startup or prove a wall-time benefit; the Windows rerun remains necessary.

## Final fixture and selection verification

The six zero-argument helper lambdas now accept only optional keyword `settings_readiness=None`. An independent AST comparison across all four fixture files proves their bodies, returns and assertions otherwise unchanged. The qualified pre-fix fleet run reached the predicted exact unexpected-keyword TypeError (`/private/tmp/uat-guidance-fleet-before-onu22_ch/pytest.log`), so this is a demonstrated compatibility repair, not conjecture or a guard change.

Independently parsed all five per-case JUnit files referenced in `/private/tmp/uat-guidance-compat-status.log`: each contains exactly its expected original case, passed, no failure/error/skip. Case durations8.964/2.520/3.943/4.298/3.950s. The temporary launcher sets a fresh private profile before imports, asserts exactly one matching collected node, and adds only the recognized `_private_profile_test` attribute. The existing fixture preserves that interpreter's selected profile; no native authority, config global or test function body is replaced. Its60s pytest/90s parent bounds are diagnostic-only and do not alter repository test deadlines. These qualified passes supersede the unresolved compatibility gap, while the earlier unqualified setup failures remain recorded above.

Runner AST comparison proves only two additions of the new native guidance module: existing `_PRODUCT_TESTS` (therefore full/support) and native-close diagnostic. All prior selections remain, with no budget/mode/assertion changes.

Product and new native test hashes are identical to my independently executed bytes. The final receipt now includes exactly seven repository Python files (six modified plus the new native test), and separately hashes the temporary diagnostic launcher. Root's final static summary reports zero new Ruff/Bandit findings for the six modified files; the new extracted script has only39 test assertions. I also independently parsed the five further JUnit files from `/private/tmp/uat-guidance-app-status.log`: the three existing draft-memo cases and two unchanged Mac GGUF keyboard cases all passed. These are local compatibility/functional results; native Windows qualification and any startup performance conclusion remain separate.
