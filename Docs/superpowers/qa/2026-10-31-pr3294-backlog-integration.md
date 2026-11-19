# PR3294 predecessor integration evidence

Base: `77718ebebc588000531a87b915638833971a9ea7` (merged PR3293). Rebased source: `b64f9c7cf20fc02956d10d066c6f891ddfa2e1a1`, plus the fixture repairs, combined tracker regression and exact size pin in this commit. Original remote source: `40893d4cb3afcb1cb43a146a87855ecc0fc70390`.

Preserves the shared Protect `not_needed` tracker state and Provider deliberate `skipped`/403-attention states. Both flow through the existing shared progress builder. Credential handoff, staged-generation fencing, atomic persistence and already-started retirement owners remain intact. No new ADR: existing ADR012/146/217 apply.

## Targeted verification

| Receipt (private sibling directory) | Tests | Failures | Errors | Skips | XML SHA256 |
| --- | ---: | ---: | ---: | ---: | --- |
| combined-integration-1793494805439642100 | 30 | 2 | 2 | 0 | `8c9ca16fc0026af7b7dcb93b31a7b980de86825d7e4fe0d79a5af9a24c4b6086` |
| combined-retry-1793500450204725300 | 7 | 1 | 0 | 0 | `aef5849b70e9fd2a03d2933e8ba61324e116e899964b9c06d874f1f189b46083` |
| huggingface-isolated-1793500630376555100 | 1 | 0 | 0 | 0 | `12057aba2c77d0537e1a60d9f4b1c721e3ea410a464e66f578d79fbdf21c703d` |

All three runs have identical before/after source manifests. Their production source hashes also match each other and the final working source. Together the passing receipts cover 33 distinct selected cases; this is not a claim that any failed campaign passed. Earlier task repair receipts remain qualified by their own exact-source manifests in `.superpowers/sdd/.tmp-pr3294-review-repairs/task-1-report.md`.

The initial run passed 26 cases, including combined tracker, held discovery, dismissal/unmount, queued/in-flight retirement and Protect plaintext/encrypted behavior. Two atomic-handoff cases errored because their old fixture redirected the config path after registration. Only that class now uses the existing real `select_config_source` helper, binding the persistence consumer before a real load; production admission guards remain unchanged.

Two mounted executor fixtures initially failed: the competing real settings write consumed the held writer barrier deadline, and a successful provider write was followed by an unrelated checkpoint write inside the same deadline. The stale-identity fixture now makes the competing real write on the paused executor after release and before the actual CAS; it additionally asserts exactly one result with `identity_changed`. The success fixture stubs only its post-Model checkpoint after setup while retaining the real provider write, receipt, exactly-one-write and model assertions. Original 2-second entry and 3-second barrier/save limits remain unchanged.

The seven-case rerun passed both handoff cases, the credential/Moonshot/custom stale-identity controls, and the unchanged-identity real-write control. Its additional HuggingFace parameter hit the unchanged 3-second save deadline during substantial real config-write work. It then passed unchanged in isolation (21.45 seconds including setup/teardown) with other session-owned heavy tests idle. Logs do not locate the original owner precisely inside CAS versus refresh at timeout; no production defect or timing-bound guarantee is claimed. The failed receipt is retained as a timing limitation.

## Static checks and review scope

All 26 changed production owners match the diagnostic inventory; no await/DOM census additions. Twenty-one production blobs are conserved from the earlier independently reviewed repair; five reflect PR3293 integration/shared progress changes. Scoped Ruff diagnostic multisets are unchanged: three existing findings in the large wizard test, three existing import-order findings in setup-state tests, none in the atomic-integration test. `git diff --check` passes. Exact affected module rows match measured lines: providers_models_card 2398, FirstRunSetupWizard 3107, ModelStep 975, ProviderStep 1690. The extra Wizard line carries the dated user size waiver. No guard, test assertion or protection is weakened.

The independent task review found no blocking issue; the final whole-branch integration review is tied to the final commit separately. Hosted CI and size limits are waived by the user; this evidence does not claim a full suite, paid/live provider authentication, or hosted CI pass. TASK34100.6 remains In Progress with AC14/32 and existing follow-ups open. No release or main promotion is part of this session.
