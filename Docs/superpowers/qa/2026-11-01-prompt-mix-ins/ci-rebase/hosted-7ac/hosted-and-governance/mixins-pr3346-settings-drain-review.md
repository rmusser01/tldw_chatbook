# Independent UI24 settings-task drain review

Verdict: no remaining actionable Critical, Important, or Minor findings in the settled patch against HEAD `7ac214599d4969a4d1a57db8bef41ff02eb433c9`.

Read-only review of the test drain helper, real `ConsoleSettingsDurabilityOwner` launch/retirement lifecycle, regression control, exact UI24 oracle evidence, and final gate enrollment. No product/test edits or runtime tests were performed by this reviewer.

## Correctness

The owner registers its real asyncio Task and removes it via an add_done_callback callback. A task may therefore already be done while that callback remains queued. The previous test helper repeatedly awaited gather over that same done registry, which can complete inline and starve the callback required to empty it.

The only pre-existing function changed is `_drain_settings_tasks`: after its original gather it now awaits `asyncio.sleep(0)`, yielding to queued retirement callbacks before checking the registry again. The helper neither clears the registry itself nor closes admission. It still waits for all observed work and propagates gather exceptions/cancellation unchanged; no return_exceptions or error suppression is introduced. Production ownership, shutdown, cancellation, timeouts, and thresholds remain unchanged.

The new pure control uses the actual owner, acquires a real lease, launches immediate work, and yields exactly once to establish the explicit precondition `task.done() and task in owner.tasks`. It then exercises the helper and requires an empty registry, still-open admission, and successful subsequent lease acquisition/release. Finally it calls the original owner close/drain cleanup. The five-second watchdog bounds this isolated regression; it does not relax any existing UI oracle or production behavior.

## Retained proof inspected

Directory: `/var/folders/sn/m80n2j152t9gw3w8qwk2nykh0000gn/T/mixins-ci-7ac-ui24-3o9r7m46`.

- `red.log` shows the new regression timing out after five seconds before the fix, with the traceback inside gather and its argument a finished `already-done-settings` Task with result None. This concretely qualifies the completed-but-registered starvation mechanism.
- `green.log` records **2 passed in 10.86 seconds**: the pure regression and exact gated `test_applying_on_anthropic_submits_none_of_the_hidden_fields` UI case (9.13-second call). The provider/hidden-field assertions themselves are unchanged.
- `static-proof.json`, as reported by the implementer, retains two existing lint findings before/after with no additions; final helper-file SHA below matches the settled implementer hash.

Results were produced by the implementer and read here, not rerun. No full-suite or hosted all-green claim is made.

## Gate enrollment

The initial review found the new pure control lived in an otherwise ungated helper module; importing that module from the hidden-fields test would not execute it. The parent resolved this by appending the exact function selector only. Static comparison confirms **1,331 -> 1,332** entries, the original 1,331-entry prefix unchanged, no overlapping selector for this file/function, and the effective coverage floor raised to 1,332. No whole helper suite, workflow, shard algorithm, or timeout change is included. Parent-owned checker/CI-unit results are separate.

## Reviewed working-tree SHA-256

- `Tests/UI/test_console_provider_apply_defaults_flow.py`: `dab7f701fc1162c1b169c56c2376b59a764efd5097136e17101c42850726808c`
- `scripts/ui_pr_gate_census.txt`: `6f4ec1a64f93adbd5414bdab44c9215700db2168803170cd9dfee46983e15132`
- `scripts/check_ui_pr_gate_census.py`: `5afc1f4f7f486030c3adc30a2e6f8549efb5f8672b9724c7992399d199f9ccd7`
