# UI24 settings task retirement starvation

Baseline7ac214599d4969a4d1a57db8bef41ff02eb433c9. Only `Tests/UI/test_console_provider_apply_defaults_flow.py` changed. Parent task38544 records the confirmed fixture-fix plan. Existing ADR095 applies; no new ADR for test-only correction. No production owner, settings authority, source/cancellation policy, CI timeout, size pin or provider assertion changed.

## Cause and bounded reproduction

Hosted exact node `Tests/UI/test_console_settings_hidden_fields.py::test_applying_on_anthropic_submits_none_of_the_hidden_fields` timed out180s in the shared `_drain_settings_tasks` loop, with idle worker threads. The helper repeatedly gathers the owner.tasks registry. `ConsoleSettingsDurabilityOwner.launch` adds tasks to that registry and retires them with a scheduled add_done_callback. A task can therefore be done while still registered. Python3.12 asyncio.gather optimizes already-done tasks and completes inline; awaiting it need not yield. The helper then loops on the same done task indefinitely, preventing its retirement callback from running.

A new pure regression uses the actual owner and actual helper. It launches an immediate coroutine, yields one real loop turn, and asserts both task.done() and task in owner.tasks. The unchanged helper then reproduces the starvation under a5s func-only test watchdog:1 expected failed in7.16s. The traceback explicitly shows a finished Task with resultNone being repeatedly gathered. The watchdog is specific to this tiny regression and leaves the original hosted180s bound unchanged. Its finally uses existing production close_and_drain to retire the diagnostic owner; no mocking or manual registry manipulation.

## Fix and qualification

The shared test helper yields with asyncio.sleep(0) after each successful gather. Scheduled retirement callbacks can run before the next quiescence check. It still waits for the actual registry to empty, propagates gather failures, and does not close admission or replace production shutdown.

The regression requires an empty owner registry and open admission after draining; a new lease can still be acquired/released. Existing production close_and_drain remains unchanged.

Targeted GREEN: new pure regression plus exact hosted Anthropic hidden-fields journey,2 passed in10.86s; Anthropic call9.13s. The original UI test remains byte-identical: hidden fields are omitted from committed settings and provider request; visible temperature0.6 and actual modelclaude-sonnet-4-5 remain asserted. Exact command/cwd/exit/time in red.json and green.json. No broader suite or repeated module runs. All processes stopped.

Static proof: only `_drain_settings_tasks` differs among original top-level definitions; one new regression added. Ruff baseline2/current2 with zero additions. Existing format debt retained; new regression formatter-identical. Diff check clean. Final source SHA256dab7f701fc1162c1b169c56c2376b59a764efd5097136e17101c42850726808c. Root owns gate enrollment, manifests, commit/push and hosted confirmation. Independent read-only review requested; its separate report records disposition.
