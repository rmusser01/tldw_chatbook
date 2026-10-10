# Console hook preparation lifetime verification

Task: TASK-34563.13. Base: 6d8d56e75c819d993e0e1df299e6a7a2727e5a52.
ADR: [ADR-225](../../backlog/decisions/225-console-send-preparation-and-io-ownership.md).
Plan: [finite hook preparation lifetime](../superpowers/plans/2026-10-07-console-hook-preparation-lifetime.md).

## Result

The original cancellation regression is fixed within the three selected hook reads. The final integrated selection passed **46 tests, with zero failures, errors or skips**. These checks establish native lifetime and compatibility behavior; they do not establish faster Send or the 100 ms rendered/input-feedback target.

The controller's hook-admission snapshot and runtime's v2 configuration/context-key captures retain the actual executor Future through repeated cancellation. One physical record is observed by its original controller and runtime, including an already-issued standalone read later attached to a runtime. Replacement cannot detach it. Close/dispose drain exact retirement notices before releasing turn custody or store ownership, after the unchanged generic grace. Completed native work does not extend unrelated custom/provider waits.

Source checks prevent a held read from publishing against replacement permission, store, session, controller or controller workspace inputs. The stock context body consumes its captured sources. Cold canonical owner creation and custom accessors remain on the worker. The reusable hook lifecycle keeps its existing freshness policy: transient preparation witnesses do not permanently invalidate it after a supported controller rebind.

No admission index conversion, executor, scheduler, permission cache or durability policy was added. Saved-turn failure still keeps the draft and refuses dispatch. Existing temporary chats, WAL/NORMAL, required consent/checkpoint/effect ordering, and awaited history are preserved.

## Evidence

All native runs used the existing contained runner sequentially, with original deadlines and private profiles. No full suite or timing campaign was run.

| Run label | Outcome | Scope |
| --- | --- | --- |
| hook-preparation-native-red | Expected assertion failure | Unchanged original Send task finished cancellation while the original snapshot still held raw operations/leases. |
| hook-preparation-integrated | 45 passed, 1 failed | One new observer setup failed before Send: Python 3.12 rejects local PY_UNWIND. Product source was unchanged afterward. |
| hook-preparation-workspace-corrected | 1 passed | Corrected observer proved actual WorkspaceDB connection/lease retirement through disposal cancellation. |
| hook-preparation-integrated-final | 46 passed | Fresh process after correcting observer registration and setup cleanup. |

Final selection: 15 helper cases; 14 actual creator/lifetime/source/compatibility cases; 10 existing demand-driven hook cases; 7 existing controls covering initial review Close/dispose, unrelated provider tail, exact received claim, real saved-turn failure, mounted Send, and permission revocation. The source tests use original permission and WorkspaceDB producers, with passive resource observations. No external hook command or paid provider call is required by this selection.

Every run's owned process tree became empty normally, its native identity and diagnostic tasks retired, and its private profile was removed. Force and identity overflow were zero. The first integrated run recorded one diagnostic process-lookup race; the other three recorded zero. Native containment retirement is distinct from the original resource assertions inside the tests. The final run has no unawaited-coroutine, destroyed-task or ignored-exception signature; its warnings are existing deprecations.

Evidence is retained locally under `.superpowers/sdd/2026-10-07-console-hook-preparation-lifetime/`: baseline-source.json, red-audit.json, first-integrated-audit.json, final-static.json and final-audit.json. Runner logs/XML/result/custody receipts are in the shared preparation checks directory under the labels above.

## Frozen source

| File | SHA256 |
| --- | --- |
| Chat/console_hook_preparation.py | 37d391fb5397f4903826f7524e1e58c1d2c1a2b3310058adbb8b96e4b60e57d2 |
| Chat/console_chat_controller.py | 79269e290c595f8c5e3d0e10504a50d7e3a57ffefc5d1fe41e0ea4f55f45c64e |
| Chat/console_runtime.py | c17e117e0d50418b3c19ef78d1c2d2ee6d3a5328cee31a0937e684590be9adc9 |
| Tests/Chat/test_console_hook_preparation_reads.py | 983acf224c9b5cf958bcefba9f9800314c4188037b37c55b46ff86e49e7fe339 |
| Tests/Chat/test_console_hook_preparation_lifetime.py | 0f09a8d50dd42f702f8f2f89f7d3ef37516edbc21ac261e202c73b2984a4dfdd |

Two independent source reviews found the missing controller app/context-provider checks and overbroad reusable-lifecycle witness; both were corrected and tested. Final reviewed scope has no remaining actionable finding. The Python observer setup incident is recorded in lessons-testing-evidence.md.

## Static and remaining scope

AST and diff checks pass. New helper/tests and runtime have zero Ruff findings; controller's 60 findings are unchanged. No introduced formatter transformations; existing controller/runtime transformations remain 15/38. The module-size ratchet is still not green: controller grows from 30,286 to 30,465 lines against 29,299; unchanged store is 22,814 against 22,344, and interrupts 6,520 against 6,471. Caps were not changed.

The task's four behavior criteria are verified, but its status remains In Progress because full repository DoD/static hygiene is not established. This slice does not complete the larger latency objective or cover every native preparation operation. Reference expansion, Library reads, early unconfigured UI receipt with exact input revisions, and real rendered/input and action-to-adapter qualification remain follow-on work. No primary checkout edits, push or merge.
