# Windows ec139 default rollback causal review

The only observed terminal failure is the **test parent's 900-second child timeout**. The second accepted later rollback was actively progressing through publication and post-publication validation; no successful terminal validation or product exception was retained. This is not evidence of an admission deadlock or an Evals failure. Controller verification reports 42 native passes and 9 Evals passes; the default composite case failed in 1717.652 seconds.

Evidence: `later-parent-failure.json.log`, `home/later-phase-timing.log`, `home/later-stacks.log`, and `home/later-admission-timing.log` under:

`/private/tmp/uat-windows-ec139-default/backup-platform-windows-2022-py3.12-replacement-default-evals-ec139a0241f2de0b8c5da62916e25157f9ac4b35/test-logs/product-pytest/test_default_profile_service_r0`

Four retained rollback-worker snapshots advance through `_complete_move → _begin_move → _validate_installed → _preserved_builtin_validation`. The final sample is native `fstat` while observing a preserved regular file. Exact ec139 source: `publication.py:550,2074,2305`; `later_rollback.py:584,612`; `publication.py:974–975`; `journal.py:320`. The last loop can select only `persona.visual_identity_builtin` or `persona.assets`; the metadata does not identify which owner/member was current. Earlier snapshots show `_flush_records` rereading authenticated journal records (`journal.py:1515,1595`). The service waiter is waiting for that worker's future; the UI loop is polling. No progress from these samples should be converted into a claim that validation eventually succeeded.

In the final 55.297-second CPU window, process CPU advanced **52.984375 seconds**, of which the rollback worker advanced **50.890625 seconds**. That supports ongoing work rather than an idle wait, but does not apportion CPU across journal validation, native security checks, hashing, or preserved-member validation. Native-call timing was disabled. All 28 observed `_groups` calls completed without errors: total 1.954958 seconds, max 75.516 milliseconds. All 1612 `_tokens` calls completed: total 5.253706 seconds, max 8.064 milliseconds. These inclusive aggregates are not additive; they do not explain the elapsed child time.

The earlier successful exact 9aff fixture has the same inspected publication/later-rollback/journal/Windows filesystem code and later-child driver. Measured child checkpoint comparison:

| Checkpoint or interval | 9aff success | ec139 timeout |
|---|---:|---:|
| Begin → final acknowledged review | 543.437s | 555.531s |
| Final review → validated rollback | 344.938s | No terminal result |
| Validated rollback → loop closed | 8.218s | Not reached in evidence |
| Begin → loop closed | 896.593s | Parent killed after 900s |

The first deliberate credential stop and untouched Abort passed in both runs. Four reviews remain roughly 79–83 seconds each. ec139 had consumed 12.094 seconds more before the final attempt. Its remaining child allowance was **less than 344.469 seconds**, because the 900-second parent timer also includes imports before the begin marker. This establishes a tight composite observation envelope, without projecting the previous tail as a measured ec139 completion time.

**Budget assessment:** the 900-second value is a test-only `subprocess.run` timeout in `test_later_rollback_credential_ui.py`; its comment describes four reviews, two execution attempts, Abort, and readback. It is not passed to product admission or recovery APIs. The final individual service wait remains 360 seconds. Reusing an actual remaining absolute 2400-second enclosing testcase budget is a justified bounded harness design candidate, not a demonstrated product-speed fix. It must account for setup/earlier work charged to the enclosing test and reserve the default wrapper's subsequent 135-second ordinary reopen (`test_default_service_replacement.py:26`, `test_f9_replacement_workflow.py:118`) plus error recording/cleanup/assertions. A fresh 2400-second child allowance, consuming all remaining time, or an arbitrary 1200-second replacement is not justified. Changing the observation envelope still changes test acceptance and requires the controller's stated authorization; no edit or rerun was performed here. The unrelated 60-second startup regression criterion remains separate.

No product correction, guard relaxation, or specific hot-call optimization is established by these samples. Existing evidence is sufficient to review the composite-budget issue; exact native cost attribution would require separate bounded measurements if that becomes the authorized question. No Linux logs, live UAT profiles, or repository files were accessed for mutation.
