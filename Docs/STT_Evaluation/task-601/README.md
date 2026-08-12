# TASK-601 native process-tree evidence

## Scope

This evidence closes TASK-601 acceptance criterion 6 for process ownership and
cleanup ordering. It proves that the local STT worker and a preparation
descendant are contained and terminated as one native process tree before
generation scratch cleanup. It does not test FFmpeg media correctness, model
loading, transcription accuracy, or performance.

## Passing matrix

[GitHub Actions run 31577352552](https://github.com/rmusser01/tldw_chatbook/actions/runs/31577352552)
passed against executable commit
`5c6a446c8d050587f141561319e58e1ce72c528d`:

| Runner | Host | Architecture | Result |
| --- | --- | --- | --- |
| `ubuntu-24.04` | Linux | x86_64 | passed |
| `windows-2022` | Windows | x86_64 | passed |
| `macos-15-intel` | macOS (Darwin) | x86_64 | passed |

Every platform passed these exact nodes:

- `Tests/STT/test_executor_process_tree.py::test_native_force_stop_removes_worker_and_descendant_before_scratch_cleanup`
- `Tests/STT/test_executor_process_tree.py::test_native_crashed_leader_reaps_descendant_before_scratch_cleanup`
- `Tests/STT/test_local_stt_executor.py::test_force_stop_detaches_before_kill_and_cleans_generation_scratch`

The workflow installed only the repository development dependencies. It used no
STT model/runtime extra, STT model/runtime download, inference, or general CI
result. The normalized source records and exact run binding are in
[`platform-evidence.json`](platform-evidence.json).

## Remediation and reruns

- Matrix attempt 1, [run 31575959082](https://github.com/rmusser01/tldw_chatbook/actions/runs/31575959082), tested `48c4ccadb9ef93f64ddfffd9954aede9526ea48e`. Linux and macOS passed; Windows remained red because a POSIX-emulation unit test required `os.killpg` to pre-exist. All three required native nodes passed on Windows. The test-only monkeypatch was made portable.
- Matrix attempt 2, [run 31576646463](https://github.com/rmusser01/tldw_chatbook/actions/runs/31576646463), tested `83c68c30d82fe04b53db02612ff358fb2fb6a0ec`. Linux and macOS passed; Windows reached the same POSIX-emulation unit test's unavailable `SIGKILL` symbol. All three required native nodes again passed on Windows. The test-only signal seam was made portable.
- Matrix attempt 3, run 31577352552 above, tested `5c6a446c8d050587f141561319e58e1ce72c528d`; all three native lanes and all required nodes passed. No production defect or production remediation was needed.

## Validation

From the repository root:

```bash
python .github/scripts/task601_process_tree_evidence.py \
  --validate-aggregate Docs/STT_Evaluation/task-601/platform-evidence.json
```
