# TASK-22512 dependency qualification

Status: BLOCKED on the pinned native Windows boundary. Native macOS ARM64 and
Linux ARM64 package, parser, and environment rows pass. Native Windows x64
CPython 3.11 ran on Windows build 26100: package identity, platform floor,
ConPTY construction, Job admission and cleanup, bounded I/O, profile discovery,
app-crash cleanup, and managed RSS passed, but alternate-buffer isolation and
post-exit EOF/output integrity failed. ADR-099 therefore requires Windows
Terminal to remain fail closed; no Windows dependency is admitted by this
artifact.

## Scope and decision

This is the ADR-099 pre-product qualification gate. It records package and
platform facts, hashes, row statuses, booleans, counts, timings, bounded memory
measurements, and exact command metadata. It contains no environment values,
profile content, terminal output, captured stdout or stderr, command output,
credentials, tokens, or secrets.

The six collected rows are four native macOS host rows, one genuine Ubuntu
24.04 Linux ARM64 Docker row, and one native Windows Server 2025 x64 GitHub
Actions row. The parser and applicable environment slices pass on all six.
Every non-Windows pywinpty probe is an explicit `UNSUPPORTED_FAIL_CLOSED` host
refusal. The executed Windows row is `FAIL_CLOSED` because two mandatory native
rows failed. `BLOCKED` describes the overall Task 1 state; the other seven
unexecuted Windows rows remain `UNSUPPORTED_FAIL_CLOSED`.

textual-terminal source adaptation: none

## Package versions, hashes, and licenses

pyte==0.8.2
- sha256:85db42a35798a5aafa96ac4d8da78b090b2c933248819157fc0e6f78876a0135
- license: GNU Lesser General Public License v3 (LGPLv3); embedded `LICENSE` sha256:da7eabb7bafdf7d3ae5e9f223aa5bdc1eece45ac569dc21b3b037520b4464768
- wheel: `pyte-0.8.2-py3-none-any.whl`, 31,627 bytes, tag `py3-none-any`
- installed primary file: `pyte/__init__.py` sha256:5279e4cfba52135248b5ce33c7a915100046c6a679f4aff3d6f873643fb1eb44
- resolved dependency: `wcwidth==0.8.3`, wheel sha256:d5b73dba6158a595ec9370350e7f2637bcac8d6c5e4fde34f30fcffb6103a5e4, 331,669 bytes, license: MIT, embedded `LICENSE` sha256:70b98a95a2144eb70af8017fa8c6d95ce247e40867436e8bc649e137fe13d21a
- installed wcwidth primary file: `wcwidth/__init__.py` sha256:2feead2f63ec7862737414280fe225acb61f252c18ec7a53af4d330ecec29ae0

regex==2026.4.4
- sha256:1b1ce5c81c9114f1ce2f9288a51a8fd3aeea33a0cc440c415bf02da323aa0a76
- license: Apache-2.0 AND CNRI-Python; embedded `LICENSE.txt` sha256:bff55ef4cdcc8c14ce259f8e8ab60e264418440d6335f4dc138273fbd506144d
- wheel: `regex-2026.4.4-cp312-cp312-macosx_11_0_arm64.whl`, 289,628 bytes, tag `cp312-cp312-macosx_11_0_arm64`
- installed native extension: `regex/_regex.cpython-312-darwin.so` sha256:0b8e9ae442ad428e3fa5dda72054c1b369eb5035181e7cad5448e18c7431e78a
- installed package entry point: `regex/__init__.py` sha256:039934ae6f0b9fb1cab1f1bef2c11661e85e93ccc22847bab105b94065009925
- qualification: the pinned package requires Python 3.10 or newer, publishes
  regular-GIL CPython 3.11-3.14 wheels for the admitted macOS ARM64, Linux
  ARM64, and Windows AMD64 targets, and documents `\X` as conforming to UAX
  #29. Chatbook passes immutable strings no longer than 1,024 code points,
  does not request concurrent matching, and consumes at most 65 matches.

pywinpty==3.0.5
- sha256:af7a8720c78776ddd6259b71dd567944f766a6cd67f8d2887fbc4973967bacda
- license: MIT; embedded `LICENSE.txt` sha256:f878d4767f9ad2e43d3083efa00201b000ce937d9ee8626e00ba5c72aac951e2
- wheel: `pywinpty-3.0.5-cp311-cp311-win_amd64.whl`, 2,092,466 bytes, tag `cp311-cp311-win_amd64`
- qualification: native install and execution on Windows build 26100, CPython
  3.11.9 x64; installed `winpty/__init__.py`
  sha256:0a7e8f1e0c9c867049153c277c766d3e1394ef5bac7f4aeec46bd735ade0a3d0;
  installed `RECORD`
  sha256:6c382628f2e0ad8aa08419062476e5d94777d8c5f4c1659e6e0c356f1505f1b5

The pyte and wcwidth wheel, license, and installed-primary-file hashes are
identical across all six isolated rows. The pywinpty CPython 3.11 x64 wheel and
embedded-license hashes were verified before and after native installation;
the remaining regular-GIL Windows wheel matrix below was checked against PyPI
release metadata. Sources: [pyte 0.8.2 release JSON](https://pypi.org/pypi/pyte/0.8.2/json),
[wcwidth 0.8.3 release JSON](https://pypi.org/pypi/wcwidth/0.8.3/json),
[regex 2026.4.4 release JSON](https://pypi.org/pypi/regex/2026.4.4/json),
[pywinpty 3.0.5 release JSON](https://pypi.org/pypi/pywinpty/3.0.5/json), and
[pywinpty v3.0.5 source](https://github.com/andfoy/pywinpty/tree/v3.0.5).

The isolated-environment bootstrap inventory, which was supplied by each host
interpreter rather than downloaded as a task requirement, was: `pip==24.0` on
Linux 3.12 and macOS 3.11, `pip==25.0.1` on macOS 3.12, `pip==26.0.1` on
macOS 3.13, `pip==26.1.2` on macOS 3.14, and `setuptools==65.5.0` on macOS
3.11. Their exact installed-file hashes remain in the raw manifests and are not
presented as downloaded-artifact hashes.

## Qualification hosts and commands

Collection dates: 2026-08-29 through 2026-08-30 UTC (2026-08-29
America/Los_Angeles).

The macOS rows ran natively on macOS 15.6 build 24G84, Darwin 24.6.0,
Apple ARM64. The Linux row ran as CPython 3.12.3 on ARM64 Ubuntu 24.04 in
Docker, with Linux 6.12.76-linuxkit visible inside the container. The worktree
HEAD was `8c085f2ea2145f2f3469a5858b83418221bcd3e6`. Formatter base
`origin/dev` was resolved once to
`bf239298573c5148ba1d7805d67582fa9a4a4b34`.

The Windows row ran as CPython 3.11.9 x64 on the native `windows-2025`
GitHub-hosted runner, Windows 10.0.26100, in Actions run `33288554975`. Its
canonical generation was validated before upload and again after download.

| Row ID | Runtime | Interpreter | Generation | Exact source identity |
| --- | --- | --- | --- | --- |
| macos-arm64-py311 | native host | CPython 3.11.13 | `2238b1be6b07472dadb7ab3ee0256818` | `/private/tmp/tldw-task-22512-current-macos-arm64-py311.I529w4` |
| macos-arm64-py312 | native host | CPython 3.12.11 | `da323923a7f348b295cfc0af98b333a8` | `/private/tmp/tldw-task-22512-current-macos-arm64-py312.dyUMVp` |
| macos-arm64-py313 | native host | CPython 3.13.13 | `158028b2fab44117a4358f2a6d8f93e9` | `/private/tmp/tldw-task-22512-current-macos-arm64-py313.dalhl2` |
| macos-arm64-py314 | native host | CPython 3.14.6 | `10204b44af804f73ad13094935ae2a6c` | `/private/tmp/tldw-task-22512-current-macos-arm64-py314.GoF4Rh` |
| linux-arm64-py312 | Docker container `91d10c1c7cab` | CPython 3.12.3 | `3a382396e74f455892b1d887af84fb05` | host `/private/tmp/tldw-task-22512-current-linux-arm64-py312.zl5BUA/row`; container `/qualification/row` |
| win-amd64-py311 | native `windows-2025` runner | CPython 3.11.9 | `2b1adae613fb4d17bbb0be992bd10d50` | `D:\a\_temp\tldw-task-22512-win-amd64-py311-f34b7e35703c497aa3710711c702cccb` |

### Exact machine-recorded argv

Every one of the 36 raw JSON files stores the exact generating process in
`.command.argv` and `.command.working_directory`, plus the exact collection
process in `.collection_command.argv` and
`.collection_command.working_directory`. Those arrays are the authoritative
machine record; they are not inferred from the reproduction templates below.
For `.command.argv[0]`, `common.command_facts()` records the active
`sys.executable` of the invoked process and appends the original argument
suffix. Thus every probe names its isolated row interpreter, while
`prepare-row` names the selected host or container interpreter. A separate
runtime base-executable path is not retained and no shell alias spelling is
claimed.

The exact artifact-preparation argv for the six rows were:

```text
/Users/macbook-dev/.local/bin/python3.11 scripts/terminal_qualification/common.py prepare-row --row-id macos-arm64-py311 --row-dir /private/tmp/tldw-task-22512-current-macos-arm64-py311.I529w4 --requirement pyte==0.8.2 --requirement 'wcwidth>=0.2.14,<1' --json-out /private/tmp/tldw-task-22512-current-macos-arm64-py311.I529w4/artifacts.json
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python scripts/terminal_qualification/common.py prepare-row --row-id macos-arm64-py312 --row-dir /private/tmp/tldw-task-22512-current-macos-arm64-py312.dyUMVp --requirement pyte==0.8.2 --requirement 'wcwidth>=0.2.14,<1' --json-out /private/tmp/tldw-task-22512-current-macos-arm64-py312.dyUMVp/artifacts.json
/opt/homebrew/opt/python@3.13/bin/python3.13 scripts/terminal_qualification/common.py prepare-row --row-id macos-arm64-py313 --row-dir /private/tmp/tldw-task-22512-current-macos-arm64-py313.dalhl2 --requirement pyte==0.8.2 --requirement 'wcwidth>=0.2.14,<1' --json-out /private/tmp/tldw-task-22512-current-macos-arm64-py313.dalhl2/artifacts.json
/opt/homebrew/opt/python@3.14/bin/python3.14 scripts/terminal_qualification/common.py prepare-row --row-id macos-arm64-py314 --row-dir /private/tmp/tldw-task-22512-current-macos-arm64-py314.GoF4Rh --requirement pyte==0.8.2 --requirement 'wcwidth>=0.2.14,<1' --json-out /private/tmp/tldw-task-22512-current-macos-arm64-py314.GoF4Rh/artifacts.json
/usr/bin/python3 -B scripts/terminal_qualification/common.py prepare-row --row-id linux-arm64-py312 --row-dir /qualification/row --requirement pyte==0.8.2 --requirement 'wcwidth>=0.2.14,<1' --json-out /qualification/row/artifacts.json --runtime-kind docker --runtime-image ubuntu:24.04 --runtime-image-id sha256:561618e2c15bf2397621dd04f96926663a3b5616c189cf7e38db7e82f5c538ea --runtime-container-id 91d10c1c7cab
C:\hostedtoolcache\windows\Python\3.11.9\x64\python.exe -B scripts\terminal_qualification\common.py prepare-row --row-id win-amd64-py311 --row-dir D:\a\_temp\tldw-task-22512-win-amd64-py311-f34b7e35703c497aa3710711c702cccb --requirement pywinpty==3.0.5 --requirement pyte==0.8.2 --requirement 'wcwidth>=0.2.14,<1' --json-out D:\a\_temp\tldw-task-22512-win-amd64-py311-f34b7e35703c497aa3710711c702cccb\artifacts.json
```

The Linux generating commands used working directory `/worktree`. Its exact
Docker runtime metadata was image tag `ubuntu:24.04`, image ID
`sha256:561618e2c15bf2397621dd04f96926663a3b5616c189cf7e38db7e82f5c538ea`,
and container ID `91d10c1c7cab`. The six exact collection argv were:

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B scripts/terminal_qualification/common.py collect-row --row-dir /private/tmp/tldw-task-22512-current-macos-arm64-py311.I529w4 --evidence-root Docs/superpowers/reviews/evidence/task-22512/raw --replace
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B scripts/terminal_qualification/common.py collect-row --row-dir /private/tmp/tldw-task-22512-current-macos-arm64-py312.dyUMVp --evidence-root Docs/superpowers/reviews/evidence/task-22512/raw --replace
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B scripts/terminal_qualification/common.py collect-row --row-dir /private/tmp/tldw-task-22512-current-macos-arm64-py313.dalhl2 --evidence-root Docs/superpowers/reviews/evidence/task-22512/raw --replace
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B scripts/terminal_qualification/common.py collect-row --row-dir /private/tmp/tldw-task-22512-current-macos-arm64-py314.GoF4Rh --evidence-root Docs/superpowers/reviews/evidence/task-22512/raw --replace
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -B scripts/terminal_qualification/common.py collect-row --row-dir /private/tmp/tldw-task-22512-current-linux-arm64-py312.zl5BUA/row --evidence-root Docs/superpowers/reviews/evidence/task-22512/raw --replace
D:\a\_temp\tldw-task-22512-win-amd64-py311-f34b7e35703c497aa3710711c702cccb\venv\Scripts\python.exe -B scripts\terminal_qualification\common.py collect-row --row-dir D:\a\_temp\tldw-task-22512-win-amd64-py311-f34b7e35703c497aa3710711c702cccb --evidence-root D:\a\_temp\task22512-collected
```

The POSIX rows used the same explicit five probes with their concrete row
interpreter and row directory: default, Bash, and Zsh environment probes plus
the pyte and pywinpty probes. The Windows row used six: default, PowerShell,
and CMD environment probes plus artifact, pyte, and pywinpty probes. Exact
per-probe argv are retained in the corresponding raw files and bound by the
hashes in the Raw evidence section. Every retained command names the same fresh
isolated row venv used by its probe.

The provenance command was:

```bash
../../.venv/bin/python -B -m pytest Tests/test_probe_import_provenance.py -q
```

The immutable formatter snapshot used all 16 paths in
`format-baseline.json`. Verification reads the stored base SHA rather than
resolving `origin/dev` again:

```bash
../../.venv/bin/python scripts/terminal_qualification/format_ratchet.py verify \
  --head HEAD \
  --baseline Docs/superpowers/reviews/evidence/task-22512/format-baseline.json
```

### Reproduction templates

These templates create new row identities and therefore cannot reproduce the
old temporary directory or container ID byte-for-byte. Their result schemas and
facts are comparable; the raw `.command` and `.collection_command` objects
remain the exact record of the runs summarized here.

For a native POSIX host, use the concrete interpreter and row ID being claimed,
then run the five explicit probes shown in
`scripts/terminal_qualification/README.md`.

For the recorded Linux architecture and image, first verify that the local
image tag resolves to the recorded immutable image ID, then run the isolated
container row. This template deliberately fails if the tag moved or the exact
image is unavailable:

```bash
TASK22512_LINUX_ROOT=$(mktemp -d /tmp/tldw-task-22512-linux-arm64-py312.XXXXXX)
TASK22512_WORKTREE=$PWD
TASK22512_IMAGE_ID=sha256:561618e2c15bf2397621dd04f96926663a3b5616c189cf7e38db7e82f5c538ea
test "$(docker image inspect ubuntu:24.04 --format '{{.Id}}')" = "$TASK22512_IMAGE_ID"
test "$(docker image inspect ubuntu:24.04 --format '{{.Os}}/{{.Architecture}}')" = "linux/arm64"
docker run --platform linux/arm64 --rm \
  --mount type=bind,src="$TASK22512_WORKTREE",dst=/worktree,readonly \
  --mount type=bind,src="$TASK22512_LINUX_ROOT",dst=/qualification \
  --workdir /worktree \
  "$TASK22512_IMAGE_ID" \
  bash -lc '
    set -eu
    test "$(uname -m)" = aarch64
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-venv python3-pip zsh vim less procps
    TASK22512_CONTAINER_ID=$(hostname)
    python3 -B scripts/terminal_qualification/common.py prepare-row --row-id linux-arm64-py312 --row-dir /qualification/row --requirement pyte==0.8.2 --requirement "wcwidth>=0.2.14,<1" --json-out /qualification/row/artifacts.json --runtime-kind docker --runtime-image ubuntu:24.04 --runtime-image-id sha256:561618e2c15bf2397621dd04f96926663a3b5616c189cf7e38db7e82f5c538ea --runtime-container-id "$TASK22512_CONTAINER_ID"
    /qualification/row/venv/bin/python -B scripts/terminal_qualification/environment_probe.py --shell default --json-out /qualification/row/env-default.json
    /qualification/row/venv/bin/python -B scripts/terminal_qualification/environment_probe.py --shell bash --json-out /qualification/row/env-bash.json
    /qualification/row/venv/bin/python -B scripts/terminal_qualification/environment_probe.py --shell zsh --json-out /qualification/row/env-zsh.json
    /qualification/row/venv/bin/python -B scripts/terminal_qualification/pyte_probe.py --artifact-manifest /qualification/row/artifacts.json --json-out /qualification/row/pyte.json
    if /qualification/row/venv/bin/python -B scripts/terminal_qualification/pywinpty_probe.py --artifact-manifest /qualification/row/artifacts.json --json-out /qualification/row/pywinpty.json; then exit 1; else test "$?" -eq 1; fi
  '
../../.venv/bin/python -B scripts/terminal_qualification/common.py collect-row \
  --row-dir "$TASK22512_LINUX_ROOT/row" \
  --evidence-root Docs/superpowers/reviews/evidence/task-22512/raw \
  --replace
```

## Wheel matrix

The universal pyte and wcwidth wheels installed offline on native macOS ARM64
CPython 3.11, 3.12, 3.13, and 3.14 and Linux ARM64 CPython 3.12. Wheel presence
is packaging information, not native Windows qualification.

The pinned regex release's official PyPI metadata was checked for each regular-
GIL interpreter and admitted target. These compiled wheel rows establish
artifact availability; the macOS ARM64 CPython 3.12 row was also installed and
hashed above.

| Python | Platform | Filename | Size | SHA-256 |
| --- | --- | --- | ---: | --- |
| 3.11 | macOS ARM64 | `regex-2026.4.4-cp311-cp311-macosx_11_0_arm64.whl` | 289,225 | `6aa809ed4dc3706cc38594d67e641601bd2f36d5555b2780ff074edfcb136cf8` |
| 3.11 | Linux ARM64 | `regex-2026.4.4-cp311-cp311-manylinux2014_aarch64.manylinux_2_17_aarch64.manylinux_2_28_aarch64.whl` | 792,434 | `33424f5188a7db12958246a54f59a435b6cb62c5cf9c8d71f7cc49475a5fdada` |
| 3.11 | Windows AMD64 | `regex-2026.4.4-cp311-cp311-win_amd64.whl` | 278,399 | `9542ccc1e689e752594309444081582f7be2fdb2df75acafea8a075108566735` |
| 3.12 | macOS ARM64 | `regex-2026.4.4-cp312-cp312-macosx_11_0_arm64.whl` | 289,628 | `1b1ce5c81c9114f1ce2f9288a51a8fd3aeea33a0cc440c415bf02da323aa0a76` |
| 3.12 | Linux ARM64 | `regex-2026.4.4-cp312-cp312-manylinux2014_aarch64.manylinux_2_17_aarch64.manylinux_2_28_aarch64.whl` | 796,651 | `760ef21c17d8e6a4fe8cf406a97cf2806a4df93416ccc82fc98d25b1c20425be` |
| 3.12 | Windows AMD64 | `regex-2026.4.4-cp312-cp312-win_amd64.whl` | 277,768 | `db0ac18435a40a2543dbb3d21e161a6c78e33e8159bd2e009343d224bb03bb1b` |
| 3.13 | macOS ARM64 | `regex-2026.4.4-cp313-cp313-macosx_11_0_arm64.whl` | 289,487 | `8fae3c6e795d7678963f2170152b0d892cf6aee9ee8afc8c45e6be38d5107fe7` |
| 3.13 | Linux ARM64 | `regex-2026.4.4-cp313-cp313-manylinux2014_aarch64.manylinux_2_17_aarch64.manylinux_2_28_aarch64.whl` | 796,646 | `298c3ec2d53225b3bf91142eb9691025bab610e0c0c51592dde149db679b3d17` |
| 3.13 | Windows AMD64 | `regex-2026.4.4-cp313-cp313-win_amd64.whl` | 277,733 | `3384df51ed52db0bea967e21458ab0a414f67cdddfd94401688274e55147bb81` |
| 3.14 | macOS ARM64 | `regex-2026.4.4-cp314-cp314-macosx_11_0_arm64.whl` | 289,692 | `76d67d5afb1fe402d10a6403bae668d000441e2ab115191a804287d53b772951` |
| 3.14 | Linux ARM64 | `regex-2026.4.4-cp314-cp314-manylinux2014_aarch64.manylinux_2_17_aarch64.manylinux_2_28_aarch64.whl` | 796,979 | `e7cd3e4ee8d80447a83bbc9ab0c8459781fa77087f856c3e740d7763be0df27f` |
| 3.14 | Windows AMD64 | `regex-2026.4.4-cp314-cp314-win_amd64.whl` | 280,992 | `e0aab3ff447845049d676827d2ff714aab4f73f340e155b7de7458cf53baa5a4` |

| Python | Windows architecture | Filename | Size | SHA-256 | Native status |
| --- | --- | --- | ---: | --- | --- |
| 3.11 | amd64 | `pywinpty-3.0.5-cp311-cp311-win_amd64.whl` | 2,092,466 | `af7a8720c78776ddd6259b71dd567944f766a6cd67f8d2887fbc4973967bacda` | FAIL_CLOSED |
| 3.11 | arm64 | `pywinpty-3.0.5-cp311-cp311-win_arm64.whl` | 818,395 | `c2406f54f699eab75953fb75ce805f2ae55a33a957cd070890abd454fb4b7680` | UNSUPPORTED_FAIL_CLOSED |
| 3.12 | amd64 | `pywinpty-3.0.5-cp312-cp312-win_amd64.whl` | 2,090,915 | `d62946adf14b15b54c0b8d785f93fe18b04da23f4ad59e2e8c4612646e9abd23` | UNSUPPORTED_FAIL_CLOSED |
| 3.12 | arm64 | `pywinpty-3.0.5-cp312-cp312-win_arm64.whl` | 815,934 | `e9391c05fbfa7a992a97e831fc6849887b4014a614192e3d984a7ca59592b376` | UNSUPPORTED_FAIL_CLOSED |
| 3.13 | amd64 | `pywinpty-3.0.5-cp313-cp313-win_amd64.whl` | 2,090,471 | `48db1b0ad9d0a1b81dcaaa7163a99a7808deaceb0c1b2344716dc1fc090c3c4c` | UNSUPPORTED_FAIL_CLOSED |
| 3.13 | arm64 | `pywinpty-3.0.5-cp313-cp313-win_arm64.whl` | 815,518 | `2c6008fb2d3774b48693b2fcb7f2cc317ade9dc581289a964ffeeaf81307c9b5` | UNSUPPORTED_FAIL_CLOSED |
| 3.14 | amd64 | `pywinpty-3.0.5-cp314-cp314-win_amd64.whl` | 2,090,663 | `03bb3c16d691d9242267201830bcd0e64a9b663170e9042bc84b210da9de15ac` | UNSUPPORTED_FAIL_CLOSED |
| 3.14 | arm64 | `pywinpty-3.0.5-cp314-cp314-win_arm64.whl` | 815,700 | `89c5c6ef08997a3b4b277b214a35fe15cab4dd6d119f0140aa71df5b1168fdbc` | UNSUPPORTED_FAIL_CLOSED |

Published free-threaded 3.13t/3.14t artifacts and Python 3.10 artifacts are not
claimed by this matrix.

## Platform matrix

| Host row | Artifact preparation | Parser | Environment | Native pywinpty |
| --- | --- | --- | --- | --- |
| macos-arm64-py311 | PASS | PASS | default/Bash/Zsh PASS | UNSUPPORTED_FAIL_CLOSED |
| macos-arm64-py312 | PASS | PASS | default/Bash/Zsh PASS | UNSUPPORTED_FAIL_CLOSED |
| macos-arm64-py313 | PASS | PASS | default/Bash/Zsh PASS | UNSUPPORTED_FAIL_CLOSED |
| macos-arm64-py314 | PASS | PASS | default/Bash/Zsh PASS | UNSUPPORTED_FAIL_CLOSED |
| linux-arm64-py312 | PASS | PASS | default/Bash/Zsh PASS | UNSUPPORTED_FAIL_CLOSED |
| win-amd64-py311 | PASS | PASS | default/PowerShell/CMD PASS | FAIL_CLOSED |
| win-arm64-py311 | not run | not run on Windows | PowerShell/CMD not run | UNSUPPORTED_FAIL_CLOSED |
| win-amd64-py312 | not run | not run on Windows | PowerShell/CMD not run | UNSUPPORTED_FAIL_CLOSED |
| win-arm64-py312 | not run | not run on Windows | PowerShell/CMD not run | UNSUPPORTED_FAIL_CLOSED |
| win-amd64-py313 | not run | not run on Windows | PowerShell/CMD not run | UNSUPPORTED_FAIL_CLOSED |
| win-arm64-py313 | not run | not run on Windows | PowerShell/CMD not run | UNSUPPORTED_FAIL_CLOSED |
| win-amd64-py314 | not run | not run on Windows | PowerShell/CMD not run | UNSUPPORTED_FAIL_CLOSED |
| win-arm64-py314 | not run | not run on Windows | PowerShell/CMD not run | UNSUPPORTED_FAIL_CLOSED |

## API and backend identity

The downloaded 3.0.5 x64 wheel exposes the low-level `winpty.PTY` type from
`winpty/_winpty`. Its stub exposes `spawn`, `read(blocking=False)`, `write`,
`set_size`, `isalive`, `iseof`, `cancel_io`, `pid`, and `fd`.
`winpty.Backend.ConPTY` is the only allowed constructor selector. Source
references: [low-level stub](https://github.com/andfoy/pywinpty/blob/v3.0.5/winpty/_winpty.pyi)
and [backend enum](https://github.com/andfoy/pywinpty/blob/v3.0.5/winpty/enums.py).

Static source guards reject the high-level process wrappers, legacy backend,
and ordinary-pipe fallback. The Windows x64 CPython 3.11 row proved native
low-level construction and behavior; its mandatory stream-semantics failures
still reject the dependency as a whole.

Host-independent tests pass for the intended native contracts, but those tests
use fake terminals, fake handles, static guards, and direct helper exercise;
they are not native behavior evidence. The output-credit helper returns bytes
from an actual `PTY.read(blocking=True)`, permits exactly one unacknowledged
chunk capped at 64 KiB, and requires explicit acknowledgement. Before handoff,
the concurrency helper drains and acknowledges startup output from a dedicated
fourth terminal until it is quiet, requires the real read, write, resize, and
cancellation method paths all to enter, permits the naturally synchronous
write/resize/cancel calls to return, and requires those three calls to be
complete. The known blocking read alone must remain unresolved. Entered,
completed-at-handoff, and completed-post-close outcomes are separate facts;
priority terminal close must terminate the read boundedly. There is no
requirement that synchronous calls remain unresolved or return after close.

Post-exit drain continues until `iseof()` or a one-second cutoff and checks
multi-buffer frame order and digest separately from EOF. Normal cleanup first
retains waitable handles for the complete stable Job-member set, closes the
Job, requires positive and exactly equal expected, retained, and
`WAIT_OBJECT_0` counts with `normal_cleanup_all_wait_object_0=true`, and reaps
the controller process. Observations are built as a candidate transaction and
published only after that cleanup succeeds. Any retention/open/wait/reap
exception, timeout, `WAIT_FAILED`, unknown wait result, false all-waited fact,
or partial count invalidates the candidate and fails the native result closed.

The application-crash fixture is specifically supervisor -> separate
app/controller -> admitted worker. The app/controller is the sole owner of the
non-inheritable kill-on-close Job handle and actually aborts. Before the abort,
the supervisor owns only pre-opened `SYNCHRONIZE` handles for the stable known
app/worker descendant set; it never owns or duplicates the Job handle. The app
abort closes that sole Job handle, after which every retained wait must return
`WAIT_OBJECT_0`; timeout, wait failure, or an unknown result fails closed.
Terminal-child crash has separate facts and cannot satisfy
`app_crash_observed`. The native row passed this complete app-crash contract:
seven stable known descendants were retained and every wait returned
`WAIT_OBJECT_0` after the sole Job owner aborted.

| Binding row | Requirement | Result |
| --- | --- | --- |
| package-pywinpty-3.0.5 | MANDATORY | PASS |
| windows-platform-floor | MANDATORY | PASS |
| windows-low-level-api | MANDATORY | PASS |
| windows-conpty-only | MANDATORY | PASS |
| windows-job-admission-membership | MANDATORY | PASS |
| windows-handle-inheritance | MANDATORY | PASS |

## Parser matrix

All parser rows passed in all six isolated environments with `TERM=linux`.
The five POSIX rows used real available shell and full-screen programs; the
Windows row used bounded PowerShell, CMD, editor, pager, and monitor parser
fixtures because this gate qualifies parser behavior separately from the real
Windows environment and ConPTY probes. Captures were parsed in memory and
discarded, and bounded cleanup reaped each real capture's process group.

| Binding row | Requirement | Result | Content-free summary |
| --- | --- | --- | --- |
| package-pyte-0.8.2 | MANDATORY | PASS | version, wheel hash, and installed primary-file hash bound to manifest |
| package-regex-2026.4.4 | MANDATORY | PASS | pin, license, supported wheel matrix, and bounded UAX #29 use recorded |
| parser-shell-captures | MANDATORY | PASS | two unique available shells captured in every row |
| parser-powershell-cmd-fixtures | MANDATORY | PASS | two bounded fixtures |
| parser-full-screen-programs | MANDATORY | PASS | editor, pager, and monitor classes passed |
| parser-unicode-cells | MANDATORY | PASS | wide placeholders 2; combining normalization true; cursor column 6 |
| parser-alternate-screen | MANDATORY | PASS | DEC 1049 entry/exit used isolated buffers and restored the primary buffer |
| parser-resize | MANDATORY | PASS | 80x24 to 120x40 |
| parser-bracketed-paste | MANDATORY | PASS | bounded fixture |
| parser-terminal-queries | MANDATORY | PASS | bounded fixture |
| parser-malformed-controls | MANDATORY | PASS | handled without exception |
| parser-incomplete-sequence-bounds | MANDATORY | PASS | 10 rejected fixtures; 6 accepted fixtures; CSI numeric parameters at most 4 digits and 9,999; non-CSI encoded controls at most 16 bytes |
| parser-mutable-collections | MANDATORY | PASS | zero unclassified collections |

Mutable collection classifications: `charset` and `mode` static; `buffer`,
`dirty`, and `tabstops` viewport-bounded; `savepoints` adapter-capped.

## Environment key sets and profile behavior

All twelve native macOS environment probes began with exactly these key names:
`HOME, LANG, LC_ALL, LC_CTYPE, LOGNAME, PATH, SHELL, TERM, TMPDIR, USER`.
The three Linux probes began with exactly:
`HOME, LC_CTYPE, LOGNAME, PATH, SHELL, TERM, TMPDIR, USER`. Every sensitive
initial-key count was zero. Values were not retained.

Every POSIX host row found two standard account profile candidates. Actual normal
account startup and command discovery completed for the default account shell,
Bash, and Zsh after the probe waited for interactive readiness. Separate
controlled temporary-home standard-profile fixtures proved startup execution,
intentional sensitive-key repopulation, and command discovery.

The executed Windows harness preserved normal startup through a supervisor-
owned disposable local account and loaded profile, not through `USERPROFILE`
alone. The supervisor launches only a waiting bootstrap under that account
with `CreateProcessWithLogonW(LOGON_WITH_PROFILE)`, assigns it to the
kill-on-close Job, verifies membership plus disposable identity/profile/hive,
and releases it only after admission. The bootstrap self-verifies membership
before it writes profile fixtures or starts the selected shell. PowerShell
launches with `-NoLogo` and deliberately without `-NoProfile` or
`-NonInteractive`. CMD launches with `/Q` and deliberately without `/D`; its
`Command Processor` `AutoRun` fixture is written only through the verified
disposable process's `RegOpenCurrentUser`. The interactive user's
`HKEY_CURRENT_USER` is never opened for write. Supervisor-owned account/profile
cleanup runs in `finally`, including after probe crash. Missing account/profile
privileges, alternate-user launch, Job admission, or identity/profile/hive
verification fails closed before the real shell starts. Host-independent tests
pass for the default-shell policy and both named startup recipes. Native
default, PowerShell, and CMD execution also passed on the collected Windows
x64 row. Each began with exactly `APPDATA, COMSPEC, HOMEDRIVE, HOMEPATH,
LOCALAPPDATA, PATH, PATHEXT, PROGRAMDATA, PROGRAMFILES, PROGRAMFILES(X86),
PROGRAMW6432, SYSTEMROOT, TEMP, TERM, TMP, USERNAME, USERPROFILE, WINDIR`; the
sensitive initial-key count was zero. Only booleans, counts, key names, and byte
counts are retained.

| Binding row | Requirement | Result |
| --- | --- | --- |
| environment-default-shell | MANDATORY | PASS |
| environment-bash | MANDATORY | PASS |
| environment-zsh | MANDATORY | PASS |
| environment-powershell | MANDATORY | PASS |
| environment-cmd | MANDATORY | PASS |

PowerShell and CMD profile/module discovery passed natively and were not
substituted by POSIX fixtures.

## I/O, EOF, and output integrity

The native Windows x64 row passed real `PTY.read` credit/ack bookkeeping, the
32 KiB upstream internal-buffer identity, quiet-terminal startup drain,
four-operation entry, priority close, and exact retained-handle cleanup. It
measured 621 acknowledged chunks with one reader, one outstanding credit, and
an 8,760-byte maximum delivered chunk under the 64 KiB cap. Normal cleanup
retained and waited both expected processes with `WAIT_OBJECT_0`.

The same supported host failed the mandatory stream semantics. DEC 1049 entry,
exit, Unicode round trip, and primary restoration were observed, but alternate
content was not isolated. The abrupt-exit fixture proved process death and a
bounded drain but did not observe EOF, did not complete the ordered payload,
and retained only 26,303 bytes; digest and output-integrity facts were false.
The terminal-child crash likewise proved process death without stream EOF.
These are `FAIL_CLOSED` results, not skipped or inferred failures.

| Binding row | Requirement | Result |
| --- | --- | --- |
| windows-one-credit-bounded-read | MANDATORY | PASS |
| windows-concurrent-io-close | MANDATORY | PASS |
| windows-profile-module-discovery | MANDATORY | PASS |
| windows-unicode-alternate-screen | MANDATORY | FAIL_CLOSED |
| windows-app-crash-descendant-cleanup | MANDATORY | PASS |
| windows-eof-output-integrity | MANDATORY | FAIL_CLOSED |

## Memory and resource bounds

Each parser stress row fed 2,000 rows into a 300x120 viewport. Tracemalloc peaks
were 689,899 bytes (macOS 3.11), 687,643 bytes (macOS 3.12), 689,931 bytes
(macOS 3.13), 687,094 bytes (macOS 3.14), 683,139 bytes (Linux 3.12), and
889,787 bytes (Windows 3.11), all
below the 64 MiB probe bound. Parser-probe peak RSS was 33,341,440,
32,309,248, 39,469,056, 41,205,760, and 26,337,280 bytes respectively.

The four-session 256 MiB Windows limit is an incremental managed-overhead
measurement. Its baseline includes controller and worker; the sample includes
controller, worker/IPC, and helpers while excluding exactly the four fixture
workloads whose population is separately validated in the Job. The native row
measured 73,965,568 bytes across one controller, one worker, and five helpers,
below the 268,435,456-byte limit; all four fixture workloads were identified
and excluded as designed.

| Binding row | Requirement | Result |
| --- | --- | --- |
| four-session-managed-rss | MANDATORY | PASS |

## Raw evidence

The retained set is exactly 36 JSON files and six `.current-generation`
manifests: six files and one commit marker in each of six row directories.
`collect-row` validated each complete source sibling set against the current
shape-specific schema and semantic rules before publishing, bound one
row/platform/runtime identity, rejected source generation metadata, then
injected one generation ID and exact collection command across all six files.
Each committed manifest names and hashes exactly those six canonical siblings.

Publication is recoverable and fail closed, not universally multi-file atomic.
For these stale pre-marker migrations, the collector first validated the exact
complete legacy generation, copied its six bytes into recovery, created and
validated a recovery hash manifest, and wrote the pending marker. It then
replaced all six staged files and committed `.current-generation` last.
Catchable replacement errors restore the complete previous generation; abrupt
death leaves the pending marker so no partial or mixed set can validate as
current, and recovery restores the previous generation. A restored legacy set
still requires a later successful marker commit before it is accepted as
current. The current audit found no pending marker or recovery residue.

The current audit revalidated each retained generation and its matching source
directory, rehashed the exact source wheels, and confirmed every consuming
probe's manifest binding, including pre-install and post-install artifact
hashes and installed primary/`RECORD` hashes.

Probe and result-row IDs select exact schemas, so fields valid only for another
probe or row are rejected along with unknown keys, wrong types or placement,
and nested command objects. Credential assignments, JWT-shaped or
Authorization/Bearer values, and GitHub-style tokens are rejected even in
otherwise allowed license or argv fields. Wheels, virtual environments,
captured output, and profiles remain outside this artifact. Each line below was
recalculated from the current file bytes.

raw/linux-arm64-py312/.current-generation | sha256:308f3fdf01064ce2abc6c6f3ef9e377750675f1eaed3563f84a4e46d74677553
raw/linux-arm64-py312/linux-arm64-py312-artifacts.json | sha256:c386ceecc2f73444b4b32b8e64aa2f210c59044ff6c1ca350d31bc0bbe15ce3d
raw/linux-arm64-py312/linux-arm64-py312-environment-bash.json | sha256:4f8e0080338ae25cc2a59d1d5f452897e3ab1131efd6b41f7afc5d6865d3ee51
raw/linux-arm64-py312/linux-arm64-py312-environment-default.json | sha256:f4aab1916861e8fbc8a26f65af065f9faab40ab4383ff80b59305c00b170502c
raw/linux-arm64-py312/linux-arm64-py312-environment-zsh.json | sha256:ad53dd237bb2baf90abf3fc947a6ad788b02ba1795784194693b5e5a210e0a41
raw/linux-arm64-py312/linux-arm64-py312-pyte.json | sha256:e6a4d6bab2d85ef5de9a7aed84d3d9419f2ee41de6b03638b6f1d93c0041917d
raw/linux-arm64-py312/linux-arm64-py312-pywinpty.json | sha256:c29d21e7341079be408984cf934326637e030cd2badfb4fe8e892b35473649b9
raw/macos-arm64-py311/.current-generation | sha256:308090e1fb933e78058cf6dab0f72164d3b2a4b4c42bcddf82d405a06335c24d
raw/macos-arm64-py311/macos-arm64-py311-artifacts.json | sha256:724b0b669c258d2450ad60b71b47047a631a4c0d923f9470af775d7afb8f475e
raw/macos-arm64-py311/macos-arm64-py311-environment-bash.json | sha256:217eeb651ca999f71439c9d68519788260e14f5944291306a4dc7c617872c9ef
raw/macos-arm64-py311/macos-arm64-py311-environment-default.json | sha256:a36b79bbc9a920a868ef3300a7ca849a75381b9db5836aaa3512e477a8dd877f
raw/macos-arm64-py311/macos-arm64-py311-environment-zsh.json | sha256:97756e5673fb846268b4be777d3365d905b8459906e6f4ed3fa6d49a8971b1ae
raw/macos-arm64-py311/macos-arm64-py311-pyte.json | sha256:66dfc65b7940a80ec75e8759e9b180c8e09f0a3b4fb4b785e9bf5874b79be0b6
raw/macos-arm64-py311/macos-arm64-py311-pywinpty.json | sha256:c522b2445912d7e7e9e641136eb695162fdc48cb32d0774c3c65f133015b450a
raw/macos-arm64-py312/.current-generation | sha256:0e04697a6bd19c1da4c60950d69e719cc6345e21be3265544081928dbbabe26c
raw/macos-arm64-py312/macos-arm64-py312-artifacts.json | sha256:27361df3767c531e0070b33e523a2d7d69bbcb29873286518ed3452330eec496
raw/macos-arm64-py312/macos-arm64-py312-environment-bash.json | sha256:155f723b766b0695cf2fb0a59130afffbecd67c70c3f554cbe4726feacc40397
raw/macos-arm64-py312/macos-arm64-py312-environment-default.json | sha256:b9750d4aacddf2c4bc2386fd897804930e5f553da2242c8f8425cdac766c10a1
raw/macos-arm64-py312/macos-arm64-py312-environment-zsh.json | sha256:3765e7563b9fdfea36e920668246dd81df6cb57549308c7f6a8a8e228565f1f0
raw/macos-arm64-py312/macos-arm64-py312-pyte.json | sha256:a35060740b7b174ccbe6f84eda8f1f1330390025367854bd17f88f47fe956435
raw/macos-arm64-py312/macos-arm64-py312-pywinpty.json | sha256:ec5bf563327e463cf710be0ee190e170def05081716555cf211cdb565282ec07
raw/macos-arm64-py313/.current-generation | sha256:b1c58ae2045c5c4b33d94a80843f25d05758d6fa7eff19201adece865f1fd616
raw/macos-arm64-py313/macos-arm64-py313-artifacts.json | sha256:26e8ddbf4784bbb1dd78cb3cb090d5f9e3aa009922beebe57e871c636f567a28
raw/macos-arm64-py313/macos-arm64-py313-environment-bash.json | sha256:2d5551fe54e03f44c8d8cb077fc61835a94ac2c5256cfb0e91edd4718d00e57b
raw/macos-arm64-py313/macos-arm64-py313-environment-default.json | sha256:58a70848015d835b0becd355314755e8cebe37497792fba6732453d8b5b6e50e
raw/macos-arm64-py313/macos-arm64-py313-environment-zsh.json | sha256:191e3f42a0a6cdad5a1eafbf91003eaeaa8fd881993042db0c2d7b8daa4e5958
raw/macos-arm64-py313/macos-arm64-py313-pyte.json | sha256:00e3c647a440034961a5dd6e4ad4162c82ff948fb4c7689baadd013467343cca
raw/macos-arm64-py313/macos-arm64-py313-pywinpty.json | sha256:9cb976174aa7fb3ad0888493839a2076212f5115e782a2116818e563d658deb5
raw/macos-arm64-py314/.current-generation | sha256:7ac7b9109e3edeed4216820a4e562a8200fc90d4230a5d3a851dbbcd695cd795
raw/macos-arm64-py314/macos-arm64-py314-artifacts.json | sha256:cb89ca8580b946620b3f08eeb919bc9d66bcd9301d93033978db2eba63a9ff3c
raw/macos-arm64-py314/macos-arm64-py314-environment-bash.json | sha256:b5e74cec6730607971753fac9f1c61703f05e84b574a977df847540c1093fe70
raw/macos-arm64-py314/macos-arm64-py314-environment-default.json | sha256:f8d739b36edb7ea9cccf3022d248fbf03e040d355f9bab28bddb358e2d43a86e
raw/macos-arm64-py314/macos-arm64-py314-environment-zsh.json | sha256:97844b966954a4d20a4446f35f29a1415d45d602f360eddd8621138d76251efb
raw/macos-arm64-py314/macos-arm64-py314-pyte.json | sha256:ccf0957fcad56e6c268a07a7b2b0429c2f142185e6e80b4b8ac6e385f790dd9a
raw/macos-arm64-py314/macos-arm64-py314-pywinpty.json | sha256:e6958e4603e940fd7bd4d8725023efb8818b408d50156b991a923d81a917065e
raw/win-amd64-py311/.current-generation | sha256:32b3e957d2389ad7d994a4df4685dbb249e0751b02ad7cc47238bc8c72c1fbd5
raw/win-amd64-py311/win-amd64-py311-artifacts.json | sha256:c78fe5964d120e3a05122b0e1310a57fae345d537270f242bbe16e86ae929fd1
raw/win-amd64-py311/win-amd64-py311-environment-cmd.json | sha256:2688e03e4acecb8401adb1960cfd521656334989dde7b1edb699455bac00aaef
raw/win-amd64-py311/win-amd64-py311-environment-default.json | sha256:389bd05738ab3875da8ab0d6d6e35e0c41557e4a86b0724d83dacb306870db41
raw/win-amd64-py311/win-amd64-py311-environment-powershell.json | sha256:f74b996d8e24d6011e07b3caa2447c2a365b523b02d66eb4c2b3beba0b8a1436
raw/win-amd64-py311/win-amd64-py311-pyte.json | sha256:4d4e146c9030726ca756e9eec4b78ea7650730250b2f4352c407f084d70bda04
raw/win-amd64-py311/win-amd64-py311-pywinpty.json | sha256:4925048d4be6f75ee3a3a2325a806871f091c1d0f1e948a1a9914a2c0caab855

## Task 8 POSIX controlling-PTY backend qualification

The native implementation qualification was executed on `Darwin 24.6.0 arm64`
(`macOS-15.6-arm64-arm-64bit`) with CPython `3.12.11` from
`../../.venv/bin/python`. The earlier portable qualification was executed in the
`linux/arm64` Docker image `ubuntu:24.04`, exact inspected local image ID
`sha256:561618e2c15bf2397621dd04f96926663a3b5616c189cf7e38db7e82f5c538ea`,
container ID
`056e6751a000054d702b8f6c4842e77ff4afe2e1f9d3df4cce958f5b4115ff30`.
That container reported Ubuntu 24.04.4 LTS, `aarch64`, kernel
`6.12.76-linuxkit`, and Python 3.12.3.

The Linux repository mount was read-only at `/worktree`; `/qualification` was
a disposable tmpfs. Tests ran as the ordinary `qualification` account with
uid/gid 1001, `LANG=C.UTF-8`, and `LC_ALL=C.UTF-8`. `tini==0.19.0` ran via
`docker exec` as an explicit `/usr/bin/tini -s --` subreaper ancestor of
pytest; it was not container PID 1. This is qualification harness plumbing: the
container's bare `sleep` PID 1 was non-reaping and could not provide valid
cleanup-proof evidence. Declared core
requirements and the focused test dependencies were installed into a
disposable virtual environment. Relevant installed versions were
`psutil==7.2.2`, `pyte==0.8.2`, `wcwidth==0.8.3`, `pydantic==2.13.5`,
`regex==2026.4.4`, `textual==8.2.8`, `pytest==9.1.1`,
`pytest-asyncio==1.4.0`, `pytest-timeout==2.4.0`, and `httpx==0.28.1`.

The final corrected Linux qualification used container
`task22512-posix-linux-final-20260831`, exact container ID
`6fce5575c51777aef6dbf4efaf78cacd65cb425f8df3f1901d6b59a1e90fb785`, and
the same exact local image ID
`sha256:561618e2c15bf2397621dd04f96926663a3b5616c189cf7e38db7e82f5c538ea`.
It reported Ubuntu 24.04.4 LTS, `aarch64`, kernel `6.12.76-linuxkit`, and
Python 3.12.3. Docker `init=true` supplied PID 1 as
`/sbin/docker-init -- sleep infinity`. Tests ran as ordinary uid/gid 1001
against the read-only repository. Scratch space was tmpfs; because that tmpfs
was mounted `noexec`, the executable disposable virtual environment was moved
to `/opt/task22512-venv`.

| Qualification row | Executed command | Observed result |
| --- | --- | --- |
| Assertion-level launcher-order RED before production implementation | `PYTHONPATH="$PWD" ../../.venv/bin/python -B -m pytest Tests/Terminal/test_posix_backend.py -q` | RED: 12 collected; 7 failed and 5 errored. The real helper assertion reported `launcher did not report its gated identity`; collection/import succeeded. |
| Unrelated-Darwin-enumeration regression RED | `PYTHONPATH="$PWD" ../../.venv/bin/python -B -m pytest Tests/Terminal/test_posix_backend.py::test_unrelated_darwin_enumeration_denial_does_not_poison_death_proof -x -vv --basetemp=/private/tmp/task22512-regression-red-1` | RED: 1 failed, 1 warning in 1.92 seconds. The real shell was reaped, PTY EOF was true, and the owned census was empty, but cleanup returned `process_dead=False`. |
| Cleanup/input/death-proof/crash/job-control review RED | `PYTHONPATH="$PWD" ../../.venv/bin/python -B -m pytest Tests/Terminal/test_posix_backend.py -q --basetemp=/private/tmp/task22512-gap-red-1 -k 'short_write_accepts or eagain_write_is_buffered or pending_input_bound or cleanup_continuous_output or process_dead_requires_pty_eof or explicit_raw_cleanup_discard or real_foreground_background_groups or app_crash_master_close'` | RED: 8 failed, 13 deselected, and 1 warning in 9.62 seconds. The failures exposed short-write and `EAGAIN` rejection, immediate pending-input rejection, multiple cleanup reads in one turn, `process_dead=True` without PTY EOF, the raw-cleanup regression stopping on that same incorrect death proof, the absent PGID-transition handshake, and missing `shell_birth`. |
| Focused review regressions | `PYTHONPATH="$PWD" ../../.venv/bin/python -B -m pytest Tests/Terminal/test_posix_backend.py -q --basetemp=/private/tmp/task22512-focused-green-3 -k 'short_write_accepts or eagain_write_is_buffered or pending_input_bound or cleanup_continuous_output or process_dead_requires_pty_eof or explicit_raw_cleanup_discard or real_foreground_background_groups or app_crash_master_close'` | PASS: 8 passed, 13 deselected, and 1 warning in 2.82 seconds. |
| App-crash import-isolation RED | `env -u PYTHONPATH ../../.venv/bin/python -B -m pytest Tests/Terminal/test_posix_backend.py::test_app_crash_master_close_hangs_up_ordinary_child_but_not_detached_limit -q --basetemp=/private/tmp/task22512-app-crash-isolation-red` | RED: 1 failed and 1 warning in 11.38 seconds. The path-launched clean child raised `ModuleNotFoundError: No module named 'tldw_chatbook'`. |
| App-crash import-isolation GREEN | `env -u PYTHONPATH ../../.venv/bin/python -B -m pytest Tests/Terminal/test_posix_backend.py::test_app_crash_master_close_hangs_up_ordinary_child_but_not_detached_limit -q --basetemp=/private/tmp/task22512-app-crash-isolation-green` | PASS: 1 passed and 1 warning in 1.86 seconds. |
| Parser-failure manager/backend integration RED | `env -u PYTHONPATH ../../.venv/bin/python -B -m pytest Tests/Terminal/test_posix_backend.py::test_manager_parser_failure_raw_drains_large_real_pty_under_original_attempt -q --basetemp=/private/tmp/task22512-parser-integration-red` | RED: 1 failed and 1 `RequestsDependencyWarning` in 2.04 seconds. The manager had no explicit parser-failure cleanup seam sharing the original `CleanupAttempt`/T0 with the POSIX backend. |
| Parser-failure manager seam GREEN | `env -u PYTHONPATH ../../.venv/bin/python -B -m pytest Tests/Terminal/test_session_manager.py::test_parser_failure_prefers_explicit_cleanup_with_original_attempt Tests/Terminal/test_session_manager.py::test_parser_failure_disables_input_and_raw_drains_only_after_death Tests/Terminal/test_session_manager.py::test_parser_failure_never_raw_drains_without_process_death_proof -q --basetemp=/private/tmp/task22512-parser-manager-green` | PASS: 3 passed and 1 `RequestsDependencyWarning` in 1.52 seconds. |
| Saturated parser-failure integration RED | `env -u PYTHONPATH ../../.venv/bin/python -B -m pytest Tests/Terminal/test_posix_backend.py::test_manager_parser_failure_closes_direct_flood_under_original_attempt -q --basetemp=/private/tmp/task22512-saturated-parser-red` | RED: 1 failed and 1 `RequestsDependencyWarning` in 1.98 seconds because the manager lacked the bounded output-actor accounting seam needed to prove saturation before first-feed failure. |
| Saturated parser-failure integration GREEN | `env -u PYTHONPATH ../../.venv/bin/python -B -m pytest Tests/Terminal/test_posix_backend.py::test_manager_parser_failure_closes_direct_flood_under_original_attempt -q --basetemp=/private/tmp/task22512-saturated-parser-green-1` | PASS: 1 passed and 1 `RequestsDependencyWarning` in 2.52 seconds. The real launcher, controlling PTY, saturated actor, original T0, process-only proof, bounded raw drain, EOF, and sole reaper were exercised. |
| Overall absolute-deadline guards RED | `env -u PYTHONPATH ../../.venv/bin/python -B -m pytest Tests/Terminal/test_posix_backend.py::test_cleanup_expired_at_entry_does_not_inspect_or_signal Tests/Terminal/test_posix_backend.py::test_cleanup_scan_crossing_deadline_cannot_revalidate_or_signal Tests/Terminal/test_posix_backend.py::test_wait_stage_expired_at_entry_does_not_scan -q` | RED: 3 failed and 1 `RequestsDependencyWarning` in 1.45 seconds. Expired entry and a scan crossing the final deadline still allowed forbidden work; pytest also emitted unrelated stale-temporary-directory removal warnings after the failure summary. |
| Overall absolute-deadline guards GREEN | `env -u PYTHONPATH ../../.venv/bin/python -B -m pytest Tests/Terminal/test_posix_backend.py::test_cleanup_expired_at_entry_does_not_inspect_or_signal Tests/Terminal/test_posix_backend.py::test_cleanup_scan_crossing_deadline_cannot_revalidate_or_signal Tests/Terminal/test_posix_backend.py::test_wait_stage_expired_at_entry_does_not_scan -q --basetemp=/private/tmp/task22512-deadline-green-2` | PASS: 3 passed and 1 `RequestsDependencyWarning` in 1.43 seconds, with zero expired-entry scans/revalidations/signals/reads and zero expired-stage scans. |
| Per-stage no-later-than boundaries RED | `env -u PYTHONPATH ../../.venv/bin/python -B -m pytest Tests/Terminal/test_posix_backend.py::test_cleanup_skips_signal_stages_expired_at_entry Tests/Terminal/test_posix_backend.py::test_cleanup_scan_crossing_stage_boundary_skips_that_signal -q --basetemp=/private/tmp/task22512-stage-boundaries-red` | RED: 6 failed and 1 `RequestsDependencyWarning` in 1.74 seconds because late entry and boundary-crossing scans could still emit an expired stage's signal. |
| Per-stage no-later-than boundaries GREEN | `env -u PYTHONPATH ../../.venv/bin/python -B -m pytest Tests/Terminal/test_posix_backend.py::test_cleanup_skips_signal_stages_expired_at_entry Tests/Terminal/test_posix_backend.py::test_cleanup_scan_crossing_stage_boundary_skips_that_signal -q --basetemp=/private/tmp/task22512-stage-boundaries-green-1` | PASS: 6 passed and 1 `RequestsDependencyWarning` in 1.41 seconds. |
| Real scanner-seam denial RED | `env -u PYTHONPATH ../../.venv/bin/python -B -m pytest Tests/Terminal/test_posix_backend.py::test_unrelated_darwin_enumeration_denial_does_not_poison_death_proof -q --basetemp=/private/tmp/task22512-denial-red` | RED: 1 failed and 1 `RequestsDependencyWarning` in 1.69 seconds because the old `psutil.Process.children` patch was vacuous and its denial branch was never reached. |
| Real scanner-seam denial GREEN | `env -u PYTHONPATH ../../.venv/bin/python -B -m pytest Tests/Terminal/test_posix_backend.py::test_unrelated_darwin_enumeration_denial_does_not_poison_death_proof -q --basetemp=/private/tmp/task22512-denial-green` | PASS: 1 passed and 1 `RequestsDependencyWarning` in 1.62 seconds after patching the scanner's real `psutil.pids`/`os.getsid` seam, proving incomplete group enumeration forbids broad signalling without poisoning exact death proof. |
| Stale tracked-PID reuse RED | `env -u PYTHONPATH ../../.venv/bin/python -B -m pytest Tests/Terminal/test_posix_backend.py::test_gone_tracked_descendant_pid_reuse_does_not_poison_cleanup -q --basetemp=/private/tmp/task22512-stale-pid-red` | RED: 1 failed and 1 warning in 1.42 seconds with `CleanupProof(process_dead=False, stream_closed=True, output_complete=True)`. |
| Stale tracked-PID reuse GREEN | `env -u PYTHONPATH ../../.venv/bin/python -B -m pytest Tests/Terminal/test_posix_backend.py::test_gone_tracked_descendant_pid_reuse_does_not_poison_cleanup -q --basetemp=/private/tmp/task22512-stale-pid-green-1` | PASS: 1 passed and 1 warning in 1.18 seconds. Ten subsequent serial repetitions of the affected live PTY test also passed, with an empty exact fixture census after every repetition. |
| Native POSIX backend repetition 1 | `env -u PYTHONPATH ../../.venv/bin/python -B -m pytest Tests/Terminal/test_posix_backend.py -q --basetemp=/private/tmp/task22512-posix-repeat-1` | PASS: 32 passed and 1 `RequestsDependencyWarning` in 12.62 seconds. |
| Native POSIX backend repetition 2 | `env -u PYTHONPATH ../../.venv/bin/python -B -m pytest Tests/Terminal/test_posix_backend.py -q --basetemp=/private/tmp/task22512-posix-repeat-2` | PASS: 32 passed and 1 `RequestsDependencyWarning` in 12.94 seconds. |
| Native POSIX backend repetition 3 | `env -u PYTHONPATH ../../.venv/bin/python -B -m pytest Tests/Terminal/test_posix_backend.py -q --basetemp=/private/tmp/task22512-posix-repeat-3` | PASS: 32 passed and 1 `RequestsDependencyWarning` in 12.98 seconds. |
| Native combined Task 8 suite | `env -u PYTHONPATH ../../.venv/bin/python -B -m pytest Tests/Terminal/test_posix_backend.py Tests/Terminal/test_launch.py Tests/Terminal/test_session_manager.py -q --basetemp=/private/tmp/task22512-host-combined` | PASS: 142 passed, 1 expected native-Windows skip, and 1 `RequestsDependencyWarning` in 13.58 seconds. The warning came from the installed macOS Requests dependency stack; it was not suppressed or counted as a test failure. |
| Prompt-prefixed JSON parser RED | `env -u PYTHONPATH ../../.venv/bin/python -B -m pytest Tests/Terminal/test_posix_backend.py::test_json_line_accepts_prompt_prefixed_payload -q --basetemp=/private/tmp/task22512-json-prefix-red` | RED: 1 failed and 1 `RequestsDependencyWarning` in 1.53 seconds because the test helper required JSON to start at byte zero. |
| Prompt-prefixed JSON parser GREEN | `env -u PYTHONPATH ../../.venv/bin/python -B -m pytest Tests/Terminal/test_posix_backend.py::test_json_line_accepts_prompt_prefixed_payload Tests/Terminal/test_posix_backend.py::test_interactive_pty_retains_state_and_round_trips_unicode -q --basetemp=/private/tmp/task22512-json-prefix-green-1` | PASS: 2 passed and 1 `RequestsDependencyWarning` in 1.75 seconds. This helper-only regression was added after the three 32-test native repetitions. |
| Sealed-commit native combined Task 8 suite | `env -u PYTHONPATH ../../.venv/bin/python -B -m pytest Tests/Terminal/test_launch.py Tests/Terminal/test_session_manager.py Tests/Terminal/test_posix_backend.py --basetemp=/private/tmp/task22512-final-commit-macos -q` | PASS: 143 passed, 1 expected native-Windows skip, and 1 `RequestsDependencyWarning` in 14.02 seconds. This is the post-prompt-regression final commit state. |
| Valid Linux POSIX backend qualification | `/qualification/venv/bin/python -B -m pytest Tests/Terminal/test_posix_backend.py -q` from `/worktree`, as uid/gid 1001 under `/usr/bin/tini -s --` | PASS: 33 passed in 10.21 seconds. |
| Valid Linux combined Task 8 qualification | `/qualification/venv/bin/python -B -m pytest Tests/Terminal/test_posix_backend.py Tests/Terminal/test_launch.py Tests/Terminal/test_session_manager.py -q` from `/worktree`, as uid/gid 1001 under `/usr/bin/tini -s --` | PASS: 142 passed and 2 skipped in 11.16 seconds. The skips were the native Windows profile APIs and unavailable Zsh in the Ubuntu image. |
| Valid Linux post-run fixture census | `docker exec task22512-posix-linux-20260831 pgrep -af 'descendant_holds_tty\.py\|job_control_tree\.py\|terminal_child\.py\|posix_app_crash_probe\.py\|tldw_chatbook\.Terminal\.posix_launcher'` | PASS: exit 1 with no output after the focused and combined runs, meaning no matching process remained. |
| Independent process-death field RED | `env -u PYTHONPATH ../../.venv/bin/python -B -m pytest Tests/Terminal/test_posix_backend.py::test_cleanup_continuous_output_is_turn_bounded_and_preserved Tests/Terminal/test_posix_backend.py::test_process_dead_is_independent_of_pty_eof_after_reap_and_two_zero_scans Tests/Terminal/test_session_manager.py::test_process_only_proof_without_eof_retains_cleanup_unproven -q --basetemp=/private/tmp/task22512-independent-proof-red` | RED: 2 failed, 1 passed, and 1 `RequestsDependencyWarning` in 1.52 seconds. Both backend regressions showed `process_dead=False` after exact process-only proof when PTY EOF was absent. |
| Independent process-death field GREEN | `env -u PYTHONPATH ../../.venv/bin/python -B -m pytest Tests/Terminal/test_posix_backend.py::test_cleanup_continuous_output_is_turn_bounded_and_preserved Tests/Terminal/test_posix_backend.py::test_process_dead_is_independent_of_pty_eof_after_reap_and_two_zero_scans Tests/Terminal/test_session_manager.py::test_process_only_proof_without_eof_retains_cleanup_unproven -q --basetemp=/private/tmp/task22512-independent-proof-green` | PASS: 3 passed and 1 `RequestsDependencyWarning` in 1.59 seconds. The process field is now independent, while the manager regression still retains the session until both process death and stream closure are proven. |
| Parser-failure buffered-output ordering RED | `env -u PYTHONPATH ../../.venv/bin/python -B -m pytest Tests/Terminal/test_posix_backend.py::test_parser_failure_retains_buffer_without_process_only_proof Tests/Terminal/test_posix_backend.py::test_parser_failure_discards_buffer_only_after_process_only_proof -q --basetemp=/private/tmp/task22512-parser-buffer-red` | RED: 2 failed and 1 `RequestsDependencyWarning` in 1.51 seconds because parser-failure cleanup cleared preserved bytes before process-only proof. |
| Parser-failure buffered-output ordering GREEN | `env -u PYTHONPATH ../../.venv/bin/python -B -m pytest Tests/Terminal/test_posix_backend.py::test_parser_failure_retains_buffer_without_process_only_proof Tests/Terminal/test_posix_backend.py::test_parser_failure_discards_buffer_only_after_process_only_proof -q --basetemp=/private/tmp/task22512-parser-buffer-green` | PASS: 2 passed and 1 `RequestsDependencyWarning` in 1.47 seconds. The buffer remains untouched without proof and is discarded only after exact process-only proof under the original attempt/T0; `output_complete` remains false. |
| Healthy cleanup-output handoff RED | `env -u PYTHONPATH ../../.venv/bin/python -B -m pytest Tests/Terminal/test_posix_backend.py::test_manager_hands_cleanup_tail_to_screen_before_output_is_complete -q --basetemp=/private/tmp/task22512-cleanup-handoff-red-5` | RED: 1 failed and 1 `RequestsDependencyWarning` in 2.39 seconds: cleanup reported output complete while the retained screen was blank because backend-preserved tail bytes had no manager-owned handoff. |
| Healthy cleanup-output handoff GREEN | `env -u PYTHONPATH ../../.venv/bin/python -B -m pytest Tests/Terminal/test_posix_backend.py::test_manager_hands_cleanup_tail_to_screen_before_output_is_complete -q --basetemp=/private/tmp/task22512-cleanup-handoff-green-12` | PASS: 1 passed and 1 `RequestsDependencyWarning` in 2.59 seconds. A real launcher/PTY/session-manager run transferred the bounded cleanup tail to the retained actor/parser before `output_complete=True`, with exact shell/holder cleanup and one shell reap. |
| Master-FD reuse serialization RED | `env -u PYTHONPATH ../../.venv/bin/python -B -m pytest Tests/Terminal/test_posix_backend.py::test_close_proven_serializes_master_syscalls_before_fd_reuse -q --basetemp=/private/tmp/task22512-fd-reuse-red-2` | RED: 3 failed and 1 `RequestsDependencyWarning` in 1.66 seconds; read, write, and resize each allowed proven close/reuse to reach the same numeric descriptor while its syscall was blocked. |
| Master-FD reuse serialization GREEN | `env -u PYTHONPATH ../../.venv/bin/python -B -m pytest Tests/Terminal/test_posix_backend.py::test_close_proven_serializes_master_syscalls_before_fd_reuse -q --basetemp=/private/tmp/task22512-fd-reuse-green` | PASS: 3 passed and 1 `RequestsDependencyWarning` in 2.19 seconds. Barrier assertions showed the original descriptor identity before and after each syscall and no close/reuse until the syscall released. |
| Post-review native POSIX file | `env -u PYTHONPATH ../../.venv/bin/python -B -m pytest Tests/Terminal/test_posix_backend.py -q --basetemp=/private/tmp/task22512-spec-review-posix-1` | PASS: 39 passed and 1 `RequestsDependencyWarning` in 14.79 seconds. |
| Post-review bounded combined Task 8 suite | `env -u PYTHONPATH ../../.venv/bin/python -B -m pytest Tests/Terminal/test_posix_backend.py Tests/Terminal/test_launch.py Tests/Terminal/test_session_manager.py -q --timeout=15 --basetemp=/private/tmp/task22512-spec-review-combined-2` | PASS: 149 passed, 1 expected native-Windows skip, and 1 `RequestsDependencyWarning` in 15.74 seconds. |
| Post-review exact native fixture census | `pgrep -af 'descendant_holds_tty\.py\|job_control_tree\.py\|terminal_child\.py\|posix_app_crash_probe\.py\|tldw_chatbook\.Terminal\.posix_launcher'` | PASS: exit 1 with no output after the bounded combined suite and final static checks; no matching Task 8 fixture or launcher process remained. |
| Cleanup-tail release-gate RED | `env -u PYTHONPATH ../../.venv/bin/python -B -m pytest Tests/Terminal/test_posix_backend.py::test_manager_hands_cleanup_tail_to_screen_before_output_is_complete -q --timeout=20 --basetemp=/private/tmp/task22512-cleanup-tail-gate-red` | TEST/FIXTURE RED: 1 failed and 1 `RequestsDependencyWarning` in 5.01 seconds. The new deterministic invocation failed before a holder PID appeared because the fixture did not yet accept the release-gate argument; this was not a product failure. |
| Cleanup-tail release-gate GREEN | `env -u PYTHONPATH ../../.venv/bin/python -B -m pytest Tests/Terminal/test_posix_backend.py::test_manager_hands_cleanup_tail_to_screen_before_output_is_complete -q --timeout=20 --basetemp=/private/tmp/task22512-cleanup-tail-gate-green-1` | PASS: 1 passed and 1 `RequestsDependencyWarning` in 2.69 seconds. The test proved the exact descendant still held the slave after backend cleanup started, then released the marker for preserved-buffer manager handoff before `output_complete=True`. |
| Cleanup-tail neighboring regressions | `env -u PYTHONPATH ../../.venv/bin/python -B -m pytest Tests/Terminal/test_posix_backend.py::test_cleanup_continuous_output_is_turn_bounded_and_preserved Tests/Terminal/test_posix_backend.py::test_process_dead_is_independent_of_pty_eof_after_reap_and_two_zero_scans Tests/Terminal/test_posix_backend.py::test_parser_failure_retains_buffer_without_process_only_proof Tests/Terminal/test_posix_backend.py::test_parser_failure_discards_buffer_only_after_process_only_proof Tests/Terminal/test_posix_backend.py::test_manager_parser_failure_closes_direct_flood_under_original_attempt Tests/Terminal/test_posix_backend.py::test_manager_hands_cleanup_tail_to_screen_before_output_is_complete Tests/Terminal/test_posix_backend.py::test_shell_exit_with_descendant_holding_slave_is_not_mistaken_for_eof -q --timeout=20 --basetemp=/private/tmp/task22512-cleanup-tail-nearby-green-1` | PASS: 7 passed and 1 `RequestsDependencyWarning` in 4.82 seconds. |
| Initial final-Linux portability attempt | `/opt/task22512-venv/bin/python -B -m pytest Tests/Terminal/test_posix_backend.py -q` from `/worktree` as uid/gid 1001 | TEST-HARNESS PORTABILITY FAILURE: 38 passed and 1 failed in 11.90 seconds. Linux delivered the cleanup-tail marker before cleanup and the test's exit synchronization consumed it, leaving the handoff observer empty; no fixture process remained. This was not a product RED. |
| Corrected final-Linux cleanup-tail focus | `/opt/task22512-venv/bin/python -B -m pytest Tests/Terminal/test_posix_backend.py::test_manager_hands_cleanup_tail_to_screen_before_output_is_complete -q` from `/worktree` as uid/gid 1001 | PASS: 1 passed in 2.48 seconds. |
| Corrected final-Linux POSIX file | `/opt/task22512-venv/bin/python -B -m pytest Tests/Terminal/test_posix_backend.py -q` from `/worktree` as uid/gid 1001 | PASS: 39 passed in 11.80 seconds. |
| Corrected final-Linux combined Task 8 suite | `/opt/task22512-venv/bin/python -B -m pytest Tests/Terminal/test_posix_backend.py Tests/Terminal/test_launch.py Tests/Terminal/test_session_manager.py -q` from `/worktree` as uid/gid 1001 | PASS: 148 passed and 2 skipped in 12.88 seconds. The skips were the native Windows profile APIs and unavailable Zsh. |
| Corrected final-Linux exact fixture census | `docker exec task22512-posix-linux-final-20260831 pgrep -af 'descendant_holds_tty\.py\|job_control_tree\.py\|terminal_child\.py\|posix_app_crash_probe\.py\|tldw_chatbook\.Terminal\.posix_launcher'` | PASS: exit 1 with no output; no matching Task 8 fixture or launcher process remained. |
| Final corrected native combined Task 8 suite | `env -u PYTHONPATH ../../.venv/bin/python -B -m pytest Tests/Terminal/test_launch.py Tests/Terminal/test_session_manager.py Tests/Terminal/test_posix_backend.py --basetemp=/private/tmp/task22512-final-portable-macos -q` | PASS: 149 passed, 1 expected native-Windows skip, and 1 `RequestsDependencyWarning` in 15.73 seconds. The warning came from the installed macOS Requests dependency stack and was not suppressed. |
| Final owned-Python-path lint | `../../.venv/bin/python -m ruff check tldw_chatbook/Terminal/posix_backend.py tldw_chatbook/Terminal/posix_launcher.py tldw_chatbook/Terminal/session_manager.py Tests/Terminal/test_posix_backend.py Tests/Terminal/test_session_manager.py Tests/fixtures/terminal` | PASS: `All checks passed!` |
| Final owned-Python-path format check | `../../.venv/bin/python -m ruff format --check tldw_chatbook/Terminal/posix_backend.py tldw_chatbook/Terminal/posix_launcher.py tldw_chatbook/Terminal/session_manager.py Tests/Terminal/test_posix_backend.py Tests/Terminal/test_session_manager.py Tests/fixtures/terminal` | PASS: 9 files already formatted. |
| Final patch whitespace check | `git diff --check` | PASS: exit 0 with no output. |
| Final native fixture-process census | `ps -ef \| rg '/task-22512-posix-backend/Tests/fixtures/terminal/([t]erminal_child\|[d]escendant_holds_tty\|[j]ob_control_tree\|[p]osix_app_crash_probe)\.py\|[T]ests\.fixtures\.terminal\.posix_app_crash_probe\|[t]ldw_chatbook\.Terminal\.posix_launcher'` | PASS: exit 1 with no output; no matching Task 8 launcher or fixture process remained. |

Earlier disposable-container attempts failed before product execution because
core dependencies were absent, then exposed a root-account prompt assumption
and the non-reaping bare PID 1 harness. They are qualification-method caveats,
not product RED evidence. The valid Linux rows above use the corrected
prompt-tolerant test helper, ordinary-user execution, and the explicit
subreaper ancestor.

The real app-crash subprocess was launched as a module from the explicit
repository root with inherited `PYTHONPATH` removed. It recorded the admitted
shell PID plus birth time before crashing and demonstrated that ordinary
PTY-master close removed that exact shell identity and its same-session child.
It also deliberately demonstrated the accepted limitation: a process that
creates a new session can survive ordinary master close. The test revalidated
and terminated that exact detached PID by PID plus birth time; this backend
does not claim containment of deliberately detached host-authority processes.

## Limitations and fail-closed boundaries

- One supported native Windows row ran. It is genuine host evidence, not a
  mock, Wine result, or upstream claim. Its two mandatory stream-semantics
  failures reject pywinpty 3.0.5 for the ADR-099 product boundary despite the
  other native rows passing.
- Windows support remains blocked until a new dependency/API boundary is
  independently qualified and ADR-099 is superseded or amended. The probe must
  not convert missing EOF, incomplete output, or alternate-buffer leakage into
  success.
- Remaining unexecuted native rows are `win-arm64-py311`,
  `win-amd64-py312`, `win-arm64-py312`, `win-amd64-py313`,
  `win-arm64-py313`, `win-amd64-py314`, and `win-arm64-py314`.
- The exact native interpreter path must be resolved and recorded on each host.
  The CPython 3.11 x64 plan command resolves it with
  `$Task22512Python = (& py -3.11-64 -c "import sys; print(sys.executable)")`.
  Repeat with the host's verified matching interpreter and row ID; an ARM64
  launcher selector is deliberately not guessed here.
- After assigning concrete `$Task22512RowId` and `$Task22512Python` values on a
  native host, run this exact sequence:

```powershell
$Task22512Row = Join-Path $env:TEMP ("tldw-task-22512-" + $Task22512RowId + "-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $Task22512Row | Out-Null
& $Task22512Python scripts/terminal_qualification/common.py prepare-row --row-id $Task22512RowId --row-dir $Task22512Row --requirement pywinpty==3.0.5 --requirement pyte==0.8.2 --requirement "wcwidth>=0.2.14,<1" --json-out "$Task22512Row\artifacts.json"
& "$Task22512Row\venv\Scripts\python.exe" -B scripts/terminal_qualification/environment_probe.py --shell default --json-out "$Task22512Row\environment-default.json"
& "$Task22512Row\venv\Scripts\python.exe" -B scripts/terminal_qualification/environment_probe.py --shell powershell --json-out "$Task22512Row\environment-powershell.json"
& "$Task22512Row\venv\Scripts\python.exe" -B scripts/terminal_qualification/environment_probe.py --shell cmd --json-out "$Task22512Row\environment-cmd.json"
& "$Task22512Row\venv\Scripts\python.exe" -B scripts/terminal_qualification/pyte_probe.py --artifact-manifest "$Task22512Row\artifacts.json" --json-out "$Task22512Row\pyte.json"
& "$Task22512Row\venv\Scripts\python.exe" -B scripts/terminal_qualification/pywinpty_probe.py --artifact-manifest "$Task22512Row\artifacts.json" --json-out "$Task22512Row\pywinpty.json"
& "$Task22512Row\venv\Scripts\python.exe" -B scripts/terminal_qualification/common.py collect-row --row-dir $Task22512Row --evidence-root Docs/superpowers/reviews/evidence/task-22512/raw
```

This is exactly one six-file Windows generation: artifacts, default,
PowerShell, CMD, pyte, and pywinpty. `default` uses the code-owned selection
order `pwsh.exe`, `powershell.exe`, then validated `COMSPEC`; it is a genuine
probe result and is never an alias for either named-shell file. Each environment
probe uses the disposable-account, admitted-bootstrap, and capped-pipe runner
described above. Output overflow is an explicit failure fact, terminates the
owned Job immediately, and retains at most the configured ceiling plus one
fixed 8-KiB read chunk. Collection rejects any missing, extra, duplicate, or
mixed-generation sibling before publication.

- The formatter baseline was regenerated after rebasing the PR. The exact
  `verify --head HEAD` command accepted its immutable base hashes, normalized
  formatter-diff hashes, debt facts, red-path set, and recorded Ruff version.
  Its base is `3e5e75e4aa884d4f362aa63c1e151c3855f07a36`; verification does not modify
  source.
- This artifact adds no product code, workflow, or adapted third-party terminal
  source beyond the reviewed `regex==2026.4.4` validation dependency pin.
