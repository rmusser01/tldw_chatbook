# Modified WezTerm for the portable Windows prototype

This build implements [ADR-255](../../../../backlog/decisions/255-bundled-wezterm-portable-runtime-roots.md)
on official WezTerm `20240203-110809-5046fc22`. It is a modified, unsigned prototype.

Run from the repository root with Python 3.12, Git, CMake and installed Visual
Studio C++ build tools/Windows SDK:

```powershell
python Packaging/windows/portable/wezterm/build.py --work output/wezterm-build --jobs 4
```

The driver verifies the official source, Rust 1.75 components and portable
Strawberry Perl 5.38.4.1 against `inputs.json`; extracts them into ignored build
storage; applies `portable-roots.patch`; and builds the GUI, CLI and mux server
against the upstream `Cargo.lock`. Visual Studio environment changes affect only
the build subprocess. There is no global tool installation or PATH modification.
The existing compiler and SDK versions are recorded in `provenance.json`.
MSVC's vendored OpenSSL build has a native generated-path limit. The driver uses
an already-existing Windows short-name spelling of the build directory, with no
new alias or OS setting, and requires that spelling to fit within 80 UTF-8 bytes.
If short names are unavailable, supply a short task-owned absolute `--work` path
(for example a fresh directory under the native temporary directory). The build
directory and its spelling are recorded in provenance; end-user bundle paths use
the separately tested socket-address handling below.

The three Windows process-local selectors are:

- `CHATBOOK_WEZTERM_HOME`: config/home root.
- `CHATBOOK_WEZTERM_DATA_DIR`: persistent terminal state root.
- `CHATBOOK_WEZTERM_RUNTIME_DIR`: IPC, discovery, logs and PKI root.

A present selector must contain an absolute nonempty Windows path. Invalid values
fail rather than falling back to the host profile. Absent selectors preserve the
upstream behavior. The launcher validates/prepares these paths beneath `Data` and
sets portable `TEMP`/`TMP` for temporary blob storage. The dedicated Lua config
must set `ssh_domains = {}` to prevent automatic host SSH-config discovery.

The pinned Windows AF_UNIX implementation permits at most 107 UTF-8 bytes in a
socket address. Published paths, discovery files and logs retain their absolute
`Data` paths. Only bind/connect addresses use a short path relative to the process
working directory, after proving they remain beneath the explicit runtime root.
The launcher and matching CLI queries must use `Data` as their working directory.
Optional features that spawn the WezTerm CLI must also select that directory;
Chatbook's restored-profile child can have its own config working directory.
Outside-root paths, parent traversal, an unsuitable working directory and an
address over the byte limit fail explicitly. Helpers never change process CWD or
fall back to a host/global temporary directory. Failed address validation does
not make discovery delete a potentially live socket.
The portable GUI propagates private IPC startup errors.

Windows also uses temporary named mutex/shared-memory kernel objects for GUI
discovery. Portable builds scope their names by window class and a stable digest
of the explicit runtime root; the digest isolates collisions and provides no
authentication. These are OS objects, not filesystem state. Resolved portable
socket leaf names become absolute paths under the runtime root before connection.
With the override absent, upstream object names and resolution stay unchanged.

Successful output is `output/wezterm-build/src/target/release/wezterm-gui.exe`,
`wezterm.exe`, and `wezterm-mux-server.exe`. `build.log`, `test.log` and
`provenance.json` preserve the exact commands, focused override-test result,
archive/toolchain/patch hashes and executable hashes. Retain the matching ANGLE
`libEGL.dll`/`libGLESv2.dll`, `mesa/opengl32.dll` and `strip-ansi-escapes.exe`
support files from the pinned upstream Windows ZIP, plus upstream license notices,
including `LICENSE.md` and font OFL files from `assets/fonts`. The driver collects
the source and fetched dependency license declarations/notices in
`output/wezterm-build/third-party-licenses/`, with a hashed source index.

The portable packager omits the optional `conpty.dll`/`OpenConsole.exe` pair.
The pinned sideloaded ConPTY crashes with `0xC0000409` at a long executable
location; the same bytes work at short ASCII and Unicode locations, with the
same long Data root and child. Removing the pair selects upstream's existing
`kernel32.dll` ConPTY fallback, verified at the failing long location. Windows
10 October 2018 or later remains the source's existing minimum for this API.
Build-tree copies retain the upstream companions; they are not delivered in the
portable payload. The paired A/B evidence is in
`output/wezterm-build/packaging-crash/` (matrix, location matrix and
`system-conpty-both-absent-result.json`). No native source change is needed.

The five std-only Rust helper tests compile directly from the patched helper
module and the exact pure discovery method/test extracted from the pinned source.
This keeps the production native feature graph stable. The full Cargo release
build and real GUI/CLI query verify the callers. Cargo's narrower test graph can
rebuild native OpenSSL/libssh even in release mode because its dependency features
differ; no full upstream or Chatbook test sweep is part of this build command.

The source patch's five regressions cover absent values, invalid root selectors,
absolute spaces/Unicode and UNC roots, a deep Unicode root with a short relative
socket address, outside-root/wrong-CWD/traversal refusal and the exact UTF-8 byte
limit, and stable/different discovery keys for the same/different roots.
Qualification also requires a real GUI/CLI pane query from the
portable launcher and evidence that the host WezTerm namespace was unchanged.
Normal system fonts, clipboard and OS activity are outside terminal file/IPC
containment. No clean-machine, removable-filesystem or signing claim is made.
