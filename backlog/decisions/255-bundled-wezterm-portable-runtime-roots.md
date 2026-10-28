# ADR-255: Bundled WezTerm portable runtime roots

Status: Accepted for the Windows portable prototype
Date: 2026-10-28
Related task: TASK-37205
Extends: [ADR-254](254-windows-portable-wezterm-spike.md)

## Decision

Build the same pinned upstream WezTerm source release with a small Windows-only
patch. Three explicit process-local selectors, CHATBOOK_WEZTERM_HOME,
CHATBOOK_WEZTERM_DATA_DIR and CHATBOOK_WEZTERM_RUNTIME_DIR, select its config home,
persistent data and IPC/log runtime roots. A present override must be an absolute
nonempty path; invalid values refuse startup. With no override upstream behavior
is unchanged. The launcher supplies validated directories inside Data. Temporary
blob storage retains the existing portable TEMP/TMP selection.

Build the GUI, CLI and mux-server together against the release's Cargo.lock so
all three agree on discovery paths. Preserve the official source SHA-256, patch,
toolchain versions, license files and resulting executable hashes. Rust and native
build tools belong to ignored build storage; they are not installed globally or
shipped as end-user prerequisites. This is a modified WezTerm build and must be
identified as such in the bundle manifest and documentation.

Disable automatic SSH-domain discovery in the dedicated terminal config. The
Chatbook terminal launches the bundled CLI directly and does not need to inspect
the real Windows user's SSH files. Ordinary system fonts, clipboard and the OS
remain host facilities; this decision concerns terminal-owned files and IPC.

The pinned Windows AF_UNIX implementation limits socket addresses to fewer than
108 UTF-8 bytes. Deep portable paths can exceed this even when every directory
is valid. For Windows socket bind/connect calls only, use a short CWD-relative
address when the absolute socket is proven beneath CHATBOOK_WEZTERM_RUNTIME_DIR
and the process working directory provides a safe relative address. Published
discovery paths and file paths remain absolute. Refuse when no safe short address
can be formed; never fall back to a host directory or change process-wide CWD.
The launcher and matching CLI use Data as their working directory. Exercise both
bind and connect from a deep Unicode extraction, including the exact byte limit.

Windows discovery uses ephemeral shared-memory/window-class names as well as
socket files. Scope those object names by a digest of the explicit runtime root
so portable bundles and ordinary WezTerm do not overwrite each other's discovery
publication. Normalize the published socket leaf beneath that root for portable
clients; no override preserves upstream names and behavior. These are Windows
kernel objects, not host-profile filesystem state.

Fresh restored-profile processes retain the three explicit terminal selectors.
The GUI and matching manual CLI run with Data as CWD. An optional CLI child whose
CWD cannot form a safe short runtime address refuses; it never falls back to a
host runtime directory.

Omit the pinned optional `conpty.dll`/`OpenConsole.exe` pair. A native A/B
qualification found that the same payload fast-fails with exit 0xC0000409 from
a 168-character executable path, while short ASCII and Unicode paths succeed;
removing only the optional DLL makes that long-path launch succeed. Use
WezTerm's existing kernel32 ConPTY fallback, requiring Windows 10 version 1809
or later, rather than depending on end-user 8.3 names or relocating executables
outside the bundle. Retain the other upstream support DLLs and notices.

## Alternatives

- HOME/USERPROFILE or XDG_RUNTIME_DIR alone: the pinned Windows known-folder
  implementation ignores those selectors for runtime discovery.
- Patch/import-hook Windows known-folder APIs: adds broad interception outside
  terminal ownership and changes unrelated consumers.
- Delete host-side files after exit: still writes outside Data and cannot keep
  simultaneous terminal instances independent.
- Replace WezTerm: the user selected it and its current caller paths can share
  these three explicit roots without a new terminal implementation.

## Verification

Exercise override parsing, build the pinned binaries, launch real Chatbook and
query its pane with the matching CLI. Assert IPC discovery and terminal logs
are under Data, and snapshot the existing host WezTerm namespace before/after
to establish that the prototype did not change it. Repeat from a path containing
spaces and Unicode. Do not claim zero OS activity or qualification of all optional
terminal features.
