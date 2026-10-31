# Experimental Chatbook Windows portable bundle

This is a Windows x64 proof of concept, not a signed production release.
Extract the complete ZIP into a writable **local NTFS** directory, then open
`Chatbook.cmd`. Python and WezTerm are included. No PATH changes, installation,
administrator account, or separately installed terminal are needed.
Windows 10 version 1809 or later with system ConPTY is required. The pinned
optional ConPTY DLL/host is omitted because it crashes from long executable
paths; WezTerm uses its existing Windows system implementation instead.

The bundle uses a modified WezTerm with explicit Data roots and isolated Windows
discovery. A moved used profile requires a destination-side review, then imports
into fresh recovery authority. Copied source/control evidence stays unchanged
and inert. Native admission, committed-profile validation and activation reviews
remain enabled. This is local NTFS/same-account qualification, not a signed
production release or a claim of zero OS activity.

## Portable data

The first start creates `Data` beside the launcher. Close Chatbook, then move
the whole folder, including Data. No preparation at the original is required;
it may already be unavailable. The next start shows validated content, storage,
destination and setup requirements. Type `relocate` to import the reviewed sealed
snapshot, or cancel to preserve the copied source. Later external edits remain
in that retained source. Copied pending/ambiguous publications block import.
The launcher never imports the installed Chatbook profile.

- Config: `Data/home/.config/tldw_cli/config.toml`.
- Default databases and logs: `Data/home/.local/share/tldw_cli/<profile>/`.
- Models/caches: beneath the portable home or `Data/cache`.
- Temporary files: `Data/tmp`.
- WezTerm persistent data and socket/log roots: `Data/wezterm/data` and
  `Data/wezterm/runtime`.
- Fresh restored profiles and independent control: `Data/.portable-relocation/attempt-.../`.

Each start validates committed journal/catalog/native activation associations.
A second move uses the same reviewed process. An interrupted local publication
continues its existing operation; no ancestry reset or blind new restore is used.
Earlier generations are retained and never automatically deleted.

Import uses installed owner/schema/dependency checks and private SQLite WAL
normalization. Raw copies cannot prove original cross-store coherence, so their
intermediate archives are explicitly partial. Unknown durable files and unsafe
or unsupported layouts refuse unchanged. Initial unsupported cases include
nonempty Chroma/recovered-media/loose-voice layouts and populated unqualified
embedding caches. External workspace locators stay unresolved pending review;
old targets are not opened. Exact installed built-in artwork may supply validated
ID/hash/reference dependencies; arbitrary custom assets cannot be substituted.

Windows keyring integration and inherited provider/API environment variables are
disabled for this independent profile. Managed credentials are excluded from
the new profile and need setup again. Restored-content activation reviews remain
required before tools, sync, schedules and other gated features reconnect;
copied approvals stay historical. Portable storage is **not encrypted by default**.
Retained source still contains its original credentials: protect all of Data.
No credential is shipped in the ZIP.
User-selected absolute paths, external workspaces and external programs remain
external. This spike does not rewrite those paths when you move it.

Data links/junctions and hard-link aliases are refused. Existing private-path/ACL checks
remain enabled. Do not put this prototype in Program Files, a shared-writable
directory, a network share, or on FAT/exFAT. A different Windows account/machine
may require addressing ownership and re-entering credentials; same-machine
relocation is the qualification target. Do not remove a drive while SQLite or
workers are running. Delete Data only when you intend to discard your profile.

Ctrl+Shift+C/V copy and paste. Chatbook receives its normal keybindings.
Fonts, graphics drivers, clipboard and Windows kernel objects remain host
facilities. Optional WezTerm CLI children must use Data as CWD; an unsafe or
overlong IPC address refuses rather than using a host runtime folder.
Optional ML models, external tools, GPU backends and optional feature extras are
not included. The manifest identifies the exact Python/WezTerm archives; the pip
report lists resolved dependencies and their download hashes. Python 3.12.10 and
WezTerm's 2024 stable release are pinned spike inputs, not a production update
or security-support policy. Ship no installer from this experiment.

## Build from the repository

Use Windows Python 3.12 x64 with `build`, `setuptools>=77`, `wheel`, and `pip`.
The terminal build also needs Visual Studio 2022 C++ tools, a Windows SDK and
CMake. Its driver provisions pinned Rust/Perl locally and uses upstream
Cargo.lock. Native build paths must be short; the driver uses existing 8.3 names
or requests a shorter owned work directory. See [the build guide](wezterm/README.md).
Run from the repository root:

```powershell
python Packaging/windows/portable/wezterm/build.py --work output/wezterm-build --jobs 4
python -m build --wheel --outdir output/portable-wheel
python Packaging/windows/portable/build_portable.py --wheel output/portable-wheel/tldw_chatbook-0.2.3-py3-none-any.whl --wezterm-build output/wezterm-build --output output/Chatbook-Windows-Portable --cache .portable-cache
```

The builder checks `inputs.json` SHA-256 values before extracting official
archives, modified GUI/CLI/mux provenance and preserved license-notice hashes.
It refuses an existing output directory and installs the built wheel
plus its declared base dependencies. It runs no application and creates no Data
directory before writing the ZIP. Runtime licenses/notices stay with their
packages, including CPython, WezTerm's assets/licenses and Chatbook's wheel.
Keep the source checkout corresponding to the shipped Chatbook wheel available
under its AGPL license. Every new ZIP excludes Data even after testing a bundle.
Never redistribute a previously used Data directory. Stock WezTerm is never
silently substituted for the required modified build.

For a console-only probe, from any working directory:

```powershell
& 'C:\path\Chatbook-Windows-Portable\runtime\python.exe' -B 'C:\path\Chatbook-Windows-Portable\portable.py' --diagnose
```

The probe verifies installed package provenance and default data paths, finds
packaged migrations, and creates/reopens a fixed real Chatbook note. Repeat to
verify persistence. A moved used profile requires relocation review before opening.
It is a storage/packaging check, not a claim that all product features passed.

Governance: ADR-283, ADR-255, ADR-256 and TASK-37205. Exact qualification results
are recorded in the repository task notes.
