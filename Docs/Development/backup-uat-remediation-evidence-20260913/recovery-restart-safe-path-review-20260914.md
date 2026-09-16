# Recovery restart safe-path review

APPROVED, no actionable findings. Reviewed the exact three-file hashes in /private/tmp/uat-recovery-restart-safe-path-hashes.json and independently verified each current file.

The only product change adds Python -P before -c in the existing fixed argv. The existing filtered environment already supplies the launching package root as its exact PYTHONPATH; -P prevents inherited cwd from preceding that root. The POSIX exec and Windows CreateProcess branches receive the same argv, while process release, path arguments, environment filtering, and UI entry stay unchanged. Python >=3.12 supports this flag.

The native regression runs the real restart with a harmless competing cwd package, substituting only the eventual UI entry with import-origin output. It therefore detects the actual search-path defect rather than merely checking argument shape. Existing direct/native Windows argv checks remain relevant. Independent focused selection: 4 passed, 11 deselected in 0.90s, /private/tmp/uat-restart-safe-path-independent.log. Author full15 passed67.90s. Native Windows acceptance remains pending; local execution verifies the POSIX branch and Windows argument contract, not Windows process creation.
