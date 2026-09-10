# TASK-32164 sanitizer qualification

All nine unchanged native HTTP shutdown cases passed once on Linux ARM with ASan and UBSan enabled. The instrumented no-argument executable first reached usage and exit 2 in 0.609 seconds. The isolated container completed in 18.872 seconds, including compilation; every case exited 0 without sanitizer diagnostics. No native code repair was needed.

Toolchain: Debian GNU/Linux 12, aarch64, GCC/G++ 12.2.0 (`12.2.0-14+deb12u1`). Immutable image: `sha256:596934820d0c869610526649be433cefcdce0b9b4abdbd8040fc6b2894b83bcd`. It extends the originally supplied image with build-essential, libasan8 and libubsan1. The existing local parent tag was verified against its exact image ID before building, and the resulting image's parent layers were verified unchanged. No macOS system toolchain was modified.

[Results](results.json) retain the full case stdout/stderr, timings, compiler versions, native binary and sanitizer-runtime hashes. [Build commands](build-commands.txt) use `-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer`; runtime options are `ASAN_OPTIONS=detect_leaks=1:halt_on_error=1` and `UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`. Expected transport disconnect messages are preserved and are not sanitizer diagnostics. [The full container log](container-stdout.log) also retains every emitted result. [The toolchain recipe](Dockerfile), [build receipt](toolchain-build-receipt.json), [full build log](toolchain-build.log), [image inspection](image-inspect.json), and [Docker versions](docker-version.json) pin the tooling.

The unique container used `--rm`, no network, two CPUs, 1 GiB memory, read-only root/source mounts and a private evidence directory. Its outer deadline was 240 seconds; each compile had 60 seconds and each case 8 seconds. [The controller receipt](controller.json) confirms that the exact container no longer exists, all 507 recorded source inputs match before/after, and the native patch hash remains `faa692effa31ef5b7836e2b727fda0a9452a19008a8c679686dceda1c1831dac`. Only this task's image/container was created; preexisting PostgreSQL and inference containers were untouched. No model or audio ran.

The original macOS startup blocker remains explicit: Apple clang 17.0.0's ASan runtime on macOS 26.5.2 re-enters initialization through dyld metadata allocation and waits in `StaticSpinMutex::LockSlow` before `main`, including outside the sandbox. [The raw stack](macos-startup-sample.txt) and [original startup receipt](../macos-unsandboxed-startup.json) are retained. These Linux passes qualify the unchanged transport under a working alternative; they do not prove macOS ASan startup is repaired. The earlier [Linux compiler-missing receipt](../linux/results.json) likewise remains historical evidence; later explicit authorization allowed the separate build-tools image.

Reproduction, using the retained task-local controller and a fresh output directory:

```bash
python3 /private/tmp/tts-macos-burndown/native/run_linux_sanitized_transport.py --image sha256:596934820d0c869610526649be433cefcdce0b9b4abdbd8040fc6b2894b83bcd --output /private/tmp/tts-macos-burndown/native/transport-tests/linux-arm-sanitized-recheck
```

Raw binaries and all separate logs remain at [the local run directory](/private/tmp/tts-macos-burndown/native/transport-tests/linux-arm-sanitized-02). The initial Dockerfile attempt is preserved at [build-01](/private/tmp/tts-macos-burndown/native/sanitizer-toolchain/build-01): Docker interpreted a raw `sha256:` FROM value as a registry name and failed before provisioning. The successful build used the verified existing local tag; this did not alter the supplied image or native source.
