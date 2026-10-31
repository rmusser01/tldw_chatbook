"""Run the patched std-only Rust regressions without a second native feature graph."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def check_helpers(source: Path, rustc: Path, output: Path) -> None:
    """Compile the exact helper module and pure discovery method/test from source."""
    output.mkdir(parents=True, exist_ok=True)
    discovery = (source / "wezterm-client/src/discovery.rs").read_text(encoding="utf-8")
    method_start = discovery.index("        fn compute_names_for(")
    method_end = discovery.index("        /// Publish path as the path", method_start)
    tests_start = discovery.index("    #[cfg(test)]\n    mod tests {")
    tests_end = discovery.index("    /// Convert a rust string", tests_start)
    namespace = output / "namespace-tests.rs"
    namespace.write_text(
        "use std::path::Path;\nstruct NameHolder;\nimpl NameHolder {\n"
        + discovery[method_start:method_end]
        + "}\n"
        + discovery[tests_start:tests_end],
        encoding="utf-8",
    )
    for name, path in (
        ("portable", source / "config/src/portable.rs"),
        ("namespace", namespace),
    ):
        binary = output / f"{name}-tests.exe"
        command = [
            str(rustc),
            "--edition=2018",
            "--test",
            "-C",
            "opt-level=3",
            "-C",
            "target-feature=+crt-static",
            str(path),
            "-o",
            str(binary),
        ]
        subprocess.run(command, check=True)
        subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--rustc", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    check_helpers(arguments.source, arguments.rustc, arguments.output)
