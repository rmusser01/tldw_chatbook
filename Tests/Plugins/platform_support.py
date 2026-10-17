"""Native capability gates and collected evidence; skips never qualify support."""

import ctypes
import json
import os
import sys
from pathlib import Path

import pytest

MANAGED_RUNTIME = sys.platform == "darwin" and ctypes.sizeof(ctypes.c_void_p) == 8
PACKAGE_CAPTURE = (
    os.name == "posix" and hasattr(os, "O_NOFOLLOW") and os.open in os.supports_dir_fd
)
VENDOR_COMMAND = sys.platform == "darwin"
V2_COMMAND = sys.platform != "win32"

managed_runtime = pytest.mark.skipif(
    not MANAGED_RUNTIME,
    reason="managed runtime ownership is qualified only on 64-bit Darwin",
)
package_capture = pytest.mark.skipif(
    not PACKAGE_CAPTURE,
    reason="package capture requires POSIX descriptor-relative no-follow operations",
)
vendor_command = pytest.mark.skipif(
    not VENDOR_COMMAND,
    reason="vendor command normalization is qualified only on Darwin",
)
v2_command = pytest.mark.skipif(
    not V2_COMMAND, reason="standalone v2 command execution refuses Windows"
)


def require_managed_runtime():
    if not MANAGED_RUNTIME:
        pytest.skip(managed_runtime.kwargs["reason"])


def pytest_collection_finish(session):
    """Optional workflow artifact from the actual collected items and marks."""
    target = os.environ.get("PLUGIN_COLLECTION_ARTIFACT")
    if not target:
        return
    nodes = []
    for item in session.items:
        reasons = []
        declarations = []
        for mark in item.iter_markers("skipif"):
            condition = mark.args[0] if mark.args else mark.kwargs.get("condition")
            declarations.append(
                {
                    "condition": (
                        condition
                        if isinstance(condition, (str, bool))
                        else repr(condition)
                    ),
                    "reason": mark.kwargs.get("reason", "skipif"),
                }
            )
            if condition is True:
                reasons.append(mark.kwargs.get("reason", "skipif"))
        reasons.extend(
            mark.kwargs.get("reason", "skip") for mark in item.iter_markers("skip")
        )
        owned_fixtures = {
            "plugin_stack",
            "native_console",
            "source_stack",
            "revocation_case",
            "shared_connection_case",
            "variable_case",
        }.intersection(item.fixturenames)
        if owned_fixtures and not MANAGED_RUNTIME:
            reasons.append(managed_runtime.kwargs["reason"])
        nodes.append(
            {
                "nodeid": item.nodeid,
                "fixtures": item.fixturenames,
                "capability_skip_reasons": sorted(set(reasons)),
                "skipif_declarations": declarations,
                "managed_runtime_fixtures": sorted(owned_fixtures),
            }
        )
    Path(target).write_text(
        json.dumps(
            {
                "platform": sys.platform,
                "native_capabilities": {
                    "managed_runtime": MANAGED_RUNTIME,
                    "package_capture": PACKAGE_CAPTURE,
                    "vendor_command": VENDOR_COMMAND,
                    "v2_command": V2_COMMAND,
                },
                "qualification": "Collection and skip declarations only; executed outcomes are in JUnit.",
                "nodes": nodes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
