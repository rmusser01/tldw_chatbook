"""TASK-19323: pin the chunking security logger's write surface.

`scripts/check_persistent_diagnostic_inventory.py` builds the persistent-sink
topology from a fixed set of recognized sink spellings (``SINK_CALL_NAMES``
plus loguru ``add``).  A bare ``open()`` + ``json.dump`` export in
``tldw_chatbook/Chunking/engine/security_logger.py`` was therefore invisible
to the sink census while its event store held raw user XML (``xml_sample``,
captured by ``log_xxe_attempt``).  Teaching the checker to see every bare
``open`` in the repo would flood the topology with non-diagnostic file
writes, so the durable fix is scoped to the one module that owns a
security-event store: pin its write surface to the single declared loguru
sink in ``SecurityLogger.__init__`` and nothing else, pin the retired
export out, and pin the store itself to metadata-only XXE capture.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

from scripts import check_persistent_diagnostic_inventory as diagnostic_inventory


REPO_ROOT = Path(__file__).resolve().parents[2]
SECURITY_LOGGER_PATH = (
    REPO_ROOT / "tldw_chatbook/Chunking/engine/security_logger.py"
)

# Call names (the final attribute segment) that can create or write a file.
# The module's one declared sink is loguru's ``add`` -- deliberately not in
# this set, so no exemption machinery is needed: the assertion below is that
# the module contains ZERO write-capable calls.
_WRITE_CAPABLE_CALL_NAMES = {
    "NamedTemporaryFile",
    "TemporaryFile",
    "copy",
    "copy2",
    "copyfile",
    "copyfileobj",
    "dump",
    "fdopen",
    "mkstemp",
    "move",
    "open",
    "rename",
    "replace",
    "write",
    "write_bytes",
    "write_text",
    "writelines",
}


def _module_source() -> str:
    return SECURITY_LOGGER_PATH.read_text(encoding="utf-8")


def test_security_logger_has_no_file_write_calls() -> None:
    """The module must never write files outside its declared loguru sink.

    A bare-``open()`` export here is exactly the shape the sink topology
    cannot see (TASK-19323); this pin turns red on any reintroduction,
    whatever the caller count.
    """
    source = _module_source()
    tree = ast.parse(source, filename=str(SECURITY_LOGGER_PATH))
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        parts = diagnostic_inventory._attribute_parts(node.func)
        call_name = parts[-1] if parts else ""
        if call_name in _WRITE_CAPABLE_CALL_NAMES:
            segment = ast.get_source_segment(source, node) or call_name
            offenders.append(segment.replace("\n", " ")[:120])
    assert offenders == [], (
        "security_logger.py gained write-capable call(s) invisible to the "
        "persistent-sink topology; declare and review them instead: "
        + "; ".join(offenders)
    )


def test_security_logger_declared_sink_surface_is_exactly_the_init_loguru_sink() -> None:
    """Cross-check against the checker's own census: one sink, in __init__."""
    _diagnostics, sinks = diagnostic_inventory.scan_source(
        _module_source(), filename=str(SECURITY_LOGGER_PATH)
    )
    assert [(entry["kind"], entry["scope"]) for entry in sinks] == [
        ("loguru_sink", "SecurityLogger.__init__")
    ]


def test_export_events_stays_retired() -> None:
    """TASK-19323 removed the zero-caller unredacted event export for good."""
    source = _module_source()
    tree = ast.parse(source, filename=str(SECURITY_LOGGER_PATH))
    exporters = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "export_events"
    ]
    assert exporters == []
    assert "export_events" not in source


def test_xxe_event_store_retains_no_raw_xml() -> None:
    """``log_xxe_attempt`` must store metadata about the XML, never the XML."""
    from tldw_chatbook.Chunking.engine.security_logger import SecurityLogger

    marker = "TASK19323-XXE-MARKER-73912"
    malicious_xml = (
        "<?xml version='1.0'?><!DOCTYPE foo [<!ENTITY "
        f"{marker} SYSTEM 'file:///etc/passwd'>]><foo>&{marker};</foo>"
    )
    security_logger = SecurityLogger(log_file=None, enable_console=False)
    security_logger.log_xxe_attempt(malicious_xml, source="test")

    events = security_logger.get_events()
    assert len(events) == 1
    event = events[0]
    assert event["type"] == "xxe_attempt"
    # Non-vacuous: the metadata the store SHOULD keep is present...
    assert event["details"]["xml_length"] == len(malicious_xml)
    assert event["details"]["source"] == "test"
    # ...and no serialization of the store carries the user XML itself.
    payload = json.dumps(events, default=str)
    assert marker not in payload
    assert "file:///etc/passwd" not in payload
