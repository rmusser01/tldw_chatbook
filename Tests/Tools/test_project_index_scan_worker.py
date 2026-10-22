"""New index wire protocol is bounded and closed."""

import importlib
import io
import struct

import pytest


def test_oversized_header_is_refused_before_payload_read():
    wire = importlib.import_module("tldw_chatbook.Tools.project_index_scan_protocol")
    with pytest.raises(wire.ScanProtocolError):
        wire.read_frame(io.BytesIO(struct.pack("!I", 2 * 1024**2 + 1)))


def test_unknown_frame_and_extra_fields_are_refused():
    wire = importlib.import_module("tldw_chatbook.Tools.project_index_scan_protocol")
    for value in (
        {"version": 2, "type": "Unknown"},
        {"version": 2, "type": "Close", "body": "extra"},
    ):
        with pytest.raises(wire.ScanProtocolError):
            wire.write_frame(io.BytesIO(), value)


def test_unicode_newline_field_round_trip():
    wire = importlib.import_module("tldw_chatbook.Tools.project_index_scan_protocol")
    value = wire.frame(
        "Entries",
        entries=[
            dict(
                relative_path="\u96ea\nfile.txt",
                kind="file",
                identity="id",
                mtime_ns=1,
                size_bytes=0,
            )
        ],
    )
    assert wire.read_frame(io.BytesIO(wire.encode_frame(value))) == value


def test_real_worker_refuses_oversized_initial_frame_and_exits():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-I", "-m", "tldw_chatbook.Tools.project_index_scan_worker"],
        input=struct.pack("!I", 2 * 1024**2 + 1),
        capture_output=True,
        timeout=10,
    )
    wire = importlib.import_module("tldw_chatbook.Tools.project_index_scan_protocol")
    assert result.returncode == 2
    assert wire.read_frame(io.BytesIO(result.stdout))["type"] == "Error"
    assert not result.stderr


def test_worker_import_closure_has_no_runtime_or_third_party_modules():
    import json
    import subprocess
    import sys

    code = """import sys,json
import tldw_chatbook.Tools.project_index_scan_worker
print(json.dumps(sorted(name for name in sys.modules if name.split('.')[0] in {'textual','pydantic','loguru','torch','numpy','sqlite3'} or name.startswith(('tldw_chatbook.config','tldw_chatbook.DB.','tldw_chatbook.Backup_Recovery.')))))"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    assert json.loads(result.stdout) == []


def test_bool_revision_and_duplicate_json_keys_are_refused():
    wire = importlib.import_module("tldw_chatbook.Tools.project_index_scan_protocol")
    with pytest.raises(wire.ScanProtocolError):
        wire.encode_frame(wire.frame("Started", revision=True, ack=0))
    raw = b'{"version":2,"type":"Close","type":"Close"}'
    with pytest.raises(wire.ScanProtocolError):
        wire.read_frame(io.BytesIO(struct.pack("!I", len(raw)) + raw))


def test_frame_budget_includes_length_header():
    wire = importlib.import_module("tldw_chatbook.Tools.project_index_scan_protocol")
    raw = b'{"version":2,"type":"Close"}'
    maximum_payload = 2 * 1024**2 - 4
    valid = raw + b" " * (maximum_payload - len(raw))
    assert wire.read_frame(
        io.BytesIO(struct.pack("!I", len(valid)) + valid)
    ) == wire.frame("Close")
    oversized = valid + b" "
    with pytest.raises(wire.ScanProtocolError):
        wire.read_frame(io.BytesIO(struct.pack("!I", len(oversized)) + oversized))


@pytest.mark.parametrize(
    "fields",
    [
        {"directories": [""] * 257},
        {"directories": ["", ""]},
        {"directories": ["../outside"]},
        {"continuation": 1},
        {"ack": True},
    ],
)
def test_unit_seeds_are_closed_bounded_and_canonical(fields):
    wire = importlib.import_module("tldw_chatbook.Tools.project_index_scan_protocol")
    request = wire.frame(
        "Unit", directories=[""], continuation=False, revision=1, exclusions=[], ack=0
    )
    request.update(fields)
    with pytest.raises(wire.ScanProtocolError):
        wire.encode_frame(request)


@pytest.mark.parametrize("field", ["inspections", "attempts", "entries"])
def test_unit_done_rejects_each_aggregate_overflow(field):
    wire = importlib.import_module("tldw_chatbook.Tools.project_index_scan_protocol")
    done = wire.frame("UnitDone", inspections=1, attempts=1, entries=1)
    done[field] = 257
    with pytest.raises(wire.ScanProtocolError):
        wire.encode_frame(done)


def test_old_index_wire_version_is_explicitly_refused():
    wire = importlib.import_module("tldw_chatbook.Tools.project_index_scan_protocol")
    with pytest.raises(wire.ScanProtocolError):
        wire.encode_frame({"version": 1, "type": "Close"})
