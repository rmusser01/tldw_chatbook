"""Install a new runner beside a pinned original Windows journey replay."""

import argparse
import hashlib
import json
from pathlib import Path


def change(source, before, after):
    if source.count(before) != 1:
        raise ValueError("Pinned replay source no longer matches the installation point")
    return source.replace(before, after)


def install(inputs, destination):
    config = json.loads(inputs.read_bytes())
    old = Path(config["candidate"]["runner"]).parent
    seed = Path(config["candidate"]["directory"]) / "replay-manifest.json"
    seed_sha = config["candidate"]["manifest_sha256"]
    if hashlib.sha256(seed.read_bytes()).hexdigest() != seed_sha:
        raise ValueError("Previous candidate manifest changed")
    expected = json.loads(seed.read_bytes())["tool_sha256"]
    sources = {}
    for name in ("replay.py", "inspect_replay.py", "run_replay.ps1"):
        raw = (old / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected[name]:
            raise ValueError("Previous runner changed")
        sources[name] = raw
    replay = sources["replay.py"].decode().replace("\r\n", "\n")
    replay = change(replay, '    for name, expected in manifest["tool_sha256"].items():',
                    '    tools_directory = HERE\n'
                    f'    if directory == Path({str(seed.parent)!r}).resolve():\n'
                    f'        if sha((directory / "replay-manifest.json").read_bytes()) != {seed_sha!r}:\n'
                    '            raise ValueError("Original seed manifest changed")\n'
                    f'        tools_directory = Path({str(old)!r})\n'
                    '    for name, expected in manifest["tool_sha256"].items():')
    replay = change(replay, 'if sha((HERE / name).read_bytes()) != expected:',
                    'if sha((tools_directory / name).read_bytes()) != expected:')
    begin = replay.index("    if APP in changed:")
    end = replay.index("    bootstrap = child(target, PROCESS)", begin)
    replay = replay[:begin] + """    if (app.count(b'console_execution: str = "auto"') != 1 or
        app.count(b'console_execution = "process" if os.name == "nt" and type(self) is TldwCli else "local"') != 1):
        raise RuntimeError("Normal Windows process selection changed")
""" + replay[end:]
    replay = change(replay, '    fresh = {APP, PROCESS, plugin, config}', '''    writer_name = "tldw_chatbook/DB/profile_writer.py"
    writer = child(target, writer_name)
    entry = b"from tldw_chatbook.DB.profile_writer import _serve;"
    body = writer.read_bytes()
    if body.count(entry) != 1:
        raise RuntimeError("Profile writer bootstrap changed")
    atomic(writer, body.replace(entry, b"from writer_guard import guarded_serve as _serve;"))
    atomic(target / "writer_guard.py", (HERE / "writer_guard.py").read_bytes())
    fresh = {APP, PROCESS, plugin, config, writer_name, "writer_guard.py"}''')
    replay = change(replay, 'TOOLS = ("replay.py", "inspect_replay.py", "run_replay.ps1")',
                    'TOOLS = ("replay.py", "inspect_replay.py", "run_replay.ps1", "writer_guard.py")')
    inspector = sources["inspect_replay.py"].decode().replace("\r\n", "\n")
    inspector = change(inspector, '    result["coverage"] = (', '''    if receipt.get("status") == "passed":
        from writer_guard import verify_writer_guards
        result["profile_writer_guards"] = verify_writer_guards(run)
    result["coverage"] = (''')
    compile(replay, "replay.py", "exec")
    compile(inspector, "inspect_replay.py", "exec")
    sources.update({
        "replay.py": replay.encode(), "inspect_replay.py": inspector.encode(),
        "writer_guard.py": Path(__file__).with_name("writer_guard.py").read_bytes(),
    })
    destination.mkdir()
    for name, raw in sources.items():
        (destination / name).write_bytes(raw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("previous_pair_inputs", type=Path)
    parser.add_argument("new_tools_directory", type=Path)
    args = parser.parse_args()
    install(args.previous_pair_inputs, args.new_tools_directory)
