"""skill_file: the fourth runtime tool -- bindings, schema pin, dispatch.

Model: Tests/Agents/test_skill_tool_spawn.py's fake chat_call/registry
scaffolding (AgentService + AgentRunsDB + a scripted fence-protocol
provider). skill_file is NOT a ToolProvider -- its schema is pinned into
runtime_schemas (never disclosure-gated) and its authorization lives on the
per-run SkillFileBindings object, never config.allowed_tools.
"""

import json

from tldw_chatbook.Agents.agent_models import (
    AgentConfig,
    RUN_DONE,
    RUNTIME_TOOL_NAMES,
    RunBudget,
    SKILL_FILE_TOOL_NAME,
    SkillFileBindings,
)
from tldw_chatbook.Agents.agent_runtime import FENCE_OPEN
from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider, ToolCatalogRegistry
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


def _fence(name, args):
    return f"{FENCE_OPEN}\n{json.dumps({'name': name, 'arguments': args})}\n```"


def _skill_file_fence(skill_name, path):
    return _fence(SKILL_FILE_TOOL_NAME, {"skill_name": skill_name, "path": path})


def _base_config():
    return AgentConfig(
        model="m",
        system_prompt="s",
        allowed_tools=("calculator",),
        budget=RunBudget(),
    )


def _registry_with_builtins():
    reg = ToolCatalogRegistry()
    reg.register_provider(BuiltinToolProvider())
    return reg


# --- Step 1 unit tests (brief's exact contract) -----------------------------


def test_skill_file_is_a_runtime_tool_name():
    assert SKILL_FILE_TOOL_NAME == "skill_file"
    # collision exclusion rides existing consumers
    assert SKILL_FILE_TOOL_NAME in RUNTIME_TOOL_NAMES


def test_bindings_object_shape():
    b = SkillFileBindings(authorized=set(), reader=None)
    b.authorized.add("demo")
    assert "demo" in b.authorized


# --- Loop-level tests, run through AgentService (only it wires runtime_
# schemas / the reader closure -- the loop itself just dispatches by name). --


def test_skill_file_schema_offered_first_turn_and_authorized_read_succeeds(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    reg = _registry_with_builtins()

    read_calls = []

    def reader(skill_name, path):
        read_calls.append((skill_name, path))
        return {"content": "REF", "truncated": False, "size": 3}

    bindings = SkillFileBindings(authorized={"demo"}, reader=reader)

    script = [
        {
            "choices": [
                {
                    "message": {
                        "content": _skill_file_fence("demo", "references/api.md")
                    }
                }
            ]
        },
        {"choices": [{"message": {"content": "Done."}}]},
    ]
    calls = []

    def chat_call(**kwargs):
        calls.append(kwargs)
        return script.pop(0)

    service = AgentService(db, reg, chat_call=chat_call, skill_file_bindings=bindings)
    run_id, outcome = service.run_turn(
        conversation_id="c1",
        messages=[{"role": "user", "content": "go"}],
        config=_base_config(),
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE
    assert read_calls == [("demo", "references/api.md")]

    # Active from the FIRST provider call, never disclosure-gated -- unlike
    # a ToolProvider entry, which would need find_tools/load_tools first.
    first_system_content = calls[0]["messages_payload"][0]["content"]
    assert SKILL_FILE_TOOL_NAME in first_system_content

    run = db.get_run(run_id)
    results = [s for s in run["steps"] if s["kind"] == "tool_result"]
    assert any("REF" in r["result"] for r in results)


def test_skill_file_unauthorized_name_is_refused(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    reg = _registry_with_builtins()

    def reader(skill_name, path):
        raise AssertionError("reader must not be called for an unauthorized name")

    # "demo" is active in this run; "other" is not -- the model asks for
    # "other" anyway (e.g. a stale/hallucinated skill name).
    bindings = SkillFileBindings(authorized={"demo"}, reader=reader)

    script = [
        {
            "choices": [
                {
                    "message": {
                        "content": _skill_file_fence("other", "references/api.md")
                    }
                }
            ]
        },
        {"choices": [{"message": {"content": "Done."}}]},
    ]
    service = AgentService(
        db, reg, chat_call=lambda **k: script.pop(0), skill_file_bindings=bindings
    )
    run_id, outcome = service.run_turn(
        conversation_id="c1",
        messages=[{"role": "user", "content": "go"}],
        config=_base_config(),
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE
    run = db.get_run(run_id)
    results = [s for s in run["steps"] if s["kind"] == "tool_result"]
    refusal = results[0]["result"]
    assert refusal.startswith("ERROR:")
    assert "other" in refusal


def test_skill_file_bindings_none_schema_absent_and_falls_through(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    reg = _registry_with_builtins()

    script = [
        {
            "choices": [
                {
                    "message": {
                        "content": _skill_file_fence("demo", "references/api.md")
                    }
                }
            ]
        },
        {"choices": [{"message": {"content": "Done."}}]},
    ]
    calls = []

    def chat_call(**kwargs):
        calls.append(kwargs)
        return script.pop(0)

    # No skill_file_bindings passed at all -- the feature was never
    # configured for this run.
    service = AgentService(db, reg, chat_call=chat_call)
    run_id, outcome = service.run_turn(
        conversation_id="c1",
        messages=[{"role": "user", "content": "go"}],
        config=_base_config(),
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE

    first_system_content = calls[0]["messages_payload"][0]["content"]
    assert SKILL_FILE_TOOL_NAME not in first_system_content

    run = db.get_run(run_id)
    results = [s for s in run["steps"] if s["kind"] == "tool_result"]
    # Falls through to the SAME permission-gate path any other undisclosed/
    # disallowed tool name hits -- not a skill_file-specific refusal.
    assert "Tool not permitted: skill_file" in results[0]["result"]
