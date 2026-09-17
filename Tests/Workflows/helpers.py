"""Hand-authored authoring fixture; no runtime or profile imports."""


def prompt_definition() -> dict:
    return {
        "name": "Two prompts",
        "version": 1,
        "inputs": {"source_text": "hello"},
        "steps": [
            {
                "id": "prepare",
                "type": "prompt",
                "retry": 0,
                "timeout_seconds": 300,
                "config": {"template": "Prepare: {{ inputs.source_text }}"},
            },
            {
                "id": "finish",
                "type": "prompt",
                "retry": 0,
                "timeout_seconds": 300,
                "config": {"template": "Finish: {{ prepare.text }}"},
            },
        ],
        "metadata": {
            "tldw_workflow": {
                "format_version": 1,
                "workflow_id": "ba9e62aa-9859-4d42-98dc-26e3b10cdba0",
                "revision_id": "b42d45fb-dc91-4645-a6d9-a72c589282bd",
                "parent_revision_ids": [],
                "requirements": {},
            }
        },
    }
