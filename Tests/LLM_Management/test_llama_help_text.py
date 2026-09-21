"""Regression guard for the Llama.cpp argument-help panel.

`LLM_Management_Window._populate_help_text` imports
`LLAMA_CPP_SERVER_ARGS_HELP_TEXT` from `Constants` inside a
`try/except (QueryError, ImportError)`. A deleted or renamed constant therefore
does not crash -- it silently leaves the help panel empty (task-32810.2 nearly
shipped exactly that: the constant sat just past the dead `css_content` block
and was swept away with it). These asserts fail loudly instead.
"""


def test_llama_help_constant_is_importable_and_populated():
    # Import exactly as _populate_help_text does.
    from tldw_chatbook.Constants import LLAMA_CPP_SERVER_ARGS_HELP_TEXT

    assert isinstance(LLAMA_CPP_SERVER_ARGS_HELP_TEXT, str)
    assert LLAMA_CPP_SERVER_ARGS_HELP_TEXT.strip(), "help text must not be empty"
    # A couple of stable landmarks from the panel body.
    assert "--gpu-layers" in LLAMA_CPP_SERVER_ARGS_HELP_TEXT
    assert "Server & Model Params" in LLAMA_CPP_SERVER_ARGS_HELP_TEXT


def test_populate_help_text_still_imports_the_constant():
    """The panel is fed by that exact name; guard against the import drifting."""
    import inspect
    from tldw_chatbook.UI.LLM_Management_Window import LLMManagementWindow

    src = inspect.getsource(LLMManagementWindow._populate_help_text)
    assert "LLAMA_CPP_SERVER_ARGS_HELP_TEXT" in src
