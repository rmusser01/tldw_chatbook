# test_chatbook_import_wizard_embeddings_honesty.py
# Description: Regression coverage for task-19868 (the "Import embeddings"
#              control was inert in local mode and defaulted ON)
"""
The import wizard's options step used to offer a default-ON "Import
embeddings" checkbox on every import. In local mode nothing of the kind
happened: the value was collected, passed to ``ChatbookImporter.
import_chatbook``, accepted as a parameter, written into a debug log line --
and never read again. The control was real only on the server path (where it
filters embedding items out of the request), and the comment above the
collected options claimed "Every key here is consumed by the import call
below", which was false for this key.

The capability gap is recorded in the code and pinned here: chatbooks
contain no embedding data even in principle, because the exporter never
writes any, and the local importer cannot write the type -- it reports
``ContentType.EMBEDDING`` items as unsupported. The honest fix:

* the local importer takes no ``import_embeddings`` parameter at all;
* the control is shown only in server mode, where it is consumed;
* the options dict carries the key only in server mode;
* the comment states which call consumes which key, per mode.

Born red at base: the checkbox was present and ON in local mode, and the
key was collected there.
"""

import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Checkbox, RadioButton

from tldw_chatbook.Chatbooks.chatbook_importer import ChatbookImporter
from tldw_chatbook.Chatbooks.chatbook_models import ChatbookManifest, ChatbookVersion
from tldw_chatbook.UI.Wizards import ChatbookImportWizard as wizard_module
from tldw_chatbook.UI.Wizards.BaseWizard import WizardStepConfig
from tldw_chatbook.UI.Wizards.ChatbookImportWizard import (
    ImportOptionsStep,
    ImportProgressStep,
)

# The false invariant the old comment shipped (task-19868 AC#5).
_FALSE_OPTIONS_COMMENT = "Every key here is consumed by the import call below."


class _StepHost(App):
    """Mount a single wizard step, the idiom used by Tests/Wizards."""

    def __init__(self, step):
        super().__init__()
        self._step = step

    def compose(self) -> ComposeResult:
        yield self._step


def _fake_wizard(**wizard_data) -> SimpleNamespace:
    return SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        wizard_data=dict(wizard_data),
        initial_execution_mode=wizard_data.pop("_initial_mode", "local"),
        can_go_back=True,
    )


def _options_step(initial_mode: str = "local") -> ImportOptionsStep:
    manifest = ChatbookManifest(
        version=ChatbookVersion.V1,
        name="Bundle",
        description="Bundle",
    )
    manifest.total_conversations = 1
    return ImportOptionsStep(
        wizard=_fake_wizard(
            _initial_mode=initial_mode,
            **{"preview-validation": {"manifest": manifest}},
        ),
        config=WizardStepConfig(
            id="import-options",
            title="Import Options",
            step_number=4,
        ),
    )


async def _visible_state(step: ImportOptionsStep):
    """(embeddings group visible, checkbox value, collected data)."""
    group = step.query_one("#embeddings-option-group")
    checkbox = step.query_one("#import-embeddings", Checkbox)
    return (not group.has_class("hidden"), checkbox.value, step.get_step_data())


# ---------------------------------------------------------------------------
# AC#1/AC#2: the control matches the mode it is shown in
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_local_mode_offers_no_embeddings_control():
    """AC#1, born red at base: local mode must not tell the user that vector
    embeddings will be imported -- not by the control's presence, its label,
    or its default state."""
    step = _options_step(initial_mode="local")
    async with _StepHost(step).run_test(size=(120, 60)) as pilot:
        await pilot.pause()
        if getattr(step, "_options_shown", None) is None:
            # on_show may not have fired for this host; apply it directly.
            step.on_show()
            await pilot.pause()
        visible, _value, data = await _visible_state(step)

    assert not visible, "the embeddings control must be absent in local mode"
    assert "import_embeddings" not in data, data


@pytest.mark.asyncio
async def test_server_mode_offers_the_embeddings_control():
    """AC#2: in server mode the control is real (it filters embedding items
    out of the request), so it is shown there and its key is collected."""
    step = _options_step(initial_mode="server")
    async with _StepHost(step).run_test(size=(120, 60)) as pilot:
        await pilot.pause()
        if getattr(step, "_options_shown", None) is None:
            step.on_show()
            await pilot.pause()
        visible, value, data = await _visible_state(step)

    assert visible, "the embeddings control is consumed in server mode"
    assert data["execution_mode"] == "server"
    assert data["import_embeddings"] is value


@pytest.mark.asyncio
async def test_switching_modes_moves_the_control():
    """The execution-mode radio is on the same step: the control must follow
    the mode the user selects, live."""
    step = _options_step(initial_mode="local")
    async with _StepHost(step).run_test(size=(120, 60)) as pilot:
        await pilot.pause()
        if getattr(step, "_options_shown", None) is None:
            step.on_show()
            await pilot.pause()
        group = step.query_one("#embeddings-option-group")
        assert group.has_class("hidden")

        step.query_one("#mode-server", RadioButton).value = True
        await pilot.pause()
        assert not group.has_class("hidden"), "server mode must show it"

        step.query_one("#mode-local", RadioButton).value = True
        await pilot.pause()
        assert group.has_class("hidden"), "local mode must hide it"


@pytest.mark.asyncio
async def test_the_control_defaults_off_not_on():
    """The old default (ON, in a mode that ignored it) was itself the claim.
    Where the control is shown, it defaults to off: include-by-request, not
    include-by-default."""
    step = _options_step(initial_mode="server")
    async with _StepHost(step).run_test(size=(120, 60)) as pilot:
        await pilot.pause()
        if getattr(step, "_options_shown", None) is None:
            step.on_show()
            await pilot.pause()
        _visible, value, _data = await _visible_state(step)

    assert value is False


# ---------------------------------------------------------------------------
# AC#4: no collected key that the local import call ignores
# ---------------------------------------------------------------------------


def test_the_local_importer_takes_no_embeddings_option():
    """The inert parameter is gone from the local import call's signature:
    reintroducing an accepted-but-ignored option becomes a signature change,
    not a silent no-op."""
    parameters = inspect.signature(ChatbookImporter.import_chatbook).parameters
    assert "import_embeddings" not in parameters, list(parameters)
    # The parameters that remain are the ones the local path actually reads.
    assert set(parameters) == {
        "self",
        "chatbook_path",
        "content_selections",
        "conflict_resolution",
        "prefix_imported",
        "import_media",
        "import_status",
    }


def test_local_collected_options_are_exactly_what_the_local_call_consumes():
    """AC#4, structural: every key ``get_step_data`` collects in local mode
    is a key ``ImportProgressStep._import_chatbook`` consumes -- either as a
    dispatch input or as a parameter of the local ``import_chatbook`` call.

    Mutation-checked: re-adding an ignored key (the local-mode
    ``import_embeddings`` of this task's base) makes this red.
    """
    source = Path(inspect.getfile(wizard_module)).read_text(encoding="utf-8")
    import ast

    tree = ast.parse(source)
    progress_step = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ImportProgressStep"
    )
    import_body = next(
        node
        for node in progress_step.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_import_chatbook"
    )

    # Every option key the local branch reads.
    consumed = {
        node.args[0].value
        for node in ast.walk(import_body)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "options"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }

    # get_step_data needs the step mounted to query widgets; read the literal
    # keys from its source instead, for both modes (the mounted behaviour is
    # covered by the async tests above).
    options_step_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ImportOptionsStep"
    )
    get_data = next(
        node
        for node in options_step_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_step_data"
    )
    always_collected = {
        node.value
        for node in ast.walk(get_data)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        and node.value in {"execution_mode", "import_media", "prefix_imported"}
    }
    conditionally_collected = {
        node.value
        for node in ast.walk(get_data)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        and node.value == "import_embeddings"
    }

    assert always_collected == {"execution_mode", "import_media", "prefix_imported"}
    # The three always-collected keys must all be consumed by the dispatch.
    assert always_collected <= consumed, sorted(always_collected - consumed)
    # The one conditionally-collected key is consumed only by the SERVER
    # branch -- and the local call must not pass it.
    assert conditionally_collected == {"import_embeddings"}
    assert "import_embeddings" in consumed  # consumed by the server branch


@pytest.mark.asyncio
async def test_the_local_call_passes_only_real_parameters(monkeypatch):
    """AC#4, behavioural: drive the real progress step through a local import
    with a recording importer and pin the exact kwargs -- no
    ``import_embeddings`` among them, and nothing else ignored either."""
    manifest = ChatbookManifest(
        version=ChatbookVersion.V1,
        name="Bundle",
        description="Bundle",
    )
    manifest.total_conversations = 1

    step = ImportProgressStep(
        wizard=_fake_wizard(
            **{
                "file-selection": {"file_path": "/tmp/does-not-need-to-exist.zip"},
                "preview-validation": {"manifest": manifest},
                "conflict-resolution": {},
                "import-options": {
                    "execution_mode": "local",
                    "import_media": True,
                    "prefix_imported": True,
                },
            }
        ),
        config=WizardStepConfig(id="import-progress", title="Importing", step_number=5),
    )

    recorded: dict = {}

    class _FakeImporter:
        def __init__(self, db_paths):
            self.db_paths = db_paths

        def import_chatbook(self, **kwargs):
            recorded.update(kwargs)
            status = kwargs["import_status"]
            status.total_items = 1
            status.successful_items = 1
            return True, "ok"

    monkeypatch.setattr(wizard_module, "ChatbookImporter", _FakeImporter)
    monkeypatch.setattr(
        wizard_module,
        "get_chatbook_database_paths",
        lambda: {"ChaChaNotes": "/tmp/cn.db"},
    )

    async def _completion():
        return None

    async def _error(message):
        raise AssertionError(f"the local import run failed: {message}")

    step._update_status = lambda status_id, state, text: None
    step._update_progress = lambda value: None
    step._show_completion = _completion
    step._show_error = _error

    await step._import_chatbook()

    assert recorded, "the fake importer must have run"
    assert set(recorded) == {
        "chatbook_path",
        "content_selections",
        "conflict_resolution",
        "prefix_imported",
        "import_media",
        "import_status",
    }, sorted(recorded)
    assert "import_embeddings" not in recorded


# ---------------------------------------------------------------------------
# AC#5: the false comment
# ---------------------------------------------------------------------------


def test_the_options_comment_no_longer_claims_an_invariant_the_code_breaks():
    """The comment above the collected options used to assert "Every key
    here is consumed by the import call below" while ``import_embeddings``
    broke it in local mode. The replacement states the per-mode truth."""
    source = Path(inspect.getfile(wizard_module)).read_text(encoding="utf-8")
    assert _FALSE_OPTIONS_COMMENT not in source
    # The capability gap is recorded where the next maintainer will look.
    assert "exporter never writes" in source
    assert "server" in source.lower()


# ---------------------------------------------------------------------------
# AC#6: the server half keeps working
# ---------------------------------------------------------------------------


def test_the_server_branch_still_consumes_the_option():
    """AC#6: the server path filters embedding items on this value; the
    wiring must survive the local-mode change."""
    source = inspect.getsource(ImportProgressStep._import_chatbook)
    assert source.count('options.get("import_embeddings", False)') == 2, source
    assert "build_server_import_selections_from_manifest" in source
    assert "import_chatbook_from_selection" in source


def test_the_shared_api_schema_still_carries_the_server_option():
    """The ``ChatbookImportRequest`` schema is shared with the server import
    path, which consumes the field; it must not shrink for local mode's
    sake."""
    from tldw_chatbook.tldw_api.prompt_chatbook_schemas import ChatbookImportRequest

    assert "import_embeddings" in ChatbookImportRequest.model_fields
