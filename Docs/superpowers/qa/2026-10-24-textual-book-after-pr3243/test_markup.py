import json
import zipfile
from pathlib import Path
from types import SimpleNamespace
import pytest
from Tests.private_profile import private_profile_test


@pytest.mark.asyncio
@pytest.mark.parametrize("label", ["Ordinary plan", "[TODO] plan"])
@private_profile_test
async def test_preview_metadata_observation(request, tmp_path, label):
    from textual.widgets import Static
    from Tests.UI.consolidated_css import ConsolidatedCSSApp, BUNDLED_STYLESHEET
    from tldw_chatbook.Chatbooks.chatbook_models import (
        ChatbookManifest,
        ChatbookVersion,
    )
    from tldw_chatbook.UI.Wizards.ChatbookImportWizard import PreviewValidationStep
    from tldw_chatbook.UI.Wizards.BaseWizard import WizardStepConfig

    source = tmp_path / "bundle.zip"
    manifest = ChatbookManifest(
        version=ChatbookVersion.V1, name=label, description=label, author=label
    )
    with zipfile.ZipFile(source, "w") as z:
        z.writestr("manifest.json", json.dumps(manifest.to_dict()))
    wizard = SimpleNamespace(
        wizard_data={"file-selection": {"file_path": str(source)}},
        refresh_current_step=lambda: None,
    )
    step = PreviewValidationStep(
        wizard, WizardStepConfig(id="preview-validation", title="Preview")
    )

    class Host(ConsolidatedCSSApp):
        CSS_PATH = [BUNDLED_STYLESHEET]

        def compose(self):
            yield step

    app = Host()
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        step.on_show()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert step.manifest is not None and step.validation_passed
        fields = {
            key: step.query_one("#chatbook-" + key, Static).render().plain
            for key in ["title", "description", "author"]
        }
        expected = {"title": "📚 " + label, "description": label, "author": label}
        receipt = {
            "source_label": label,
            "manifest_name": step.manifest.name,
            "rendered": fields,
            "expected_literal": expected,
            "validation_passed": step.validation_passed,
            "errors": str(app._exception),
        }
        (
            Path(__file__).parent
            / ("brackets.json" if label.startswith("[") else "ordinary.json")
        ).write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        if label.startswith("["):
            assert all("[TODO]" not in text for text in fields.values())
        else:
            assert fields == expected
        assert app._exception is None
