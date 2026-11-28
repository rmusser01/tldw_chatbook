"""The audited Studio controls share one vocabulary and one identity apiece."""

from tldw_chatbook.Research_Workspace.action_registry import (
    RESEARCH_ACTIONS,
    PLANNED_OUTPUTS,
)
from tldw_chatbook.Research_Workspace.contracts import WorkspaceDataSource
from tldw_chatbook.Research_Workspace.studio_mapping import PRIMARY_OUTPUT_MAPPINGS
from tldw_chatbook.Research_Workspace.studio_models import (
    ResearchOutputKind,
    ResearchWorkProductKind,
)


def test_studio_registry_covers_outputs_and_work_products_once():
    expected = {kind.value for kind in ResearchOutputKind} | {
        kind.value for kind in ResearchWorkProductKind
    }
    assert len(ResearchOutputKind) == 10
    assert len(ResearchWorkProductKind) == 5
    assert RESEARCH_ACTIONS.keys() == expected
    assert len({row.action_id for row in RESEARCH_ACTIONS.values()}) == 15
    for action_id, row in RESEARCH_ACTIONS.items():
        assert action_id == row.action_id
        assert row.option_schema_id
        assert row.owner_by_authority.keys() == set(WorkspaceDataSource)
        assert all(
            owner.label and owner.recovery for owner in row.owner_by_authority.values()
        )


def test_primary_actions_share_existing_owners_and_planned_labels_are_inert():
    for (authority, kind), owner in PRIMARY_OUTPUT_MAPPINGS.items():
        action = RESEARCH_ACTIONS[kind.value]
        assert action.owner_by_authority[authority].label == owner.owner_label
        assert (
            action.owner_by_authority[authority].reopen_target
            == owner.reopen_destination
        )
    assert [item.label for item in PLANNED_OUTPUTS] == [
        "Research Dossier",
        "Competitive Market Memo",
        "Technical Project Spec",
    ]
    assert not any(hasattr(item, "action_id") for item in PLANNED_OUTPUTS)
