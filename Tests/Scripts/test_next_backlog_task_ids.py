"""The id allocator leads past the max and never prints a claimed id (TASK-35131)."""

from pathlib import Path

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from next_backlog_task_ids import (  # noqa: E402
    DEFAULT_BUFFER,
    allocate,
    existing_task_ids,
)

pytestmark = pytest.mark.unit


def _plant(root: Path, relative: str, frontmatter_id: int | None = None) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if frontmatter_id is not None:
        path.write_text(
            f"---\nid: TASK-{frontmatter_id}\ntitle: planted\n---\n\nbody\n",
            encoding="utf-8",
        )
    else:
        path.write_text("no frontmatter\n", encoding="utf-8")


def test_scans_all_three_directories_and_both_claim_surfaces(tmp_path):
    _plant(tmp_path, "backlog/tasks/task-10 - filename-only.md")
    _plant(tmp_path, "backlog/completed/task-11 - filename-only.md")
    # Frontmatter-only claim: hand-edited filename no longer carries the id.
    _plant(tmp_path, "backlog/archive/tasks/task-renamed - frontmatter.md", 12)
    assert existing_task_ids(tmp_path) == {10, 11, 12}


def test_allocation_leads_past_max_by_the_buffer(tmp_path):
    _plant(tmp_path, "backlog/tasks/task-100 - max.md")
    assert allocate(2, 30, tmp_path) == [130, 131]


def test_default_buffer_is_the_incident_derived_lead():
    assert DEFAULT_BUFFER == 30


def test_a_sparse_high_claim_raises_the_max_not_the_start(tmp_path):
    # The sparse claim IS the max once scanned: allocation leads past IT,
    # never past the denser cluster below. (The skip loop additionally
    # guards claims the scan cannot see -- one racing in concurrently.)
    _plant(tmp_path, "backlog/tasks/task-100 - max.md")
    _plant(tmp_path, "backlog/tasks/task-130 - sparse.md")
    assert allocate(3, 30, tmp_path) == [160, 161, 162]


def test_dotted_subtask_stems_count_via_their_integer_part(tmp_path):
    _plant(tmp_path, "backlog/tasks/task-7.3 - subtask.md")
    assert existing_task_ids(tmp_path) == {7}
    assert allocate(1, 1, tmp_path) == [8]


def test_empty_tree_allocates_from_the_buffer(tmp_path):
    assert allocate(2, 30, tmp_path) == [30, 31]


@pytest.mark.parametrize("count,buffer", [(0, 30), (1, 0), (-1, 30)])
def test_invalid_arguments_exit_with_message(tmp_path, count, buffer):
    with pytest.raises(SystemExit):
        allocate(count, buffer, tmp_path)
