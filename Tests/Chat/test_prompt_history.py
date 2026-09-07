"""Tests for the JSONL prompt history store (ported for backlog task-1364).

Covers:
- JSONL append/load round-trip with async (off-loop) file IO and real timestamps.
- Shell-style indexing: 0 is the live draft pseudo-entry, negatives walk back.
- validate_*-style index clamping at both boundaries.
- Consecutive-duplicate dedupe, including the rapid-send race (optimistic
  in-memory update with rollback on write failure).
- max_entries cap enforcement on both load and append.
- Most-recent-wins prefix completion for ghost text.
"""

import asyncio
import json

import pytest

from tldw_chatbook.Chat.prompt_history import PromptHistory


def _seed(history: PromptHistory, *inputs: str) -> None:
    """Seed stored entries directly (no file IO)."""
    history._entries = [{"input": text, "timestamp": 0.0} for text in inputs]
    history._loaded = True


@pytest.fixture
def history(tmp_path):
    return PromptHistory(tmp_path / "prompt_history.jsonl")


class TestAppendLoadRoundTrip:
    pytestmark = pytest.mark.asyncio

    async def test_append_then_load_round_trip(self, tmp_path):
        path = tmp_path / "prompt_history.jsonl"
        writer = PromptHistory(path)
        assert await writer.append("first prompt")
        assert await writer.append("second prompt")

        reader = PromptHistory(path)
        await reader.load()
        assert reader.size == 2
        assert (await reader.get_entry(-1))["input"] == "second prompt"
        assert (await reader.get_entry(-2))["input"] == "first prompt"

    async def test_append_writes_one_json_object_per_line(self, history):
        await history.append("hello world")
        lines = history.path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["input"] == "hello world"
        assert isinstance(entry["timestamp"], float)

    async def test_persisted_timestamps_survive_round_trip(self, tmp_path):
        """Stored entries report their persisted timestamp, not a fabricated one."""
        path = tmp_path / "prompt_history.jsonl"
        writer = PromptHistory(path)
        await writer.append("timed prompt")
        written = json.loads(path.read_text(encoding="utf-8").splitlines()[0])

        reader = PromptHistory(path)
        await reader.load()
        entry = await reader.get_entry(-1)
        assert entry["timestamp"] == written["timestamp"]
        assert entry["timestamp"] > 0

    async def test_load_missing_file_is_empty_history(self, history):
        await history.load()
        assert history.size == 0

    async def test_load_tolerates_corrupt_lines(self, tmp_path):
        path = tmp_path / "prompt_history.jsonl"
        path.write_text(
            '{"input": "good", "timestamp": 1.0}\n'
            "not json at all\n"
            '{"input": 42}\n'
            "\n"
            '{"input": "also good", "timestamp": 2.0}\n',
            encoding="utf-8",
        )
        history = PromptHistory(path)
        await history.load()
        assert history.size == 2
        assert (await history.get_entry(-1))["input"] == "also good"

    async def test_append_empty_text_is_skipped(self, history):
        assert await history.append("") is False
        assert history.size == 0
        assert not history.path.exists()

    async def test_consecutive_duplicates_are_skipped(self, history):
        assert await history.append("same") is True
        assert await history.append("same") is False
        assert await history.append("different") is True
        assert await history.append("same") is True  # Non-consecutive is kept.
        assert history.size == 3
        lines = history.path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 3

    async def test_append_clears_stashed_draft(self, history):
        history.stash_draft("in progress")
        await history.append("sent")
        assert history.current == ""


class TestMaxEntriesCap:
    pytestmark = pytest.mark.asyncio

    async def test_load_keeps_only_most_recent_entries(self, tmp_path):
        path = tmp_path / "prompt_history.jsonl"
        with path.open("w", encoding="utf-8") as history_file:
            for index in range(5):
                history_file.write(
                    json.dumps({"input": f"prompt {index}", "timestamp": float(index)})
                    + "\n"
                )
        history = PromptHistory(path, max_entries=3)
        await history.load()
        assert history.size == 3
        assert (await history.get_entry(-1))["input"] == "prompt 4"
        assert (await history.get_entry(-3))["input"] == "prompt 2"

    async def test_append_rewrites_file_with_tail_when_cap_exceeded(self, tmp_path):
        path = tmp_path / "prompt_history.jsonl"
        history = PromptHistory(path, max_entries=3)
        for index in range(4):
            assert await history.append(f"prompt {index}") is True

        assert history.size == 3
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 3  # File was rewritten, not just appended.
        inputs = [json.loads(line)["input"] for line in lines]
        assert inputs == ["prompt 1", "prompt 2", "prompt 3"]
        assert (await history.get_entry(-3))["input"] == "prompt 1"
        with pytest.raises(IndexError):
            await history.get_entry(-4)  # Oldest entry was dropped.

    async def test_cap_of_one_keeps_latest_only(self, tmp_path):
        history = PromptHistory(tmp_path / "prompt_history.jsonl", max_entries=1)
        await history.append("old")
        await history.append("new")
        assert history.size == 1
        assert (await history.get_entry(-1))["input"] == "new"
        lines = history.path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1

    async def test_concurrent_appends_are_serialized(self, tmp_path):
        """Racing appends (e.g. composer + controller sharing one instance)
        cannot interleave a cap rewrite with another append — every entry
        lands exactly once, in order."""
        path = tmp_path / "prompt_history.jsonl"
        history = PromptHistory(path, max_entries=5)
        await asyncio.gather(*(history.append(f"prompt {index}") for index in range(8)))

        inputs = [entry["input"] for entry in history._entries]
        assert inputs == [f"prompt {index}" for index in range(3, 8)]
        lines = path.read_text(encoding="utf-8").splitlines()
        assert [json.loads(line)["input"] for line in lines] == inputs


class TestWriteFailureRollback:
    pytestmark = pytest.mark.asyncio

    async def test_failed_write_rolls_back_optimistic_entry(self, tmp_path):
        """A failed write must not poison the dedupe check for the next send."""
        # A directory at the history path makes every write fail.
        history = PromptHistory(tmp_path / "prompt_history.jsonl")
        history.path.mkdir()

        assert await history.append("unsaved") is False
        assert history.size == 0  # Optimistic entry rolled back.

        # Retry once the path is writable: not treated as a duplicate.
        history.path.rmdir()
        assert await history.append("unsaved") is True
        assert history.size == 1

    async def test_failed_write_keeps_stashed_draft(self, tmp_path):
        history = PromptHistory(tmp_path / "prompt_history.jsonl")
        history.path.mkdir()
        history.stash_draft("draft")
        assert await history.append("unsaved") is False
        assert history.current == "draft"


class TestIndexClamping:
    def test_clamp_with_empty_history(self, history):
        assert history.clamp_index(0) == 0
        assert history.clamp_index(-1) == 0
        assert history.clamp_index(-100) == 0
        assert history.clamp_index(5) == 0

    def test_clamp_at_boundaries(self, history):
        _seed(history, "a", "b", "c")
        assert history.clamp_index(0) == 0
        assert history.clamp_index(-1) == -1
        assert history.clamp_index(-3) == -3
        assert history.clamp_index(-4) == -3  # Clamped to -size.
        assert history.clamp_index(-100) == -3
        assert history.clamp_index(1) == 0  # Positive clamps to live draft.


class TestGetEntry:
    pytestmark = pytest.mark.asyncio

    async def test_index_zero_returns_stashed_draft(self, history):
        history.stash_draft("draft text")
        entry = await history.get_entry(0)
        assert entry["input"] == "draft text"

    async def test_index_zero_without_draft_returns_empty(self, history):
        entry = await history.get_entry(0)
        assert entry["input"] == ""

    async def test_negative_indexes_walk_backwards(self, history):
        await history.append("one")
        await history.append("two")
        assert (await history.get_entry(-1))["input"] == "two"
        assert (await history.get_entry(-2))["input"] == "one"

    async def test_out_of_range_indexes_raise(self, history):
        await history.append("only")
        with pytest.raises(IndexError):
            await history.get_entry(1)
        with pytest.raises(IndexError):
            await history.get_entry(-2)

    async def test_draft_stash_restore_round_trip(self, history):
        """Stashing a draft never affects stored entries and is restored."""
        await history.append("stored")
        history.stash_draft("my draft")
        assert (await history.get_entry(-1))["input"] == "stored"
        assert (await history.get_entry(0))["input"] == "my draft"
        history.stash_draft("edited draft")
        assert (await history.get_entry(0))["input"] == "edited draft"
        history.clear_draft()
        assert (await history.get_entry(0))["input"] == ""


class TestComplete:
    def test_most_recent_match_wins(self, history):
        _seed(history, "fix the bug", "find the bug", "fix the tests")
        assert history.complete("fi") == "fix the tests"
        assert history.complete("fix") == "fix the tests"
        assert history.complete("fin") == "find the bug"

    def test_exact_match_is_not_suggested(self, history):
        _seed(history, "hello")
        assert history.complete("hello") is None

    def test_empty_prefix_matches_nothing(self, history):
        _seed(history, "hello")
        assert history.complete("") is None

    def test_no_match_returns_none(self, history):
        _seed(history, "hello")
        assert history.complete("xyz") is None
