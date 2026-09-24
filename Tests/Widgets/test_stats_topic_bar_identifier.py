"""TASK-32892 item 5: a non-ASCII chat topic killed the Stats screen at mount.

`TopicBar.compose` built a widget id out of the topic string
(`id=f"bar-{self.topic}"`). Topics come from the user's own conversations,
and Textual validates ids against `^[a-zA-Z_\\-][a-zA-Z0-9_\\-]*$`, so any
topic with a space, a leading digit, or a non-ASCII character raised
`BadIdentifier` inside compose -- which is a mount-time crash of the whole
Stats screen, not a missing bar. Two topics that slugified alike would have
crashed the same way on a duplicate id, so the fix is to stop deriving an id
at all: the bar is already reachable by class within its own `TopicBar`.

Gate-free: a bare Textual `App`, no `Tests/UI/app_factory`.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult

from tldw_chatbook.UI.Screens.stats_screen import TopicBar

#: Real shapes a chat topic takes that Textual rejects as an id.
_HOSTILE_TOPICS = ("日本語", "machine learning", "3d printing", "c++/rust", "")


class _TopicHost(App):
    def __init__(self, topics: tuple[str, ...]) -> None:
        super().__init__()
        self._topics = topics

    def compose(self) -> ComposeResult:
        for index, topic in enumerate(self._topics):
            yield TopicBar(topic, index + 1, len(self._topics) + 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("topic", _HOSTILE_TOPICS)
async def test_a_hostile_topic_still_mounts_its_bar(topic):
    async with _TopicHost((topic,)).run_test() as pilot:
        await pilot.pause()
        bar = pilot.app.query_one(TopicBar)
        assert bar.query_one(".topic-bar-fill") is not None


@pytest.mark.asyncio
async def test_two_topics_that_slugify_alike_do_not_collide():
    """The reason the fix is "no id" rather than "slugify the id"."""
    async with _TopicHost(("machine learning", "machine-learning")).run_test() as pilot:
        await pilot.pause()
        assert len(pilot.app.query(TopicBar)) == 2
