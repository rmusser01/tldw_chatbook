from tldw_chatbook.Chat.llamacpp_think_filter import StartAnchoredThinkFilter


def run_filter(text: str) -> str:
    f = StartAnchoredThinkFilter()
    return f.feed(text) + f.flush()


class TestSplitAcrossChunks:
    def test_open_tag_split_across_chunks(self):
        f = StartAnchoredThinkFilter()
        assert f.feed("<thi") == ""
        assert f.feed("nk>reasoning") == ""
        assert f.feed(" here</think>answer") == "answer"
        assert f.flush() == ""

    def test_close_tag_split_across_chunks(self):
        f = StartAnchoredThinkFilter()
        f.feed("<think>abc")
        assert f.feed("</thi") == ""
        assert f.feed("nk>done") == "done"

    def test_one_char_at_a_time(self):
        f = StartAnchoredThinkFilter()
        out = []
        for ch in "<think>hidden</think>visible":
            out.append(f.feed(ch))
        assert "".join(out) + f.flush() == "visible"


class TestAnchoring:
    def test_mid_reply_literal_tag_survives(self):
        assert run_filter("Here is XML: <think>stuff</think> done") == (
            "Here is XML: <think>stuff</think> done"
        )

    def test_leading_whitespace_before_tag_is_tolerated(self):
        assert run_filter("\n\n<think>x</think>hi") == "hi"

    def test_empty_prefix_block_is_removed(self):
        # Some Qwen generations emit an empty think prefix in no-think mode.
        assert run_filter("<think>\n\n</think>\n\nAnswer") == "Answer"

    def test_plain_text_passes_through(self):
        assert run_filter("just an answer") == "just an answer"

    def test_text_starting_like_tag_but_not_tag_passes(self):
        assert run_filter("<thumbs up>") == "<thumbs up>"

    def test_thinking_tag_variant_supported(self):
        assert run_filter("<thinking>deep</thinking>ok") == "ok"


class TestUnterminated:
    def test_unterminated_start_anchored_block_dropped_on_flush(self):
        f = StartAnchoredThinkFilter()
        assert f.feed("<think>forever") == ""
        assert f.flush() == ""

    def test_ambiguous_prefix_at_stream_end_dropped(self):
        f = StartAnchoredThinkFilter()
        assert f.feed("<thin") == ""
        assert f.flush() == ""
