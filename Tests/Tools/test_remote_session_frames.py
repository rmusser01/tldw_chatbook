import pytest
from tldw_chatbook.Tools.remote_session_frames import (
    LINE, REQUEST, STATUS, FrameError, FrameReader, decode_status,
    encode_frame, encode_status,
)

def test_roundtrip_split_across_arbitrary_chunks():
    data = encode_frame(REQUEST, 7, b"hello") + encode_frame(LINE, 8, b"") + encode_frame(STATUS, 7, encode_status(0, None))
    reader = FrameReader(max_body=1024)
    got = []
    for i in range(len(data)):
        got += reader.feed(data[i:i + 1])
    assert got == [(REQUEST, 7, b"hello"), (LINE, 8, b""), (STATUS, 7, encode_status(0, None))]

def test_oversize_header_rejected_before_body_arrives():
    reader = FrameReader(max_body=10)
    with pytest.raises(FrameError):
        reader.feed(encode_frame(REQUEST, 1, b"x" * 11)[:9])

def test_status_codec():
    assert decode_status(encode_status(75, None)) == (75, None)
    assert decode_status(encode_status(None, 9)) == (None, 9)

def test_request_id_range_checked():
    with pytest.raises(ValueError):
        encode_frame(REQUEST, 2**32, b"")
