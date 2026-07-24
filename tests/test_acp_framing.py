"""Exhaustive coverage for `extract_json_message`'s dual NDJSON / Content-Length framing.

This behavior was moved byte-for-byte from `cptr/utils/agents/acp.py::_extract_json_message`
into `cptr/utils/agents/acp_transport.py::extract_json_message`; these tests pin the quirks
that fixed a real upstream framing bug (issue #84) so a future refactor can't silently
regress them.
"""

from __future__ import annotations

import json

import pytest

from cptr.utils.agents.acp_transport import extract_json_message


def _ndjson(payload: dict) -> bytes:
    return (json.dumps(payload, separators=(",", ":")) + "\n").encode()


def _content_length_frame(payload: dict, sep: bytes = b"\r\n\r\n") -> bytes:
    body = json.dumps(payload, separators=(",", ":")).encode()
    header = f"Content-Length: {len(body)}".encode()
    return header + sep + body


class TestNdjson:
    def test_single_message(self):
        buffer = _ndjson({"jsonrpc": "2.0", "id": 1, "method": "ping"})
        result = extract_json_message(buffer)
        assert result is not None
        message, rest = result
        assert message == {"jsonrpc": "2.0", "id": 1, "method": "ping"}
        assert rest == b""

    def test_multiple_messages_extracted_in_loop(self):
        buffer = _ndjson({"id": 1}) + _ndjson({"id": 2}) + _ndjson({"id": 3})
        messages = []
        while True:
            result = extract_json_message(buffer)
            if result is None:
                break
            message, buffer = result
            messages.append(message)
        assert messages == [{"id": 1}, {"id": 2}, {"id": 3}]
        assert buffer == b""

    def test_partial_json_line_returns_none_until_newline(self):
        partial = b'{"jsonrpc": "2.0", "id": 1'
        assert extract_json_message(partial) is None
        # Completing the line (with the newline) makes it parseable.
        complete = partial + b"}\n"
        result = extract_json_message(complete)
        assert result is not None
        message, rest = result
        assert message == {"jsonrpc": "2.0", "id": 1}
        assert rest == b""

    def test_leading_whitespace_is_stripped(self):
        # `bytes.lstrip()` treats embedded "\n" as whitespace too, so a leading
        # blank-ish run (spaces + newlines) is consumed wholesale and the real
        # message underneath comes back directly -- not as a separate {} entry.
        buffer = b"   \n" + _ndjson({"id": 1})
        result = extract_json_message(buffer)
        assert result is not None
        message, rest = result
        assert message == {"id": 1}
        assert rest == b""

    def test_blank_line_between_messages_is_silently_absorbed(self):
        # A stray blank line ("\n\n") between two NDJSON messages never surfaces as
        # its own {} extraction: lstrip() at the top of extract_json_message eats
        # the blank line (and any other leading whitespace) before the "first
        # line" scan runs, so the *next* real message is returned directly.
        buffer = b"\n" + _ndjson({"id": 1})
        result = extract_json_message(buffer)
        assert result is not None
        message, rest = result
        assert message == {"id": 1}
        assert rest == b""

    def test_blank_line_yields_empty_dict_is_unreachable(self):
        # The `if not line: return {}, ...` branch exists in the source but cannot
        # be reached through the public function: by the time the newline scan
        # runs, buffer[0] is guaranteed non-whitespace (or the buffer is empty),
        # so `line` can never be blank. This test pins that finding as executable
        # documentation rather than asserting the unreachable behavior.
        import itertools

        whitespace = [b" ", b"\n", b"\r", b"\t"]
        for n in range(0, 6):
            for combo in itertools.product(whitespace, repeat=n):
                buffer = b"".join(combo) + b'{"id":1}\n'
                result = extract_json_message(buffer)
                assert result is not None
                message, _rest = result
                assert message == {"id": 1}, f"expected real message, got {message!r}"

    def test_only_whitespace_returns_none(self):
        # Nothing but spaces, no trailing newline: lstrip leaves an empty buffer,
        # and an empty buffer has no newline to find a line in.
        assert extract_json_message(b"   ") is None


class TestContentLength:
    def test_crlf_crlf_separator(self):
        buffer = _content_length_frame({"id": 1}, sep=b"\r\n\r\n")
        result = extract_json_message(buffer)
        assert result is not None
        message, rest = result
        assert message == {"id": 1}
        assert rest == b""

    def test_lf_lf_separator(self):
        buffer = _content_length_frame({"id": 1}, sep=b"\n\n")
        result = extract_json_message(buffer)
        assert result is not None
        message, rest = result
        assert message == {"id": 1}
        assert rest == b""

    def test_incomplete_header_returns_none(self):
        # No blank-line separator at all yet.
        buffer = b"Content-Length: 10"
        assert extract_json_message(buffer) is None

    def test_header_complete_but_body_short_returns_none(self):
        body = json.dumps({"id": 1}).encode()
        header = f"Content-Length: {len(body) + 5}".encode()
        buffer = header + b"\r\n\r\n" + body
        assert extract_json_message(buffer) is None

    def test_exact_length_body_preserves_trailing_bytes(self):
        body = json.dumps({"id": 1}, separators=(",", ":")).encode()
        header = f"Content-Length: {len(body)}".encode()
        trailing = b"Content-Length: 2\r\n\r\n{}"
        buffer = header + b"\r\n\r\n" + body + trailing
        result = extract_json_message(buffer)
        assert result is not None
        message, rest = result
        assert message == {"id": 1}
        assert rest == trailing

    def test_multiple_concatenated_frames(self):
        buffer = _content_length_frame({"id": 1}) + _content_length_frame({"id": 2})
        messages = []
        while True:
            result = extract_json_message(buffer)
            if result is None:
                break
            message, buffer = result
            messages.append(message)
        assert messages == [{"id": 1}, {"id": 2}]
        assert buffer == b""

    def test_case_insensitive_header_name(self):
        body = json.dumps({"id": 1}, separators=(",", ":")).encode()
        header = f"content-length: {len(body)}".encode()
        buffer = header + b"\r\n\r\n" + body
        result = extract_json_message(buffer)
        assert result is not None
        message, rest = result
        assert message == {"id": 1}
        assert rest == b""

    def test_unparseable_length_value_raises_runtime_error(self):
        buffer = b"Content-Length: not-a-number\r\n\r\n{}"
        with pytest.raises(RuntimeError):
            extract_json_message(buffer)


class TestMixedStream:
    def test_content_length_frame_then_ndjson_line(self):
        buffer = _content_length_frame({"id": 1}) + _ndjson({"id": 2})
        result = extract_json_message(buffer)
        assert result is not None
        message, rest = result
        assert message == {"id": 1}
        next_result = extract_json_message(rest)
        assert next_result is not None
        next_message, next_rest = next_result
        assert next_message == {"id": 2}
        assert next_rest == b""

    def test_ndjson_line_then_content_length_frame(self):
        buffer = _ndjson({"id": 1}) + _content_length_frame({"id": 2})
        result = extract_json_message(buffer)
        assert result is not None
        message, rest = result
        assert message == {"id": 1}
        next_result = extract_json_message(rest)
        assert next_result is not None
        next_message, next_rest = next_result
        assert next_message == {"id": 2}
        assert next_rest == b""
