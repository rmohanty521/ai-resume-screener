from types import SimpleNamespace as NS

import pytest

from resume_screener.batch import ParseCache, parse_batch
from resume_screener.dataset import RawResume
from resume_screener.parser import ParseError, ParseResult, ResumeParser, fallback_kwargs

from .conftest import fake_response, make_parsed


def client_returning(resp, calls=None):
    def parse(**kw):
        if calls is not None:
            calls.append(kw)
        return resp
    return NS(beta=NS(messages=NS(parse=parse)))


def test_parser_redacts_before_sending(parsed):
    calls = []
    ResumeParser(client=client_returning(fake_response(parsed), calls)).parse("x jane@x.com")
    assert "[EMAIL]" in calls[0]["messages"][0]["content"]
    assert "fallbacks" not in calls[0]  # Haiku has no server-side fallback


@pytest.mark.parametrize("reason", ["refusal", "max_tokens"])
def test_parser_raises_on_bad_stop_reason(parsed, reason):
    with pytest.raises(ParseError):
        ResumeParser(client=client_returning(fake_response(parsed, reason))).parse("x")


def test_fallback_only_for_supported_models():
    assert fallback_kwargs("claude-opus-5-5")["fallbacks"] == "default"
    assert fallback_kwargs("claude-haiku-5-5") == {}


class FakeParser:
    def __init__(self):
        self.calls = 0

    def parse(self, text):
        self.calls += 1
        if "bad" in text:
            raise RuntimeError("boom")
        return ParseResult(make_parsed(text), 100, 20, 0.1, False, "m")


def test_batch_caches_and_records_failures(tmp_path):
    rs = [RawResume(str(i), "IT", f"resume {i}") for i in range(3)] + [RawResume("x", "IT", "bad")]
    parser = FakeParser()
    docs, stats = parse_batch(rs, parser, ParseCache(tmp_path))
    assert [d.resume_id for d in docs] == ["0", "1", "2"]
    assert stats.failed.keys() == {"x"} and stats.input_tokens == 300

    _, stats2 = parse_batch(rs[:3], parser, ParseCache(tmp_path))
    assert stats2.cached == 3 and parser.calls == 4

    _, stats3 = parse_batch([RawResume("0", "IT", "edited")], parser, ParseCache(tmp_path))
    assert stats3.parsed == 1
