"""Every ERCOT fetch is pinned and capped (SECURITY.md RC-4 / RC-2).

`_fetch` is the single HTTP helper both ingesters use. It must hand
`secprim.safe_fetch` the ercot.com allow-list (so a redirect or DNS answer
pointing anywhere else is refused before a connection is made) and the
caller's byte cap (so a listing or archive body is never read unbounded).
"""
import pytest

from secprim.safe_fetch import UnsafeDestination

from data_ingestion import ingest_ercot_history as ingest


class _Response:
    status_code = 200
    content = b"{}"


def test_fetch_passes_allowlist_and_cap_to_safe_fetch(monkeypatch):
    seen = {}

    def recorder(url, **kwargs):
        seen.update(kwargs, url=url)
        return _Response()

    monkeypatch.setattr(ingest, "safe_fetch", recorder)
    ingest._fetch(ingest.MIS_LIST_URL, {"reportTypeId": 1}, timeout=7, max_bytes=1234)
    assert seen["url"].startswith(ingest.MIS_LIST_URL + "?")
    assert "ercot.com" in seen["allow_hosts"]
    assert all(h.endswith("ercot.com") for h in seen["allow_hosts"])
    assert seen["max_bytes"] == 1234          # the caller's cap, not a sentinel
    assert seen["timeout"] == 7


def test_fetch_refuses_an_off_list_host_before_resolving(monkeypatch):
    # The real primitive, with a resolver that fails loudly if it is ever
    # consulted: the allow-list must reject the host before any DNS or socket.
    real = ingest.safe_fetch

    def never_resolve(*a, **k):
        raise AssertionError("resolver consulted for an off-list host")

    monkeypatch.setattr(ingest, "safe_fetch",
                        lambda url, **kw: real(url, resolver=never_resolve, **kw))
    with pytest.raises(UnsafeDestination):
        ingest._fetch("https://169.254.169.254.example/misapp", {}, timeout=1, max_bytes=10)
    with pytest.raises(UnsafeDestination):
        ingest._fetch("https://www.ercot.com.attacker.example/x", {}, timeout=1, max_bytes=10)


def test_caps_are_bounded_constants():
    for name in ("MAX_LISTING_BYTES", "MAX_DOWNLOAD_BYTES", "MAX_MEMBER_BYTES", "MAX_ARCHIVE_BYTES"):
        assert 0 < getattr(ingest, name) <= 2_000_000_000, name
