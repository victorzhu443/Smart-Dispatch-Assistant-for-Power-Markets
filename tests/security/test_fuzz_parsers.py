"""Fuzz the parsers that read untrusted ERCOT archive bytes.

`parse_archive` and `parse_dam_archive` in `data_ingestion/ingest_ercot_history.py`
open a downloaded ZIP archive and read an Excel workbook out of it. The bytes
come off the public ERCOT MIS over HTTP with no authentication — fully
attacker-influenceable content (a MITM, a spoofed/compromised mirror, or a
future feed swap could deliver arbitrary bytes). A parser over that input must
*degrade safely*: it may reject the input, but only by raising one of a declared,
expected exception set — never an undeclared crash type, a hang, or a corrupt
partial result. See ../../../security-agents/fuzzing/ for the method.

We fuzz by writing the fuzzed bytes to a temp file and passing its Path, which is
exactly the shape `download_archive` hands these parsers in production.

`download_and_parse` in `data_ingestion/ingest_recent.py` (the 15-minute cron)
is the third parser over the same feed; it is fuzzed by stubbing the pinned fetch
so the hostile bytes arrive as the response body, exactly as in production.

This file is the REGISTRY of parsers that read untrusted bytes (SECURITY.md
RC-1): a parser not listed here is a review failure.
"""
import os
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

from fuzzing.fuzz_lib import assert_robust

from data_ingestion import ingest_ercot_history as base
from data_ingestion import ingest_recent
from data_ingestion.ingest_ercot_history import (
    IngestionError,
    parse_archive,
    parse_dam_archive,
)

# Declared, expected degradation modes for an archive parser fed hostile bytes:
#   BadZipFile      the bytes are not a valid ZIP container
#   IndexError      a valid-but-empty ZIP (namelist()[0] on no members)
#   ValueError      ZIP member is not a workbook pandas/openpyxl can open
#   KeyError        workbook opened but lacks an expected column
#   OSError         filesystem / decompression I/O failure
#   IngestionError  the module's own "no trustworthy rows" rejection
#   pandas ParserError / EmptyDataError  malformed tabular content
#   RecentIngestError  ingest_recent's own "not a zip / no rows" rejection
#   (secprim's MalformedInput/TooLarge are ValueError subclasses: covered)
ALLOWED = (
    zipfile.BadZipFile,
    IndexError,
    ValueError,
    KeyError,
    OSError,
    IngestionError,
    ingest_recent.RecentIngestError,
    pd.errors.ParserError,
    pd.errors.EmptyDataError,
)

HUBS = ["HB_HOUSTON", "HB_NORTH"]


def _fuzz_archive_bytes(parser):
    """Return a str->object callable that writes the fuzzed text as the bytes of
    a temp .zip and runs `parser` over its Path, mirroring the production call."""

    def run(data: str):
        raw = data.encode("latin-1", "replace")
        fd, name = tempfile.mkstemp(suffix=".zip")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(raw)
            return parser(Path(name), HUBS)
        finally:
            os.unlink(name)

    run.__name__ = parser.__name__
    return run


def test_parse_archive_degrades_safely_on_hostile_bytes():
    # Real-time 15-minute archive parser: arbitrary bytes must not crash the
    # ingest with an undeclared exception type.
    assert_robust(_fuzz_archive_bytes(parse_archive), allowed=ALLOWED,
                  max_examples=200)()


def test_parse_dam_archive_degrades_safely_on_hostile_bytes():
    # Day-ahead archive parser: a differently-shaped workbook, same guarantee.
    assert_robust(_fuzz_archive_bytes(parse_dam_archive), allowed=ALLOWED,
                  max_examples=200)()


class _Body:
    def __init__(self, content: bytes):
        self.content = content
        self.status_code = 200


def test_download_and_parse_degrades_safely_on_hostile_bytes(monkeypatch):
    # The incremental (cron) parser: the fetched body is the untrusted input.
    # Stub the pinned fetch so each fuzz case arrives as that body.
    holder = {}
    monkeypatch.setattr(base, "_fetch", lambda *a, **k: _Body(holder["raw"]))

    def run(data: str):
        holder["raw"] = data.encode("latin-1", "replace")
        return ingest_recent.download_and_parse(doc_id=1, hubs=HUBS)

    run.__name__ = "download_and_parse"
    assert_robust(run, allowed=ALLOWED, max_examples=200)()
