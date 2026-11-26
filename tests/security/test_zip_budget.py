"""Archive budgets on the ERCOT parsers (SECURITY.md RC-2).

`parse_archive` / `parse_dam_archive` admit a member only after
`secprim.safe_zip_members` has checked the archive's *declared* sizes, names and
compression ratios -- so an over-budget or hostile archive is refused before a
single byte of it is decompressed. A normal archive (synthetic, in the real
shape) still parses.
"""
import io
import os
import zipfile
from unittest import mock

import pandas as pd
import pytest

pytest.importorskip("openpyxl", reason="openpyxl not installed")

from secprim import MalformedInput, TooLarge  # noqa: E402

from data_ingestion import ingest_ercot_history as ingest  # noqa: E402

HUBS = ["HB_HOUSTON"]


def _rtm_workbook() -> bytes:
    frame = pd.DataFrame({
        "Delivery Date": ["01/15/2025"] * 4,
        "Delivery Hour": [1] * 4,
        "Delivery Interval": [1, 2, 3, 4],
        "Repeated Hour Flag": ["N"] * 4,
        "Settlement Point Name": ["HB_HOUSTON"] * 4,
        "Settlement Point Type": ["HU"] * 4,
        "Settlement Point Price": [20.0, 21.0, 22.0, 23.0],
    })
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        frame.to_excel(xw, sheet_name="Jan", index=False)
    return buf.getvalue()


def _dam_workbook() -> bytes:
    frame = pd.DataFrame({
        "Delivery Date": ["01/15/2025"] * 2,
        "Hour Ending": ["01:00", "02:00"],
        "Repeated Hour Flag": ["N"] * 2,
        "Settlement Point": ["HB_HOUSTON"] * 2,
        "Settlement Point Price": [25.0, 26.0],
    })
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        frame.to_excel(xw, sheet_name="Jan", index=False)
    return buf.getvalue()


def _zip(tmp_path, name, members):
    path = tmp_path / name
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for member_name, data in members:
            zf.writestr(member_name, data)
    return path


def test_normal_rtm_archive_parses(tmp_path):
    path = _zip(tmp_path, "RTMLZHBSPP_2025.zip", [("rpt.RTMLZHBSPP_2025.xlsx", _rtm_workbook())])
    df = ingest.parse_archive(path, HUBS)
    assert len(df) == 4
    assert set(df["settlement_point"]) == {"HB_HOUSTON"}


def test_normal_dam_archive_parses(tmp_path):
    path = _zip(tmp_path, "DAMLZHBSPP_2025.zip", [("rpt.DAMLZHBSPP_2025.xlsx", _dam_workbook())])
    df = ingest.parse_dam_archive(path, HUBS)
    assert len(df) == 2


def test_real_archive_sizes_fit_the_budget():
    # The caps must comfortably admit the largest archive ERCOT actually
    # publishes (~23 MB workbook in a ~14 MB zip, measured 2024-2026): at
    # least several times that, so a fatter year never trips the budget.
    assert ingest.MAX_MEMBER_BYTES >= 5 * 23_000_000
    assert ingest.MAX_ARCHIVE_BYTES >= ingest.MAX_MEMBER_BYTES


@pytest.mark.parametrize("parser", [ingest.parse_archive, ingest.parse_dam_archive])
def test_over_budget_member_is_refused_before_read(tmp_path, parser, monkeypatch):
    # Shrink the cap instead of writing a multi-hundred-MB file; the check is
    # on the declared size, so the behaviour is identical.
    monkeypatch.setattr(ingest, "MAX_MEMBER_BYTES", 1024)
    path = _zip(tmp_path, "big.zip", [("rpt.xlsx", b"x" * 4096)])
    with mock.patch.object(zipfile.ZipFile, "read", side_effect=AssertionError("read before budget")):
        with pytest.raises(TooLarge):
            parser(path, HUBS)


@pytest.mark.parametrize("parser", [ingest.parse_archive, ingest.parse_dam_archive])
def test_total_budget_is_enforced_across_members(tmp_path, parser, monkeypatch):
    monkeypatch.setattr(ingest, "MAX_ARCHIVE_BYTES", 5000)
    # Incompressible members so only the running total, not the ratio, trips.
    path = _zip(tmp_path, "many.zip", [(f"m{i}.xlsx", os.urandom(2000)) for i in range(4)])
    with mock.patch.object(zipfile.ZipFile, "read", side_effect=AssertionError("read before budget")):
        with pytest.raises(TooLarge):
            parser(path, HUBS)


@pytest.mark.parametrize("parser", [ingest.parse_archive, ingest.parse_dam_archive])
def test_suspicious_compression_ratio_is_refused(tmp_path, parser):
    # A highly compressible member (ratio >> 100:1) is the signature of a
    # decompression bomb; it is refused on the declared ratio, unread.
    path = _zip(tmp_path, "bomb.zip", [("rpt.xlsx", b"\0" * 50_000_000)])
    with mock.patch.object(zipfile.ZipFile, "read", side_effect=AssertionError("read before budget")):
        with pytest.raises(MalformedInput):
            parser(path, HUBS)


@pytest.mark.parametrize("name", ["../rpt.xlsx", "/abs/rpt.xlsx"])
def test_unsafe_member_name_is_refused(tmp_path, name):
    path = _zip(tmp_path, "slip.zip", [(name, _rtm_workbook())])
    with pytest.raises(MalformedInput):
        ingest.parse_archive(path, HUBS)
