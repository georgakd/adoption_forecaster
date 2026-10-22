import dataclasses

import pytest

from quotas_forecasting.correction.citations import verify_citation
from quotas_forecasting.correction.sources import SyntheticSourceStore
from quotas_forecasting.correction.types import Citation
from quotas_forecasting.synthetic.sources import SourceDoc

TOL = 0.005
TEXT = "According to the Office, the share of internet users with tertiary education in Spain reached {v} in 2026. For reference, it was {w} in 2023."


def _store(unit):
    v, w = ("61.8%", "55.0%") if unit == "percent" else ("0.618", "0.550")
    doc = SourceDoc("d1", "Office", "Spain", 2026, unit, "tertiary", TEXT.format(v=v, w=w), 0.618)
    return SyntheticSourceStore([doc]), TEXT.format(v=v, w=w)


def _cite(doc_unit, **kw):
    store, text = _store(doc_unit)
    citation = Citation("d1", text.split(". For")[0] + ".", 0.618, 2026, doc_unit)
    return store, dataclasses.replace(citation, **kw)


def test_verify_accepts_exact_quote():
    store, c = _cite("share")
    assert verify_citation(c, store, TOL)


def test_verify_accepts_percent_unit_conversion():
    store, c = _cite("percent")
    assert verify_citation(c, store, TOL)


def test_verify_accepts_value_within_tolerance():
    store, c = _cite("percent", value=0.620)
    assert verify_citation(c, store, TOL)


def test_verify_rejects_value_outside_tolerance():
    store, c = _cite("share", value=0.70)
    assert not verify_citation(c, store, TOL)


def test_verify_rejects_wrong_year():
    store, c = _cite("share", year=2025)
    assert not verify_citation(c, store, TOL)


def test_verify_rejects_wrong_unit():
    store, c = _cite("share", unit="percent")
    assert not verify_citation(c, store, TOL)
    store, c = _cite("percent", unit="share")
    assert not verify_citation(c, store, TOL)


def test_verify_rejects_quote_not_in_doc():
    store, c = _cite("share", quoted_text="reached 0.618 in 2026 per someone else")
    assert not verify_citation(c, store, TOL)


def test_verify_rejects_unknown_doc_id():
    store, c = _cite("share", doc_id="nope")
    assert not verify_citation(c, store, TOL)


def test_verify_multi_number_quote_needs_matching_year_and_number():
    store, text = _store("share")
    ok = Citation("d1", text, 0.550, 2023, "share")
    assert verify_citation(ok, store, TOL)
    assert not verify_citation(dataclasses.replace(ok, year=2022), store, TOL)


@pytest.mark.parametrize("bad", [Citation("d1", "", 0.1, 2026, "share"), Citation("d1", "x", float("nan"), 2026, "bogus"), Citation(None, None, None, None, None)])
def test_verify_never_raises_on_garbage(bad):
    store, _ = _store("share")
    assert verify_citation(bad, store, TOL) is False
