import dataclasses

import pytest

from adoption_forecaster.correction.citations import citation_supports_correction, verify_citation
from adoption_forecaster.correction.sources import SyntheticSourceStore
from adoption_forecaster.correction.types import Citation, Correction
from adoption_forecaster.synthetic.sources import SourceDoc

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


def _two_year_store():
    text = "In 2025 it was 60%; in 2026 it was 62%."
    return SyntheticSourceStore([SourceDoc("d2", "Office", "Spain", 2026, "percent", "tertiary", text, 0.62)]), text


def test_verify_binds_year_to_its_number():
    store, text = _two_year_store()
    assert verify_citation(Citation("d2", text, 0.62, 2026, "percent"), store, TOL)
    assert verify_citation(Citation("d2", text, 0.60, 2025, "percent"), store, TOL)
    assert not verify_citation(Citation("d2", text, 0.60, 2026, "percent"), store, TOL)
    assert not verify_citation(Citation("d2", text, 0.62, 2025, "percent"), store, TOL)


def test_verify_rejects_mismatched_pair_in_real_template_quote():
    store, text = _store("share")
    assert not verify_citation(Citation("d1", text, 0.550, 2026, "share"), store, TOL)
    assert not verify_citation(Citation("d1", text, 0.618, 2023, "share"), store, TOL)


def test_verify_tolerance_boundary():
    store, c = _cite("share", value=0.623)
    assert verify_citation(c, store, TOL)
    assert not verify_citation(dataclasses.replace(c, value=0.618 + TOL + 1e-6), store, TOL)


def _correction(**kw):
    store, c = _cite("share")
    base = Correction("Spain", "tertiary", 2026, 0.5, 0.618, c, 0.9, "r")
    return store, dataclasses.replace(base, **kw)


def test_supports_correction_accepts_tied_citation():
    store, corr = _correction()
    assert citation_supports_correction(corr, store, TOL)


def test_supports_correction_rejects_year_value_country_mismatch():
    store, corr = _correction(year=2025)
    assert not citation_supports_correction(corr, store, TOL)
    store, corr = _correction(new_value=0.7)
    assert not citation_supports_correction(corr, store, TOL)
    store, corr = _correction(country="France")
    assert not citation_supports_correction(corr, store, TOL)


def test_supports_correction_boundary_and_garbage():
    store, corr = _correction(new_value=0.618 + TOL)
    assert citation_supports_correction(corr, store, TOL)
    store, corr = _correction(new_value=0.618 + TOL + 1e-6)
    assert not citation_supports_correction(corr, store, TOL)
    assert citation_supports_correction(None, store, TOL) is False
