import inspect
import json
import re
import typing

import numpy as np
import pandas as pd
import pytest

from quotas_forecasting.config.config_loader import load_config
from quotas_forecasting.correction.constraints import check_constraints
from quotas_forecasting.correction.sources import SyntheticSourceStore
from quotas_forecasting.correction.tools import CorrectionSession, Rejection, tool_schemas
from quotas_forecasting.synthetic.benchmark import build_benchmark
from quotas_forecasting.synthetic.sources import SourceDoc, format_value

LAST_YEAR = 2024
SENTENCE = "According to the Office, the share of internet users with secondary education in {country} reached {value} in {year}."


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def bench():
    return build_benchmark(0, "education", n_countries=3)


def _table():
    rows = []
    for country, values in [
        ("Spain", {2023: 0.6, 2024: 0.6, 2025: 0.6, 2026: 0.6}),
        ("Italy", {2023: 0.5, 2024: 0.5, 2025: 0.5, 2026: np.nan}),
        ("Portugal", {2023: 0.4, 2024: 0.4, 2025: 0.6, 2026: 0.6}),  # baseline already breaks the yoy cap in 2025
    ]:
        for year, sec in values.items():
            rows += [(country, "secondary", year, sec), (country, "tertiary", year, 1 - sec)]
    table = pd.DataFrame(rows, columns=["country", "segment", "year", "value"])
    table["is_forecast"] = table["year"] > LAST_YEAR
    return table


def _doc(doc_id, country, year, value):
    text = SENTENCE.format(country=country, value=f"{value:.3f}", year=year) + " For reference, it was 0.600 in 2024."
    return SourceDoc(doc_id, "Office", country, year, "share", "secondary", text, value, title=f"{country} indicators {year}")


DOCS = [
    _doc("es-26", "Spain", 2026, 0.65),
    _doc("es-26-high", "Spain", 2026, 0.75),
    _doc("es-26-oob", "Spain", 2026, 1.2),
    _doc("es-24", "Spain", 2024, 0.62),
    _doc("it-26", "Italy", 2026, 0.55),
]


def _citation(doc_id):
    doc = next(d for d in DOCS if d.doc_id == doc_id)
    return {"doc_id": doc_id, "quoted_text": doc.text.split(" For reference")[0], "value": doc.value, "year": doc.year, "unit": "share"}


@pytest.fixture
def session(config):
    return CorrectionSession(_table(), "education", SyntheticSourceStore(DOCS), config)


def _propose(session, doc_id, country="Spain", segment="secondary", **overrides):
    citation = _citation(doc_id)
    args = {"country": country, "segment": segment, "year": citation["year"], "new_value": citation["value"], "citation": citation}
    args.update({"confidence": 0.9, "reason": "published figure"}, **overrides)
    return session.propose_correction(**args)


def _value(table, country, segment, year):
    return table.loc[(table["country"] == country) & (table["segment"] == segment) & (table["year"] == year), "value"].item()


def _assert_rejected(session, result, *fragments):
    assert result["accepted"] is False
    assert all(fragment in result["reason"] for fragment in fragments), result["reason"]
    pd.testing.assert_frame_equal(session.working, _table())
    assert session.corrections == [] and len(session.rejections) == 1
    assert isinstance(session.rejections[0], Rejection) and session.rejections[0].reason == result["reason"]
    json.dumps(result)


def test_accepts_valid_cited_correction_and_derives_tertiary(session):
    result = _propose(session, "es-26")
    assert result["accepted"] is True, result["reason"]
    assert _value(session.working, "Spain", "secondary", 2026) == 0.65
    assert _value(session.working, "Spain", "tertiary", 2026) == pytest.approx(0.35, abs=1e-12)
    assert len(session.corrections) == 1 and session.rejections == []
    correction = session.corrections[0]
    cell = (correction.country, correction.segment, correction.year, correction.old_value, correction.new_value)
    assert cell == ("Spain", "secondary", 2026, 0.6, 0.65)
    assert correction.citation.doc_id == "es-26"
    json.dumps(result)


def test_pre_existing_baseline_violation_does_not_block_proposals(session, config):
    assert check_constraints(session.baseline, session.baseline, "education", config["correction"]["max_yoy_change"])
    assert _propose(session, "es-26")["accepted"] is True


def test_old_value_is_the_working_value_at_proposal_time(session):
    assert _propose(session, "es-26")["accepted"]
    assert _propose(session, "es-26")["accepted"]
    assert [c.old_value for c in session.corrections] == [0.6, 0.65]


def test_rejects_hallucinated_citation_and_leaves_table_unchanged(session):
    citation = _citation("es-26") | {"quoted_text": "the share reached 0.650 in 2026 in Spain"}
    _assert_rejected(session, _propose(session, "es-26", citation=citation), "citation")


def test_rejects_citation_not_tied_to_the_proposed_value(session):
    _assert_rejected(session, _propose(session, "es-26", new_value=0.62), "citation")


def test_rejects_out_of_bounds_value(session):
    _assert_rejected(session, _propose(session, "es-26-oob"), "[0, 1]")


def test_rejects_historical_cell(session):
    _assert_rejected(session, _propose(session, "es-24", year=2024), "historical")


def test_rejects_nan_cell(session):
    _assert_rejected(session, _propose(session, "it-26", country="Italy"), "NaN")


def test_rejects_unknown_cell(session):
    _assert_rejected(session, _propose(session, "es-26", country="Atlantis"), "no forecast cell")


def test_rejects_yoy_above_cap(session):
    _assert_rejected(session, _propose(session, "es-26-high"), "yoy_cap", "Spain", "secondary", "2026")


def test_rejects_direct_tertiary_proposal(session):
    _assert_rejected(session, _propose(session, "es-26", segment="tertiary", new_value=0.35), "tertiary")


@pytest.mark.parametrize(
    "overrides",
    [
        {"year": "2026"},
        {"year": True},
        {"new_value": "0.65"},
        {"new_value": float("nan")},
        {"citation": "es-26"},
        {"citation": {"doc_id": "es-26"}},
        {"confidence": None},
        {"reason": 5},
        {"country": None},
    ],
)
def test_bad_input_types_are_rejected_not_raised(session, overrides):
    _assert_rejected(session, _propose(session, "es-26", **overrides))


def test_search_returns_only_snippets_never_ground_truth_fields(bench, config):
    session = CorrectionSession(bench.baseline, "education", SyntheticSourceStore(bench.docs), config)
    doc = bench.docs[0]
    result = session.search_sources("internet", country=doc.country)
    json.dumps(result)
    assert set(result) == {"results", "truncated", "total"}
    assert result["results"] and all(set(r) == {"doc_id", "publisher", "country", "year", "title"} for r in result["results"])
    assert all(r["country"] == doc.country for r in result["results"])
    assert result["total"] >= len(result["results"])
    assert session.read_source(doc.doc_id) == doc.text


def test_search_reports_truncation(config):
    docs = [_doc(f"d{i:02d}", "Spain", 2026, 0.65) for i in range(25)]
    session = CorrectionSession(_table(), "education", SyntheticSourceStore(docs), config)
    result = session.search_sources("internet Spain")
    assert (len(result["results"]), result["truncated"], result["total"]) == (20, True, 25)
    narrow = CorrectionSession(_table(), "education", SyntheticSourceStore(docs[:3]), config).search_sources("internet")
    assert (len(narrow["results"]), narrow["truncated"], narrow["total"]) == (3, False, 3)


def test_benchmark_accepted_corrections_never_introduce_violations(bench, config):
    store = SyntheticSourceStore(bench.docs)
    session = CorrectionSession(bench.baseline, "education", store, config)
    max_yoy = config["correction"]["max_yoy_change"]
    before = {(v.kind, v.country, v.segment, v.year) for v in check_constraints(bench.baseline, bench.baseline, "education", max_yoy)}
    for doc in (d for d in bench.docs if d.segment == "secondary"):
        stated = format_value(doc.value, doc.unit)
        quote = next(s for s in re.split(r"(?<=\.)\s+(?=[A-Z])", doc.text) if stated in s and str(doc.year) in s)
        citation = {"doc_id": doc.doc_id, "quoted_text": quote, "value": doc.value, "year": doc.year, "unit": doc.unit}
        session.propose_correction(doc.country, "secondary", doc.year, doc.value, citation, 0.8, "doc")
    assert session.corrections, [r.reason for r in session.rejections]
    after = {(v.kind, v.country, v.segment, v.year) for v in check_constraints(bench.baseline, session.working, "education", max_yoy)}
    assert after <= before
    for c in session.corrections:
        assert _value(session.working, c.country, "tertiary", c.year) == pytest.approx(1 - _value(session.working, c.country, "secondary", c.year))


def test_dispatch_routes_and_never_raises(session):
    assert "results" in session.dispatch("search_sources", {"query": "internet", "country": "Spain"})
    assert session.dispatch("read_source", {"doc_id": "es-26"}) == {"text": DOCS[0].text}
    citation = _citation("es-26")
    args = {"country": "Spain", "segment": "secondary", "year": 2026, "new_value": 0.65, "citation": citation, "confidence": 0.9, "reason": "x"}
    assert session.dispatch("propose_correction", args)["accepted"] is True
    for name, arguments in [
        ("delete_table", {}),
        ("read_source", {}),
        ("read_source", {"doc_id": "missing"}),
        ("read_source", {"doc_id": 3}),
        ("read_source", {"doc_id": "es-26", "extra": 1}),
        ("search_sources", {"query": 7}),
        ("search_sources", {"query": "x", "year": "2026"}),
        ("propose_correction", {"country": "Spain"}),
        ("search_sources", "not a dict"),
    ]:
        result = session.dispatch(name, arguments)
        assert set(result) == {"error"} and result["error"], (name, arguments)


_JSON_TYPES = {str: "string", int: "integer", float: "number", dict: "object"}


def test_tool_schemas_match_method_signatures():
    schemas = tool_schemas()
    assert [s["function"]["name"] for s in schemas] == ["search_sources", "read_source", "propose_correction"]
    for schema in schemas:
        assert schema["type"] == "function" and schema["function"]["description"]
        method = getattr(CorrectionSession, schema["function"]["name"])
        params = {n: p for n, p in inspect.signature(method).parameters.items() if n != "self"}
        hints = typing.get_type_hints(method)
        parameters = schema["function"]["parameters"]
        assert parameters["type"] == "object" and set(parameters["properties"]) == set(params)
        assert parameters["required"] == [n for n, p in params.items() if p.default is inspect.Parameter.empty]
        for name in params:
            base = [t for t in typing.get_args(hints[name]) if t is not type(None)] or [hints[name]]
            assert parameters["properties"][name]["type"] == _JSON_TYPES[base[0]], name
    citation = schemas[2]["function"]["parameters"]["properties"]["citation"]
    assert set(citation["required"]) == {"doc_id", "quoted_text", "value", "year", "unit"} == set(citation["properties"])
    json.dumps(schemas)
