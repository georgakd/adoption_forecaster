"""Oracle script for the stub benchmark mode: a ScriptedClient that stands in for a perfect model.

The oracle may look at ground truth (the real agent never does): it names the shocked country-segments that have a verifiable
document for every forecast year, then researches each with the document closest to truth, quoting its sentence verbatim.
"""

import json
import re

from adoption_forecaster.correction.agent import _editable_segments
from adoption_forecaster.correction.llm import LLMResponse, ScriptedClient, ToolCall
from adoption_forecaster.correction.sources import SourceStore
from adoption_forecaster.synthetic.benchmark import Benchmark

KEY = ["country", "segment", "year"]
SHOCK_THRESHOLD = 0.01  # a baseline cell this far from truth marks a suspect
AGREE_TOLERANCE = 0.01  # a document "agrees with the truth" when its stated value is this close and closer than the baseline


def doc_for(bench, country, segment, year):
    """The document closest to truth for a key."""
    truth = bench.truth.set_index(KEY)["value"][(country, segment, year)]
    docs = [d for d in bench.docs if (d.country, d.segment, d.year) == (country, segment, year)]
    return min(docs, key=lambda d: abs(d.value - truth))


def citation(store, doc_id, country):
    """Citation dict for the sentence of the document that names ``country`` (the target sentence), parsed from the text itself."""
    sentence = next(s for s in re.split(r"(?<=\.)\s+", store.read(doc_id)) if country in s)
    year = int(re.search(r"\b(?:19|20)\d{2}\b", sentence).group())
    number = re.search(r"(\d+\.\d)%|([01]\.\d{3})", sentence)
    value, unit = (round(float(number.group(1)) / 100, 6), "percent") if number.group(1) else (float(number.group(2)), "share")
    return {"doc_id": doc_id, "quoted_text": sentence, "value": value, "year": year, "unit": unit}


def plan(*suspects):
    return LLMResponse(text=json.dumps({"suspects": [{"country": c, "segment": s} for c, s in suspects]}))


def call(call_id, name, **arguments):
    return LLMResponse(text=None, tool_calls=[ToolCall(call_id, name, arguments)])


def propose(call_id, country, segment, cite, new_value=None):
    args = {"country": country, "segment": segment, "year": cite["year"], "new_value": cite["value"] if new_value is None else new_value}
    return call(call_id, "propose_correction", **args, citation=cite, confidence=0.8, reason="published figure")


def _forecast_years(bench: Benchmark, country: str, segment: str) -> list[int]:
    cells = bench.baseline[(bench.baseline["country"] == country) & (bench.baseline["segment"] == segment)]
    return sorted(int(y) for y in cells.loc[cells["is_forecast"] & cells["value"].notna(), "year"])


def research_script(bench, store, country, segment, fabricate=False):
    """search -> read every forecast-year doc -> propose each forecast year in order -> stop.

    ``fabricate`` (tests only) replaces each quote with text that is not in the document, to exercise the failing gate.
    """
    docs = [doc_for(bench, country, segment, y) for y in _forecast_years(bench, country, segment)]
    script = [call("s1", "search_sources", query=f"{segment} {country}", country=country)]
    script.append(LLMResponse(text=None, tool_calls=[ToolCall(f"r{i}", "read_source", {"doc_id": d.doc_id}) for i, d in enumerate(docs)]))
    for i, d in enumerate(docs):
        cite = citation(store, d.doc_id, country)
        if fabricate:
            cite = {**cite, "quoted_text": "An invented sentence that no document contains."}
        script.append(propose(f"p{i}", country, segment, cite))
    return script + [LLMResponse(text="Done: corrections proposed.")]


def oracle_suspects(bench: Benchmark, max_suspects: int) -> list[tuple[str, str]]:
    """Shocked country-segments (largest error first) whose every forecast year has a document agreeing with the truth."""
    truth = bench.truth.set_index(KEY)["value"]
    base = bench.baseline.set_index(KEY)["value"]
    found = []
    for country, segment in _editable_segments(bench.baseline, bench.module):
        years = _forecast_years(bench, country, segment)
        errors = [abs(base[(country, segment, y)] - truth[(country, segment, y)]) for y in years]
        if not years or max(errors) <= SHOCK_THRESHOLD:
            continue
        try:
            docs = [doc_for(bench, country, segment, y) for y in years]
        except ValueError:  # a forecast year without any document
            continue
        if all(
            abs(d.value - truth[(country, segment, y)]) <= min(AGREE_TOLERANCE, e) for d, y, e in zip(docs, years, errors, strict=True)
        ):
            found.append((-max(errors), country, segment))
    return [(c, s) for _, c, s in sorted(found)[:max_suspects]]


def build_oracle_client(bench: Benchmark, store: SourceStore, max_suspects: int = 25, fabricate: bool = False) -> ScriptedClient:
    suspects = oracle_suspects(bench, max_suspects)
    script = [plan(*suspects)]
    for country, segment in suspects:
        script += research_script(bench, store, country, segment, fabricate)
    return ScriptedClient(script)
