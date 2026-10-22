"""Script helpers (tests only): compose ScriptedClient replies with real, verifying quotes from the documents' text as read through the store."""

import json
import re

from quotas_forecasting.correction.llm import LLMResponse, ToolCall

KEY = ["country", "segment", "year"]


def doc_for(bench, country, segment, year):
    """The document closest to truth for a key (the test oracle may look at truth; the agent never does)."""
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


def research_script(bench, store, country, segment):
    """search -> read every forecast-year doc -> propose each forecast year in order -> stop."""
    years = sorted(bench.baseline.loc[bench.baseline["is_forecast"], "year"].unique())
    docs = [doc_for(bench, country, segment, int(y)) for y in years]
    script = [call("s1", "search_sources", query=f"{segment} {country}", country=country)]
    script.append(LLMResponse(text=None, tool_calls=[ToolCall(f"r{i}", "read_source", {"doc_id": d.doc_id}) for i, d in enumerate(docs)]))
    script += [propose(f"p{i}", country, segment, citation(store, d.doc_id, country)) for i, d in enumerate(docs)]
    return script + [LLMResponse(text="Done: corrections proposed.")]
