import copy
import json
import logging

import numpy as np
import pandas as pd
import pytest

from quotas_forecasting.config.config_loader import load_config
from quotas_forecasting.correction.agent import MAX_SUMMARY_ROWS, CorrectionResult, run_correction, summarize_table
from quotas_forecasting.correction.evaluation import evaluate
from quotas_forecasting.correction.llm import LLMResponse, LLMUnavailableError, ScriptedClient, ToolCall
from quotas_forecasting.correction.prompts import planner_system_prompt, researcher_system_prompt
from quotas_forecasting.correction.sources import SyntheticSourceStore
from quotas_forecasting.synthetic.benchmark import build_benchmark
from tests.correction.scripting import KEY
from tests.correction.scripting import call as _call
from tests.correction.scripting import citation as _citation
from tests.correction.scripting import doc_for as _doc_for
from tests.correction.scripting import plan as _plan
from tests.correction.scripting import propose as _propose
from tests.correction.scripting import research_script as _research_script


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def edu():
    # seed 0, 8 countries: France is shocked and has documents, Spain is shocked and has none, Czech Republic secondary 2025 has two
    # contradicting documents (src-0007 wrong, src-0017 right).
    return build_benchmark(0, "education")


@pytest.fixture(scope="module")
def age():
    return build_benchmark(0, "age_gender")


def _cells(table, country):
    return table[table["country"] == country].reset_index(drop=True)


def _tool_messages(call):
    return [m for m in call["messages"] if m["role"] == "tool"]


# ---- tests ----


def test_corrects_one_shocked_country_end_to_end(edu, config):
    store = SyntheticSourceStore(edu.docs)
    llm = ScriptedClient([_plan(("France", "secondary"))] + _research_script(edu, store, "France", "secondary"))
    result = run_correction(edu.baseline, "education", store, llm, config)

    assert isinstance(result, CorrectionResult) and result.error is None
    assert [(c.country, c.segment, c.year) for c in result.corrections] == [("France", "secondary", 2025), ("France", "secondary", 2026)]
    report = evaluate(edu, result.corrected, result.corrections, store, config)
    assert report.mae_after < report.mae_before
    assert (report.hallucinated_citations, report.violations, report.n_corrections) == (0, 0, 2)
    others = result.corrected["country"] != "France"
    pd.testing.assert_frame_equal(result.corrected[others], edu.baseline[others])
    assert result.log and all(isinstance(entry, dict) for entry in result.log)


def test_country_without_sources_is_left_unchanged(edu, config):
    store = SyntheticSourceStore(edu.docs)
    assert store.search("secondary Spain", country="Spain") == []
    script = [
        _plan(("Spain", "secondary")),
        _call("s1", "search_sources", query="secondary education Spain", country="Spain"),
        LLMResponse(text="No sources for Spain; leaving the value unchanged."),
    ]
    llm = ScriptedClient(script)
    result = run_correction(edu.baseline, "education", store, llm, config)

    assert result.error is None and result.corrections == [] and result.rejections == []
    pd.testing.assert_frame_equal(result.corrected, edu.baseline)
    assert len(llm.calls) == 3
    assert json.loads(_tool_messages(llm.calls[2])[-1]["content"])["total"] == 0


def test_contradicting_sources_only_verified_citation_is_committed(edu, config):
    store = SyntheticSourceStore(edu.docs)
    wrong, right = "src-0007", "src-0017"  # both about Czech Republic secondary 2025; they disagree
    fabricated = _citation(store, wrong, "Czech Republic") | {"quoted_text": "secondary education in Czech Republic was 0.540 in 2025"}
    script = [
        _plan(("Czech Republic", "secondary")),
        _call("s1", "search_sources", query="secondary education", country="Czech Republic", year=2025),
        LLMResponse(text=None, tool_calls=[ToolCall("r1", "read_source", {"doc_id": wrong}), ToolCall("r2", "read_source", {"doc_id": right})]),
        _propose("p1", "Czech Republic", "secondary", fabricated),
        _propose("p2", "Czech Republic", "secondary", _citation(store, right, "Czech Republic")),
        LLMResponse(text="Committed the corroborated figure."),
    ]
    result = run_correction(edu.baseline, "education", store, ScriptedClient(script), config)

    assert result.error is None
    assert len(result.rejections) == 1 and "citation" in result.rejections[0].reason
    assert len(result.corrections) == 1
    correction = result.corrections[0]
    assert (correction.citation.doc_id, correction.year, correction.new_value) == (right, 2025, pytest.approx(0.439))
    assert evaluate(edu, result.corrected, result.corrections, store, config).hallucinated_citations == 0


def test_rejected_proposal_is_fed_back_to_model(edu, config):
    store = SyntheticSourceStore(edu.docs)
    citation = _citation(store, _doc_for(edu, "France", "secondary", 2025).doc_id, "France")
    script = [
        _plan(("France", "secondary")),
        _propose("p1", "France", "secondary", citation, new_value=0.9),  # not what the quote says
        LLMResponse(text="Giving up."),
    ]
    llm = ScriptedClient(script)
    result = run_correction(edu.baseline, "education", store, llm, config)

    reason = result.rejections[0].reason
    feedback = _tool_messages(llm.calls[2])[-1]
    assert feedback["tool_call_id"] == "p1"
    assert json.loads(feedback["content"]) == {"accepted": False, "reason": reason}
    assistant = [m for m in llm.calls[2]["messages"] if m["role"] == "assistant"][-1]
    assert assistant["tool_calls"][0]["id"] == "p1" and assistant["tool_calls"][0]["function"]["name"] == "propose_correction"


def test_iteration_budget_stops_the_loop(edu, config):
    store = SyntheticSourceStore(edu.docs)
    budget = config["correction"]["max_iterations_per_segment"]
    citation = _citation(store, _doc_for(edu, "France", "secondary", 2025).doc_id, "France")
    invalid = [_propose(f"p{i}", "France", "secondary", citation, new_value=1.5) for i in range(budget)]
    llm = ScriptedClient([_plan(("France", "secondary"))] + invalid)  # one more call would raise "script exhausted"
    result = run_correction(edu.baseline, "education", store, llm, config)

    assert len(llm.calls) == 1 + budget
    assert len(result.rejections) == budget and result.corrections == []
    assert result.error is None
    pd.testing.assert_frame_equal(result.corrected, edu.baseline)
    assert any(entry.get("event") == "budget_exhausted" for entry in result.log)


def test_iteration_budget_is_read_from_config(edu, config):
    store = SyntheticSourceStore(edu.docs)
    small = copy.deepcopy(config)
    small["correction"]["max_iterations_per_segment"] = 2
    looping = [_call(f"s{i}", "search_sources", query="secondary") for i in range(2)]
    llm = ScriptedClient([_plan(("France", "secondary"), ("Poland", "secondary"))] + looping + looping)
    run_correction(edu.baseline, "education", store, llm, small)
    assert len(llm.calls) == 1 + 2 + 2


def test_unknown_tool_and_malformed_arguments_do_not_crash(edu, config):
    store = SyntheticSourceStore(edu.docs)
    script = [
        _plan(("France", "secondary")),
        LLMResponse(
            text=None,
            tool_calls=[
                ToolCall("u1", "delete_table", {"really": True}),
                ToolCall("m1", "propose_correction", {}, parse_error="invalid tool arguments: JSONDecodeError"),
                ToolCall("m2", "propose_correction", {"country": "France"}),
                ToolCall("m3", "read_source", {"doc_id": 42}),
                ToolCall("m4", "read_source", {"doc_id": "no-such-doc"}),
            ],
        ),
        LLMResponse(text=None, tool_calls=[]),  # neither text nor tool calls
    ]
    llm = ScriptedClient(script)
    result = run_correction(edu.baseline, "education", store, llm, config)

    assert result.error is None and result.corrections == []
    pd.testing.assert_frame_equal(result.corrected, edu.baseline)
    feedback = {m["tool_call_id"]: json.loads(m["content"]) for m in _tool_messages(llm.calls[2])}
    assert set(feedback) == {"u1", "m1", "m2", "m3", "m4"}
    assert all("error" in content for content in feedback.values())
    assert "JSONDecodeError" in feedback["m1"]["error"]
    assert "unknown tool" in feedback["u1"]["error"]


class _FailingClient(ScriptedClient):
    """Replays a script and raises ``exc`` on call number ``fail_at`` (1-based)."""

    def __init__(self, responses, fail_at, exc):
        super().__init__(responses)
        self.fail_at, self.exc, self.n = fail_at, exc, 0

    def chat(self, messages, tools):
        self.n += 1
        if self.n == self.fail_at:
            raise self.exc
        return super().chat(messages, tools)


def _one_accepted_then(edu, store):
    citation = _citation(store, _doc_for(edu, "France", "secondary", 2025).doc_id, "France")
    return [_plan(("France", "secondary")), _propose("p1", "France", "secondary", citation)]


def test_llm_unavailable_returns_baseline_with_error(edu, config):
    store = SyntheticSourceStore(edu.docs)
    llm = _FailingClient([_plan(("France", "secondary"))], fail_at=2, exc=LLMUnavailableError("LLM request failed: APIConnectionError"))
    result = run_correction(edu.baseline, "education", store, llm, config)
    assert "LLMUnavailableError" in result.error
    pd.testing.assert_frame_equal(result.corrected, edu.baseline)

    # a failure later in the run keeps what was already accepted
    llm = _FailingClient(_one_accepted_then(edu, store), fail_at=3, exc=LLMUnavailableError("LLM request failed: RateLimitError (HTTP 429)"))
    result = run_correction(edu.baseline, "education", store, llm, config)
    assert "LLMUnavailableError" in result.error
    assert [(c.country, c.year) for c in result.corrections] == [("France", 2025)]
    changed = (result.corrected["value"] != edu.baseline["value"]) & edu.baseline["value"].notna()
    assert set(result.corrected.loc[changed, "segment"]) == {"secondary", "tertiary"} and changed.sum() == 2


def test_any_client_exception_is_caught_but_script_bugs_are_not(edu, config):
    store = SyntheticSourceStore(edu.docs)
    llm = _FailingClient(_one_accepted_then(edu, store), fail_at=3, exc=IndexError("list index out of range" + "x" * 500))
    result = run_correction(edu.baseline, "education", store, llm, config)
    assert result.error.startswith("IndexError") and len(result.error) <= 220
    assert len(result.corrections) == 1

    with pytest.raises(AssertionError, match="script exhausted"):
        run_correction(edu.baseline, "education", store, ScriptedClient(_one_accepted_then(edu, store)), config)


def test_malformed_plan_is_retried_then_baseline_kept(edu, config):
    store = SyntheticSourceStore(edu.docs)
    attempts = 1 + config["correction"]["max_llm_retries"]
    llm = ScriptedClient([LLMResponse(text="I think France looks odd")] * attempts)
    result = run_correction(edu.baseline, "education", store, llm, config)
    assert len(llm.calls) == attempts
    assert result.error and "plan" in result.error
    pd.testing.assert_frame_equal(result.corrected, edu.baseline)
    # the parse failure is fed back before each retry
    assert "JSON" in llm.calls[1]["messages"][-1]["content"]


def test_plan_recovers_after_one_malformed_reply_and_ignores_unknown_cells(edu, config):
    store = SyntheticSourceStore(edu.docs)
    items = [{"country": "Atlantis", "segment": "secondary"}, {"country": "France"}, {"country": ["France"], "segment": "secondary"}, 7]
    plan = LLMResponse(text="Here: " + json.dumps({"suspects": items}))
    llm = ScriptedClient([LLMResponse(text="{not json"), plan])
    result = run_correction(edu.baseline, "education", store, llm, config)
    assert result.error is None and len(llm.calls) == 2
    pd.testing.assert_frame_equal(result.corrected, edu.baseline)


def test_nan_cells_stay_nan(edu, config):
    store = SyntheticSourceStore(edu.docs)
    baseline = edu.baseline.copy()
    nan_cell = (baseline["country"] == "France") & (baseline["year"] == 2026)  # missing forecast for both segments
    baseline.loc[nan_cell, "value"] = np.nan
    cit25 = _citation(store, _doc_for(edu, "France", "secondary", 2025).doc_id, "France")
    cit26 = _citation(store, _doc_for(edu, "France", "secondary", 2026).doc_id, "France")
    script = [
        _plan(("France", "secondary")),
        _propose("p1", "France", "secondary", cit26),  # NaN cell: rejected, never filled
        _propose("p2", "France", "secondary", cit25),
        LLMResponse(text="done"),
    ]
    result = run_correction(baseline, "education", store, ScriptedClient(script), config)

    assert result.corrected["value"].isna().tolist() == baseline["value"].isna().tolist()
    assert baseline["value"].isna().sum() >= 4  # the historical Czech Republic 2004 gaps plus the two forecast cells
    assert "NaN" in result.rejections[0].reason
    assert [c.year for c in result.corrections] == [2025]


def test_age_gender_module_runs(age, config):
    store = SyntheticSourceStore(age.docs)
    segment = "Male, 16 to 24"
    llm = ScriptedClient([_plan(("France", segment))] + _research_script(age, store, "France", segment))
    result = run_correction(age.baseline, "age_gender", store, llm, config)

    assert result.error is None and len(result.corrections) == 2
    report = evaluate(age, result.corrected, result.corrections, store, config)
    assert report.mae_after < report.mae_before
    assert (report.hallucinated_citations, report.violations) == (0, 0)


def test_prompts_state_constraints_and_interpolate_config(edu, config):
    for prompt in (planner_system_prompt(0.07), researcher_system_prompt(0.07, 4)):
        lowered = prompt.lower()
        assert "0.07" in prompt and "[0, 1]" in prompt
        assert "tertiary" in lowered and "derived" in lowered
        assert "historical" in lowered and "nan" in lowered
        assert "unchanged" in lowered
        assert "verbatim" in lowered and "citation" in lowered
        assert "corroborat" in lowered and "average" in lowered
        assert "outside knowledge" in lowered
    assert "4" in researcher_system_prompt(0.07, 4)

    small = copy.deepcopy(config)
    small["correction"]["max_yoy_change"] = 0.07
    llm = ScriptedClient([_plan()])
    run_correction(edu.baseline, "education", SyntheticSourceStore(edu.docs), llm, small)
    assert "0.07" in llm.calls[0]["messages"][0]["content"]


def test_planner_summary_is_compact_capped_and_has_no_ground_truth(edu, config):
    store = SyntheticSourceStore(edu.docs)
    llm = ScriptedClient([_plan()])
    run_correction(edu.baseline, "education", store, llm, config)
    summary = llm.calls[0]["messages"][1]["content"]
    assert "France" in summary and "secondary" in summary
    assert not any(d.text[:40] in summary or d.doc_id in summary for d in edu.docs)
    assert f"{edu.truth.set_index(KEY)['value'][('France', 'secondary', 2025)]:.3f}" not in summary

    rows = [
        (f"C{c:02d}", f"S{s:02d}", year, 0.5 + 0.001 * c + (0.05 * (c == 7 and s == 3) if year > 2024 else 0.0), year > 2024)
        for c in range(30)
        for s in range(12)
        for year in range(2020, 2027)
    ]
    big = pd.DataFrame(rows, columns=["country", "segment", "year", "value", "is_forecast"])
    text = summarize_table(big, "age_gender", MAX_SUMMARY_ROWS)
    lines = [line for line in text.splitlines() if line.startswith("C")]
    assert len(lines) == MAX_SUMMARY_ROWS
    assert lines[0].startswith("C07 | S03")  # largest forecast-vs-trend deviation ranks first
    assert str(30 * 12 - MAX_SUMMARY_ROWS) in text  # omitted rows are reported
    assert len(text) < 12_000


def test_run_is_deterministic(edu, config):
    store = SyntheticSourceStore(edu.docs)
    script = [_plan(("France", "secondary"))] + _research_script(edu, store, "France", "secondary")
    results = [run_correction(edu.baseline, "education", store, ScriptedClient(script), config) for _ in range(2)]
    pd.testing.assert_frame_equal(results[0].corrected, results[1].corrected)
    assert results[0].log == results[1].log


# ---- hardening: suspect cap, plan JSON extraction, source-count lookup ----


def test_suspects_beyond_max_suspects_are_ignored_not_researched(edu, config):
    store = SyntheticSourceStore(edu.docs)
    capped = copy.deepcopy(config)
    capped["correction"]["max_suspects"] = 2
    items = [
        {"country": "France", "segment": "secondary"},
        {"country": "Atlantis", "segment": "secondary"},  # invalid: ignored, does not count against the cap
        {"country": "France", "segment": "tertiary"},  # duplicate of France/secondary after the tertiary mapping
        {"country": "Poland", "segment": "secondary"},
        {"country": "Spain", "segment": "secondary"},  # over the cap
    ]
    done = LLMResponse(text="nothing to change")
    llm = ScriptedClient([LLMResponse(text=json.dumps({"suspects": items})), done, done])  # a third research call would exhaust the script
    result = run_correction(edu.baseline, "education", store, llm, capped)

    assert result.error is None and len(llm.calls) == 1 + 2
    plan = next(entry for entry in result.log if entry["event"] == "plan")
    assert plan["suspects"] == [["France", "secondary"], ["Poland", "secondary"]]
    assert plan["ignored"] == [items[1], items[4]]
    assert [entry["country"] for entry in result.log if entry["event"] == "done"] == ["France", "Poland"]


def test_config_caps_suspects_at_25(config):
    assert config["correction"]["max_suspects"] == 25
    assert config["correction"]["on_failure"] == "fallback_to_baseline"


def test_plan_followed_by_prose_with_braces_parses(edu, config):
    llm = ScriptedClient([LLMResponse(text='{"suspects": []} (see {note})')])
    result = run_correction(edu.baseline, "education", SyntheticSourceStore(edu.docs), llm, config)
    assert result.error is None and len(llm.calls) == 1
    assert any(entry["event"] == "plan" for entry in result.log)


def test_trailing_brace_or_prose_only_reply_is_retried_within_bound(edu, config):
    attempts = 1 + config["correction"]["max_llm_retries"]
    bad = [LLMResponse(text="Plan follows: {"), LLMResponse(text="France looks odd")] * attempts
    llm = ScriptedClient(bad[:attempts])
    result = run_correction(edu.baseline, "education", SyntheticSourceStore(edu.docs), llm, config)
    assert len(llm.calls) == attempts and result.error and "plan" in result.error
    assert [entry["event"] for entry in result.log].count("plan_invalid") == attempts


class _NoCountsStore(SyntheticSourceStore):
    def search_with_total(self, query, country=None, year=None):
        raise RuntimeError("index unavailable")


def test_source_count_failure_drops_counts_line_and_continues(edu, config, caplog):
    store = _NoCountsStore(edu.docs)
    llm = ScriptedClient([_plan(("France", "secondary")), LLMResponse(text="nothing to change")])
    with caplog.at_level(logging.WARNING, logger="quotas_forecasting.correction.agent"):
        result = run_correction(edu.baseline, "education", store, llm, config)

    assert result.error is None and len(llm.calls) == 2
    summary = llm.calls[0]["messages"][1]["content"]
    assert "France | secondary" in summary and "Source documents per country" not in summary
    assert any(r.levelno == logging.WARNING and "source counts" in r.getMessage() for r in caplog.records)
