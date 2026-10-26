"""The correction agent: one LLM, a planning call, then a bounded researcher tool-calling loop per suspicious country-segment.

Every proposal goes through ``CorrectionSession``'s deterministic gate; rejections are fed back to the model as the tool result.
The run never raises on LLM failure or malformed model output: it returns the best table so far with ``error`` set.

Message flow:
  planner:    [system, user(summary)] -> JSON {"suspects": [...]}; an unparseable reply is fed back and retried up to max_llm_retries times.
  researcher: per suspect, [system, user(segment values)] -> up to max_iterations_per_segment turns; each turn appends the assistant
              message (with its tool calls) and one tool message per call; a reply without tool calls ends the segment.
"""

import json
import logging
import math
from dataclasses import dataclass, field

from quotas_forecasting.correction.llm import LLMClient
from quotas_forecasting.correction.prompts import planner_system_prompt, researcher_system_prompt
from quotas_forecasting.correction.sources import SourceStore
from quotas_forecasting.correction.tools import CorrectionSession, Rejection, tool_schemas
from quotas_forecasting.correction.types import Correction, ForecastTable

# The planner sees at most this many country-segment rows (~100 characters each), ranked by forecast-vs-trend deviation, so a
# 30-country x 12-segment table (360 rows) stays around 6k characters; the number of omitted rows is stated in the summary.
MAX_SUMMARY_ROWS = 60
HISTORY_SHOWN = 3  # last historical values per row; the trend is the slope over these
ERROR_MAX_CHARS = 200

logger = logging.getLogger(__name__)


@dataclass
class CorrectionResult:
    corrected: ForecastTable
    corrections: list[Correction]
    rejections: list[Rejection]
    log: list[dict] = field(default_factory=list)
    error: str | None = None


def _fmt(values) -> str:
    return " ".join(f"{int(y)}={'NA' if math.isnan(v) else f'{v:.3f}'}" for y, v in values)


def _editable_segments(table: ForecastTable, module: str) -> list[tuple[str, str]]:
    """(country, segment) pairs with at least one non-NaN forecast cell; education tertiary is derived, so it is not listed."""
    cells = table[table["is_forecast"] & table["value"].notna()]
    if module == "education":
        cells = cells[cells["segment"] != "tertiary"]
    return sorted(set(zip(cells["country"], cells["segment"])))


def _segment_rows(table: ForecastTable, country: str, segment: str):
    rows = table[(table["country"] == country) & (table["segment"] == segment)].sort_values("year")
    history = list(zip(rows.loc[~rows["is_forecast"], "year"], rows.loc[~rows["is_forecast"], "value"].astype(float)))[-HISTORY_SHOWN:]
    forecast = list(zip(rows.loc[rows["is_forecast"], "year"], rows.loc[rows["is_forecast"], "value"].astype(float)))
    return history, forecast


def _trend_deviation(history, forecast) -> float:
    """Largest |forecast - linear extrapolation of the shown history|; 0 when there is too little data."""
    known = [(y, v) for y, v in history if not math.isnan(v)]
    if not known:
        return 0.0
    (y0, v0), (y1, v1) = known[0], known[-1]
    slope = (v1 - v0) / (y1 - y0) if y1 != y0 else 0.0
    deviations = [abs(v - (v1 + slope * (y - y1))) for y, v in forecast if not math.isnan(v)]
    return max(deviations, default=0.0)


def summarize_table(
    table: ForecastTable,
    module: str,
    max_rows: int = MAX_SUMMARY_ROWS,
    source_counts: dict[str, int] | None = None,
    caps: dict[tuple[str, str], float] | None = None,
) -> str:
    """Compact planner view of the forecast table: one line per editable country-segment, largest trend deviation first, capped.
    With ``caps`` each line also states that country-segment's year-over-year cap."""
    entries = []
    for country, segment in _editable_segments(table, module):
        history, forecast = _segment_rows(table, country, segment)
        entries.append((-_trend_deviation(history, forecast), country, segment, history, forecast))
    entries.sort(key=lambda e: e[:3])
    lines = [
        f"Module: {module}. Row format: country | segment | last historical values | forecast values | max deviation from recent trend"
        + (" | year-over-year cap." if caps is not None else "."),
    ]
    if module == "education":
        lines.append("Only secondary is listed: tertiary = 1 - secondary.")
    for d, c, s, h, f in entries[:max_rows]:
        lines.append(f"{c} | {s} | {_fmt(h)} | {_fmt(f)} | dev={-d:.3f}" + ("" if caps is None else f" | cap={caps[(c, s)]:.3f}"))
    if len(entries) > max_rows:
        lines.append(f"{len(entries) - max_rows} more country-segments omitted (smaller deviation from trend).")
    if source_counts is not None:
        lines.append("Source documents per country: " + ", ".join(f"{c}={n}" for c, n in sorted(source_counts.items())))
    return "\n".join(lines)


def _segment_brief(table: ForecastTable, module: str, country: str, segment: str, cap: float) -> str:
    history, forecast = _segment_rows(table, country, segment)
    note = " Tertiary is derived as 1 - secondary." if module == "education" else ""
    return (
        f"Country: {country}\nSegment: {segment}\nModule: {module}.{note}\n"
        f"Historical values (not editable): {_fmt(history)}\nForecast values (editable unless NA): {_fmt(forecast)}\n"
        f"Year-over-year cap for this country-segment: {cap:.3f} (absolute change between consecutive years)"
    )


def _parse_plan(text: str | None, module: str, editable: set[tuple[str, str]], max_suspects: int) -> tuple[list[tuple[str, str]], list]:
    """Suspects in the model's order (deduplicated, only editable segments; education tertiary maps to secondary), at most
    ``max_suspects`` of them, and the ignored items (invalid ones and valid ones beyond the cap). The JSON object is decoded from the
    first ``{``; anything after it (prose, even with braces) is ignored."""
    if not text or "{" not in text:
        raise ValueError("reply contains no JSON object")
    data, _ = json.JSONDecoder().raw_decode(text, text.index("{"))
    if not isinstance(data, dict) or not isinstance(data.get("suspects"), list):
        raise ValueError('JSON must be an object with a "suspects" list')
    suspects, ignored = [], []
    for item in data["suspects"]:
        country, segment = (item.get("country"), item.get("segment")) if isinstance(item, dict) else (None, None)
        if not (isinstance(country, str) and isinstance(segment, str)):
            ignored.append(item)
            continue
        if module == "education" and segment == "tertiary":
            segment = "secondary"
        if (country, segment) not in editable:
            ignored.append(item)
        elif (country, segment) not in suspects:
            if len(suspects) < max_suspects:
                suspects.append((country, segment))
            else:
                ignored.append(item)
    return suspects, ignored


def _source_counts(store: SourceStore, countries) -> dict[str, int] | None:
    """Documents per country for the planner summary; ``None`` (the line is dropped) when the store cannot count."""
    try:
        return {country: store.search_with_total("", country=country)[1] for country in countries}
    except Exception as exc:  # noqa: BLE001 - the counts are a planning hint only; losing them must not end the run
        logger.warning("Could not get source counts for the planner summary (%s); continuing without them", type(exc).__name__)
        return None


def _plan(session: CorrectionSession, module: str, llm: LLMClient, settings: dict, log: list[dict]) -> list[tuple[str, str]] | None:
    editable = _editable_segments(session.baseline, module)
    counts = _source_counts(session.store, sorted({c for c, _ in editable}))
    messages = [
        {"role": "system", "content": planner_system_prompt()},
        {"role": "user", "content": summarize_table(session.baseline, module, MAX_SUMMARY_ROWS, counts, session.caps)},
    ]
    attempts = 1 + settings["max_llm_retries"]
    for attempt in range(attempts):
        response = llm.chat(messages, [])
        try:
            suspects, ignored = _parse_plan(response.text, module, set(editable), settings["max_suspects"])
        except ValueError as exc:
            log.append({"event": "plan_invalid", "attempt": attempt + 1, "reason": str(exc)})
            messages += [
                {"role": "assistant", "content": response.text or ""},
                {"role": "user", "content": f"Your reply could not be used ({exc}). Reply with the JSON object only."},
            ]
            continue
        log.append({"event": "plan", "suspects": [list(s) for s in suspects], "ignored": ignored})
        return suspects
    return None


def _research(session: CorrectionSession, module: str, country: str, segment: str, llm: LLMClient, settings: dict, log: list[dict]) -> None:
    budget = settings["max_iterations_per_segment"]
    messages = [
        {"role": "system", "content": researcher_system_prompt(budget)},
        {"role": "user", "content": _segment_brief(session.working, module, country, segment, session.caps[(country, segment)])},
    ]
    schemas = tool_schemas()
    where = {"country": country, "segment": segment}
    for turn in range(1, budget + 1):
        response = llm.chat(messages, schemas)
        if not response.tool_calls:
            log.append({"event": "done", **where, "turn": turn, "text": response.text})
            return
        messages.append(
            {
                "role": "assistant",
                "content": response.text,
                "tool_calls": [
                    {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                    for tc in response.tool_calls
                ],
            }
        )
        for tc in response.tool_calls:
            result = {"error": tc.parse_error} if tc.parse_error else session.dispatch(tc.name, tc.arguments)
            log.append({"event": "tool", **where, "turn": turn, "tool": tc.name, "arguments": tc.arguments, "result": result})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
    log.append({"event": "budget_exhausted", **where, "turns": budget})


def run_correction(baseline: ForecastTable, module: str, store: SourceStore, llm: LLMClient, config: dict) -> CorrectionResult:
    settings = config["correction"]
    session = CorrectionSession(baseline, module, store, config)
    log: list[dict] = []
    error = None
    try:
        suspects = _plan(session, module, llm, settings, log)
        if suspects is None:
            error = f"planner returned no valid JSON plan after {1 + settings['max_llm_retries']} attempts"
        for country, segment in suspects or []:
            _research(session, module, country, segment, llm, settings, log)
    except AssertionError:
        # ScriptedClient raises AssertionError("script exhausted") when a test script is too short: that is a bug in the test,
        # not an LLM failure, so it must surface instead of being turned into a quiet ``error``.
        raise
    except Exception as exc:  # noqa: BLE001 - any client failure (LLMUnavailableError, IndexError on empty choices, ...) ends the run
        error = f"{type(exc).__name__}: {exc}"[:ERROR_MAX_CHARS]
    if error:
        log.append({"event": "error", "message": error})
    return CorrectionResult(session.working.copy(), list(session.corrections), list(session.rejections), log, error)
