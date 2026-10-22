"""The agent's three tools over a working forecast table, with a deterministic gate that accepts or rejects each proposed correction.

Nobody reviews the agent's output, so a proposal changes ``working`` only if every check passes: the cell is a non-NaN forecast cell,
the value is in [0, 1], the citation verifies and is tied to the proposal (``citation_supports_correction``), and the change introduces
no new hard-constraint violation. Tools never raise on bad input from the model; they return a rejection or an ``{"error": ...}``.
"""

import inspect
import math
from dataclasses import dataclass

from quotas_forecasting.correction.citations import citation_supports_correction
from quotas_forecasting.correction.constraints import check_constraints
from quotas_forecasting.correction.sources import SourceStore
from quotas_forecasting.correction.types import Citation, Correction, ForecastTable, Violation

TOOL_NAMES = ("search_sources", "read_source", "propose_correction")
CITATION_FIELDS = {"doc_id": str, "quoted_text": str, "value": float, "year": int, "unit": str}


@dataclass
class Rejection:
    proposal: dict
    reason: str


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


_CHECKS = {str: lambda x: isinstance(x, str), int: _is_int, float: _is_number}


def _type_errors(values: dict, types: dict) -> list[str]:
    return [f"{name} must be {t.__name__}" for name, t in types.items() if not _CHECKS[t](values.get(name))]


def _key(v: Violation) -> tuple:
    return (v.kind, v.country, v.segment, v.year)


class CorrectionSession:
    """Holds the working table for one module and exposes the agent's tools; ``corrections``/``rejections`` record every decision."""

    def __init__(self, baseline: ForecastTable, module: str, store: SourceStore, config: dict):
        self.baseline = baseline.copy()
        self.working = baseline.copy()
        self.module = module
        self.store = store
        self.max_yoy_change = config["correction"]["max_yoy_change"]
        self.tolerance = config["correction"]["citation_tolerance"]
        self.corrections: list[Correction] = []
        self.rejections: list[Rejection] = []

    def search_sources(self, query: str, country: str | None = None, year: int | None = None) -> dict:
        """Snippet metadata only: ``{"results": [{doc_id, publisher, country, year, title}], "truncated": bool, "total": int}``."""
        if not isinstance(query, str) or not (country is None or isinstance(country, str)) or not (year is None or _is_int(year)):
            raise ValueError("query must be a string, country a string or null, year an integer or null")
        snippets, total = self.store.search_with_total(query, country=country, year=year)
        results = [{"doc_id": s.doc_id, "publisher": s.publisher, "country": s.country, "year": int(s.year), "title": s.title} for s in snippets]
        return {"results": results, "truncated": total > len(results), "total": total}

    def read_source(self, doc_id: str) -> str:
        if not isinstance(doc_id, str):
            raise ValueError("doc_id must be a string")
        try:
            return self.store.read(doc_id)
        except KeyError:
            raise ValueError(f"unknown doc_id {doc_id!r}") from None

    def propose_correction(self, country: str, segment: str, year: int, new_value: float, citation: dict, confidence: float, reason: str) -> dict:
        proposal = {
            "country": country,
            "segment": segment,
            "year": year,
            "new_value": new_value,
            "citation": citation,
            "confidence": confidence,
            "reason": reason,
        }

        def reject(why: str) -> dict:
            self.rejections.append(Rejection(proposal, why))
            return {"accepted": False, "reason": why}

        types = {"country": str, "segment": str, "year": int, "new_value": float, "confidence": float, "reason": str}
        errors = _type_errors(proposal, types)
        if not isinstance(citation, dict):
            errors.append("citation must be an object with doc_id, quoted_text, value, year, unit")
        else:
            errors += [f"citation.{e}" for e in _type_errors(citation, CITATION_FIELDS)]
        if errors:
            return reject("invalid arguments: " + "; ".join(errors))
        if self.module == "education" and segment == "tertiary":
            return reject("tertiary cannot be proposed directly: it is derived as 1 - secondary when a secondary correction is accepted")

        table = self.working
        cell = table.index[(table["country"] == country) & (table["segment"] == segment) & (table["year"] == year)]
        if len(cell) != 1:
            return reject(f"no forecast cell for country={country!r}, segment={segment!r}, year={year}")
        idx = cell[0]
        if not table.at[idx, "is_forecast"]:
            return reject(f"{country}/{segment}/{year} is a historical cell; historical values are never changed")
        old_value = table.at[idx, "value"]
        if math.isnan(old_value):
            return reject(f"{country}/{segment}/{year} is NaN (missing data); NaN cells are never filled")
        if not 0.0 <= new_value <= 1.0:
            return reject(f"new_value {new_value} is outside [0, 1]")

        cite = Citation(citation["doc_id"], citation["quoted_text"], float(citation["value"]), citation["year"], citation["unit"])
        correction = Correction(country, segment, year, float(old_value), float(new_value), cite, float(confidence), reason)
        if not citation_supports_correction(correction, self.store, self.tolerance):
            return reject(
                "citation does not verify: quoted_text must appear verbatim in the document and state new_value in the cited unit for "
                f"the proposed year (within {self.tolerance}), citation.year must equal year, and the document must name {country}"
            )

        tentative = table.copy()
        tentative.at[idx, "value"] = float(new_value)
        if self.module == "education" and segment == "secondary":
            tertiary = tentative.index[(tentative["country"] == country) & (tentative["segment"] == "tertiary") & (tentative["year"] == year)]
            if len(tertiary) == 1 and not math.isnan(tentative.at[tertiary[0], "value"]):
                tentative.at[tertiary[0], "value"] = 1.0 - float(new_value)
        existing = {_key(v) for v in check_constraints(self.baseline, table, self.module, self.max_yoy_change)}
        introduced = [v for v in check_constraints(self.baseline, tentative, self.module, self.max_yoy_change) if _key(v) not in existing]
        if introduced:
            details = "; ".join(f"{v.kind} at {v.country}/{v.segment}/{v.year} ({v.detail})" for v in introduced)
            return reject(f"would introduce constraint violation(s): {details}")

        self.working = tentative
        self.corrections.append(correction)
        return {"accepted": True, "reason": "accepted"}

    def dispatch(self, name: str, arguments: dict) -> dict:
        """Route one tool call; return the tool's JSON-serializable result, or ``{"error": ...}`` (never raises)."""
        if name not in TOOL_NAMES:
            return {"error": f"unknown tool {name!r}; available: {', '.join(TOOL_NAMES)}"}
        if not isinstance(arguments, dict):
            return {"error": "arguments must be a JSON object"}
        method = getattr(self, name)
        try:
            inspect.signature(method).bind(**arguments)
        except TypeError as exc:
            return {"error": f"bad arguments for {name}: {exc}"}
        try:
            result = method(**arguments)
        except Exception as exc:
            return {"error": f"{name} failed: {exc}"}
        return {"text": result} if name == "read_source" else result


def tool_schemas() -> list[dict]:
    """OpenAI-format tool definitions for ``search_sources``, ``read_source`` and ``propose_correction`` (in that order)."""

    def tool(name, description, properties, required):
        return {
            "type": "function",
            "function": {"name": name, "description": description, "parameters": {"type": "object", "properties": properties, "required": required}},
        }

    citation = {
        "type": "object",
        "description": "Evidence for the value, copied from a document returned by read_source.",
        "properties": {
            "doc_id": {"type": "string", "description": "The doc_id of the document quoted."},
            "quoted_text": {"type": "string", "description": "A verbatim excerpt of the document that states the value and its year."},
            "value": {"type": "number", "description": "The stated value on the 0-1 share scale (e.g. 61.8% -> 0.618)."},
            "year": {"type": "integer", "description": "The year the quoted value refers to."},
            "unit": {
                "type": "string",
                "enum": ["percent", "share"],
                "description": "How the quote writes the number: percent (61.8%) or share (0.618).",
            },
        },
        "required": list(CITATION_FIELDS),
    }
    return [
        tool(
            "search_sources",
            "Search published source documents. Returns snippet metadata (doc_id, publisher, country, year, title), whether results were "
            "truncated, and the total number of matches; narrow the query, country or year when truncated.",
            {
                "query": {"type": "string", "description": "Keywords to search for."},
                "country": {"type": "string", "description": "Only documents about this country."},
                "year": {"type": "integer", "description": "Only documents about this year."},
            },
            ["query"],
        ),
        tool(
            "read_source",
            "Read the full text of a source document by doc_id.",
            {"doc_id": {"type": "string", "description": "A doc_id returned by search_sources."}},
            ["doc_id"],
        ),
        tool(
            "propose_correction",
            "Propose a new value for one non-missing forecast cell, backed by a verbatim citation. A deterministic gate accepts it only if the "
            "cell is a forecast cell with a value, the new value is in [0, 1], the citation verifies and states new_value for that country "
            "and year, and the change introduces no constraint violation (including the cap on year-over-year change). In the education "
            "module propose secondary only; tertiary is set to 1 - secondary automatically.",
            {
                "country": {"type": "string", "description": "Country name exactly as in the forecast table."},
                "segment": {"type": "string", "description": "Segment name exactly as in the forecast table."},
                "year": {"type": "integer", "description": "Forecast year of the cell."},
                "new_value": {"type": "number", "description": "Proposed value on the 0-1 share scale."},
                "citation": citation,
                "confidence": {"type": "number", "description": "Your confidence in the correction, 0-1."},
                "reason": {"type": "string", "description": "Why the baseline value is wrong and the cited one is right."},
            },
            ["country", "segment", "year", "new_value", "citation", "confidence", "reason"],
        ),
    ]
