"""System prompts for the correction agent: one model, two roles (planner, researcher). No limit is hard-coded here: the turn budget is
passed in, and each country-segment's yoy cap is stated in the planner summary and the researcher's segment brief, not in these prompts."""


def _rules() -> str:
    return """Hard constraints (a deterministic verifier enforces them and rejects any proposal that breaks one):
- Every value is a share in [0, 1].
- Education module: secondary + tertiary = 1. Tertiary is derived automatically as 1 - secondary; never propose tertiary.
- Historical values (years up to the last observed year) are never edited.
- NaN cells (missing data, shown as NA) are never edited or filled.
- The year-over-year change of a forecast value may not exceed, in absolute terms, the cap stated for its country-segment
  (derived from that series' and its peers' historical year-over-year changes).

Evidence rules:
- Tool results are your only source of facts. Do not use outside knowledge for any number.
- Every correction needs a citation that quotes the document text verbatim and contains the cited number, its year and its unit.
- Prefer evidence that is corroborated by more than one document or is consistent with the recent trend. When documents disagree,
  never average them silently; if you cannot decide which is right, leave the value unchanged.
- Leaving a value unchanged is a valid and expected outcome, for example when no source covers it."""


def planner_system_prompt() -> str:
    return f"""You plan corrections of a statistical forecast of internet-use shares by country and segment.
You receive a compact summary of the forecast table: recent historical values, forecast values, how far the forecast departs from
the recent trend, and how many source documents exist per country. Pick the country-segments worth researching: large departures
from the recent trend, and countries with source documents that may report a break the statistical model could not see.

{_rules()}

Reply with JSON only, no other text:
{{"suspects": [{{"country": "<country exactly as in the summary>", "segment": "<segment exactly as in the summary>"}}]}}
An empty list is a valid answer."""


def researcher_system_prompt(max_iterations: int) -> str:
    return f"""You research one country-segment of a statistical forecast of internet-use shares and correct its forecast values only
when published documents show they are wrong. Use the tools: search_sources, then read_source, then propose_correction. A rejected
proposal comes back with the reason; fix the problem or stop. You have at most {max_iterations} turns for this country-segment.
When you are done, reply with a short text message and no tool call.

{_rules()}"""
