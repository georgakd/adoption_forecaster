"""Synthetic "published statistics" documents generated from a truth ForecastTable.

Each document states one forecast-year value (the truth plus small noise, or a wrong value) inside a short prose paragraph that also
mentions distractor numbers taken from historical years (same segment, other segment). Some documents are wrong, some keys get two
documents that contradict each other, and some countries get no documents at all.

``SourceDoc.value`` (and the ground-truth ``segment``) are for tests and the evaluator only; the agent sees snippets and ``text``.
Seeds >= ``HELDOUT_SEED_START`` draw publishers and phrasing from a disjoint style pool, so held-out evaluation guards only against
memorising dev-set publisher names and sentence templates. Everything else is identical to dev: the noise clip, the wrong-value offset
range, the unit mix, the no-document fraction and the distractor structure (which numbers appear as distractors). Held-out evaluation
therefore does NOT protect against tuning on those parameters.
"""

import re
from dataclasses import dataclass

import numpy as np

from quotas_forecasting.correction.types import ForecastTable

HELDOUT_SEED_START = 100
NO_DOCS_COUNTRY_FRACTION = 0.25
NOISE_CLIP = 2.0  # noise is clipped to +-NOISE_CLIP * noise so correct docs stay clearly apart from wrong ones
WRONG_OFFSET = (0.06, 0.12)  # far enough that a wrong doc and a correct doc on the same key always disagree visibly

DEV_STYLE = {
    "publishers": [
        "National Statistics Office",
        "Digital Economy Observatory",
        "Eurobarometer Digital Survey",
        "Telecom Regulator Annual Report",
        "Market Insight Panel",
        "European Internet Monitor",
    ],
    "target": [
        "According to the {publisher}, {subject} in {country} reached {value} in {year}.",
        "In {year}, {subject} in {country} stood at {value}, the {publisher} reports.",
        "The {publisher} estimates {subject} in {country} at {value} for {year}.",
    ],
    "distractor": [
        "This compares with {value} in {year} for {subject}.",
        "For reference, {subject} was {value} in {year}.",
    ],
    "title": [
        "{publisher}: {country} digital indicators {year}",
        "{country} internet use, {year} release ({publisher})",
    ],
}

HELDOUT_STYLE = {
    "publishers": [
        "Federal Data Bureau",
        "Continental Web Census",
        "Connectivity Research Institute",
        "Online Population Tracker",
        "Household ICT Study",
        "Media Consumption Index",
    ],
    "target": [
        "Figures released by the {publisher} put {subject} in {country} at {value} for the year {year}.",
        "{country}: {subject} amounted to {value} in {year} (source: {publisher}).",
        "Survey results from the {publisher} show {subject} in {country} at {value} during {year}.",
    ],
    "distractor": [
        "Back in {year}, {subject} had been {value}.",
        "Earlier data show {value} for {subject} in {year}.",
    ],
    "title": [
        "{year} connectivity bulletin for {country} - {publisher}",
        "{publisher} | {country} online population, {year}",
    ],
}


@dataclass
class SourceDoc:
    doc_id: str
    publisher: str
    country: str
    year: int
    unit: str  # "share" (0-1) or "percent"
    segment: str
    text: str
    value: float | None  # ground truth of the number the doc states; never shown to the agent
    title: str = ""


def format_value(value: float, unit: str) -> str:
    """The exact string a document uses for a value in the given unit."""
    if unit == "percent":
        return f"{value * 100:.1f}%"
    if unit == "share":
        return f"{value:.3f}"
    raise ValueError(f"Unknown unit: {unit}")


def describe_segment(segment: str) -> str:
    if segment in ("secondary", "tertiary"):
        return f"the share of internet users with {segment} education"
    match = re.fullmatch(r"(Male|Female)s?, (\d+ to \d+)(?: years old)?", segment)
    if match:
        people = "men" if match.group(1) == "Male" else "women"
        return f"the share of {people} aged {match.group(2)} who use the internet"
    return f"the {segment} indicator"


def _stated(true_value: float, wrong: bool, noise: float, rng: np.random.Generator) -> float:
    if wrong:
        offset = rng.uniform(*WRONG_OFFSET) * rng.choice([-1.0, 1.0])
        if not 0.0 <= true_value + offset <= 1.0:
            offset = -offset
        value = true_value + offset
    else:
        value = true_value + float(np.clip(rng.normal(0.0, noise), -NOISE_CLIP * noise, NOISE_CLIP * noise))
    return round(float(np.clip(value, 0.0, 1.0)), 3)


def make_docs(truth: ForecastTable, seed: int, noise: float, wrong_fraction: float, conflict_fraction: float) -> list[SourceDoc]:
    """Generate source documents for the forecast years of ``truth`` (works on any ForecastTable, synthetic or real).

    Every forecast key of a covered country gets one primary doc; ``round(conflict_fraction * keys)`` keys get a second doc from
    another publisher whose value disagrees (exactly one of the pair is wrong). Further primary docs are made wrong so that
    ``round(wrong_fraction * all docs)`` docs are wrong overall (at least one per conflicting pair).
    """
    rng = np.random.default_rng(seed)
    style = HELDOUT_STYLE if seed >= HELDOUT_SEED_START else DEV_STYLE
    truth = truth.dropna(subset=["value"])
    history = truth[~truth["is_forecast"]]
    targets = truth[truth["is_forecast"]].sort_values(["country", "segment", "year"]).reset_index(drop=True)

    countries = sorted(targets["country"].unique())
    n_without = max(1, round(NO_DOCS_COUNTRY_FRACTION * len(countries))) if len(countries) > 1 else 0
    without = set(rng.choice(countries, size=n_without, replace=False)) if n_without else set()
    targets = targets[~targets["country"].isin(without)].reset_index(drop=True)

    n_keys = len(targets)
    n_conflict = round(conflict_fraction * n_keys)
    n_wrong = round(wrong_fraction * (n_keys + n_conflict))
    order = rng.permutation(n_keys)
    conflict_idx = set(order[:n_conflict].tolist())
    wrong_idx = set(order[n_conflict : n_conflict + max(n_wrong - n_conflict, 0)].tolist())

    specs = []  # (row, wrong, publisher)
    for i, row in enumerate(targets.itertuples(index=False)):
        publishers = rng.choice(style["publishers"], size=2, replace=False)
        if i in conflict_idx:
            primary_wrong = bool(rng.random() < 0.5)
            specs += [(row, primary_wrong, str(publishers[0])), (row, not primary_wrong, str(publishers[1]))]
        else:
            specs.append((row, i in wrong_idx, str(publishers[0])))

    docs = []
    for n, k in enumerate(rng.permutation(len(specs))):
        row, wrong, publisher = specs[k]
        unit = str(rng.choice(["share", "percent"]))
        value = _stated(float(row.value), wrong, noise, rng)
        fields = {"publisher": publisher, "country": row.country, "year": row.year, "subject": describe_segment(row.segment)}
        sentences = [str(rng.choice(style["target"])).format(value=format_value(value, unit), **fields)]
        past = history[history["country"] == row.country]
        for pool in (past[past["segment"] == row.segment], past[past["segment"] != row.segment]):
            if len(pool):
                d = pool.iloc[int(rng.integers(len(pool)))]
                distractor = {"value": format_value(float(d["value"]), unit), "year": int(d["year"]), "subject": describe_segment(d["segment"])}
                sentences.append(str(rng.choice(style["distractor"])).format(**distractor))
        sentences = [sentences[j] for j in rng.permutation(len(sentences))]
        docs.append(
            SourceDoc(
                doc_id=f"src-{n:04d}",
                publisher=publisher,
                country=row.country,
                year=int(row.year),
                unit=unit,
                segment=row.segment,
                text=" ".join(sentences),
                value=value,
                title=str(rng.choice(style["title"])).format(**fields),
            )
        )
    return docs
