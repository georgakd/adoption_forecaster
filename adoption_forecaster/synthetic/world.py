"""Synthetic world: deterministic, seeded ground truth plus the shocks that make forecasts miss."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DEFAULT_COUNTRIES = ["Austria", "Belgium", "Czech Republic", "Denmark", "France", "Germany", "Greece", "Italy", "Poland", "Spain"]
EDUCATION_START_YEAR = 2004
AGE_GENDER_START_YEAR = 2017
AGE_GROUPS = {"16 to 24": 0.9, "25 to 34": 0.88, "35 to 44": 0.84, "45 to 54": 0.78, "55 to 64": 0.68, "65 to 74": 0.52}
GENDERS = {"Males": 0.01, "Females": -0.01}
SHOCK_SEED_OFFSET = 10_000


@dataclass
class Shock:
    year: int
    delta: float


@dataclass
class World:
    countries: list[str]
    last_year: int
    pred_year: int
    truth_education: pd.DataFrame
    truth_age_gender: pd.DataFrame
    shocks: dict[str, Shock] = field(default_factory=dict)


def _logistic(years: np.ndarray, midpoint: float, rate: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-rate * (years - midpoint)))


def generate_world(
    seed: int,
    countries: list[str] | None = None,
    last_year: int = 2024,
    pred_year: int = 2026,
    shocked_fraction: float = 0.3,
) -> World:
    countries = list(countries) if countries is not None else list(DEFAULT_COUNTRIES)
    rng = np.random.default_rng(seed)

    edu_years = np.arange(EDUCATION_START_YEAR, pred_year + 1)
    ag_years = np.arange(AGE_GENDER_START_YEAR, pred_year + 1)

    tertiary: dict[str, np.ndarray] = {}
    age_gender_rows: list[tuple[str, str, np.ndarray]] = []
    for country in countries:
        lo, hi = rng.uniform(0.35, 0.45), rng.uniform(0.5, 0.6)
        curve = lo + (hi - lo) * _logistic(edu_years, rng.uniform(2010, 2018), rng.uniform(0.2, 0.4))
        tertiary[country] = curve + rng.normal(0, 0.003, len(edu_years))
        for age, ceiling in AGE_GROUPS.items():
            base_level = ceiling * rng.uniform(0.9, 1.0)
            curve = base_level * _logistic(ag_years, rng.uniform(2008, 2016), rng.uniform(0.2, 0.35))
            for gender, offset in GENDERS.items():
                values = curve + offset + rng.normal(0, 0.004, len(ag_years))
                age_gender_rows.append((country, f"{gender}, {age} years old", values))

    # shocks use a separate stream so shocked_fraction never changes the underlying curves
    shock_rng = np.random.default_rng(seed + SHOCK_SEED_OFFSET)
    shocks: dict[str, Shock] = {}
    for country in countries:
        if shock_rng.random() < shocked_fraction:
            sign = shock_rng.choice([-1.0, 1.0])
            shocks[country] = Shock(year=int(shock_rng.integers(last_year + 1, pred_year + 1)), delta=float(sign * shock_rng.uniform(0.03, 0.08)))

    edu_records = {}
    for country in countries:
        ter = tertiary[country].copy()
        if country in shocks:
            ter[edu_years >= shocks[country].year] += shocks[country].delta
        ter = np.clip(ter, 0.0, 1.0)
        edu_records[(country, "secondary")] = 1.0 - ter
        edu_records[(country, "tertiary")] = ter
    truth_education = pd.DataFrame.from_dict(edu_records, orient="index", columns=[int(y) for y in edu_years])
    truth_education.index = pd.MultiIndex.from_tuples(truth_education.index, names=["country", "education"])

    ag_records = []
    for country, indiv, values in age_gender_rows:
        values = values.copy()
        if country in shocks:
            values[ag_years >= shocks[country].year] += shocks[country].delta
        clipped = np.clip(values, 0.0, 1.0)
        ag_records.append({"country": country, "Individual type": indiv, **{str(int(y)): float(v) for y, v in zip(ag_years, clipped)}})
    truth_age_gender = pd.DataFrame(ag_records)

    return World(countries, last_year, pred_year, truth_education, truth_age_gender, shocks)
