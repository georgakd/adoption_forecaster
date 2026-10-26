"""Write the visible (<= last_year) part of a World as the timestamped CSV inputs the real loader expects."""

from pathlib import Path

import numpy as np
import pandas as pd

from adoption_forecaster.config.config_loader import load_config
from adoption_forecaster.synthetic.world import EDUCATION_START_YEAR, World

TIMESTAMP = "2024-01-01T00_00_00+0000"
NAME_VARIANTS = {"Czech Republic": "Czechia"}  # exercises country_mapping on load
AGE_RANGES = ["16-64", "16-74"]


def _path(input_dir: Path, base: str) -> Path:
    return Path(input_dir) / f"{base}_{TIMESTAMP}.csv"


def _file_name(module: str, key: str) -> str:
    return load_config()["modules"][module]["files"][key].replace(".csv", "")


def write_education_inputs(world: World, input_dir: Path) -> list[Path]:
    """Inputs consistent with the truth: tertiary share of internet users equals truth_education up to last_year."""
    input_dir = Path(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    years = list(range(EDUCATION_START_YEAR, world.last_year + 1))
    rng = np.random.default_rng(len(world.countries) + world.last_year)
    total_rows, ratio_rows, penetration_rows = [], [], []
    for i, country in enumerate(world.countries):
        name = NAME_VARIANTS.get(country, country)
        ter = world.truth_education.loc[(country, "tertiary")]
        pop3_start = rng.uniform(0.18, 0.26)
        population = rng.uniform(2e6, 6e7)
        for age_range in AGE_RANGES:
            total_rows.append({"country": name, "Age group": age_range, **{str(y): int(population * (1 + 0.003 * (y - years[0]))) for y in years}})
            ratios = {1: {}, 2: {}, 3: {}}
            pens = {1: {}, 2: {}, 3: {}}
            for y in years:
                p3 = pop3_start + 0.002 * (y - years[0])
                p12 = 1 - p3
                i3 = min(0.95, 0.7 + 0.25 * (y - years[0]) / (years[-1] - years[0]))
                t = float(ter[y])
                s = i3 * (1 - t) * p3 / (t * p12)  # makes tertiary user share equal t
                for level, p, pen in ((1, 0.3 * p12, s), (2, 0.7 * p12, s), (3, p3, i3)):
                    ratios[level][str(y)] = round(p * 100, 6)
                    pens[level][str(y)] = round(pen * 100, 6)
            for level in (1, 2, 3):
                ratio_rows.append({"country": name, "Age group": age_range, "Education level": level, **ratios[level]})
                row = {"country": name, "Age group": age_range, "Education level": level, **pens[level]}
                if i == 1:  # one country has no data in the first year
                    row[str(years[0])] = ":"
                penetration_rows.append(row)
    paths = []
    for key, rows in (
        ("total_population_by_age_range", total_rows),
        ("population_ratios", ratio_rows),
        ("internet_penetration_population", penetration_rows),
    ):
        path = _path(input_dir, _file_name("education", key))
        pd.DataFrame(rows).to_csv(path, index=False)
        paths.append(path)
    return paths


def write_age_gender_inputs(world: World, input_dir: Path) -> list[Path]:
    input_dir = Path(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    df = world.truth_age_gender.copy()
    # the pipeline pairs gender labels with predictions sorted by (country, gender male-first, age), so rows must already be in that order
    df = df.assign(_female=df["Individual type"].str.startswith("Females")).sort_values(["country", "_female"], kind="stable").drop(columns="_female")
    keep = ["country", "Individual type"] + [c for c in df.columns[2:] if int(c) <= world.last_year]
    df = df[keep]
    df["country"] = df["country"].replace(NAME_VARIANTS)
    df[df.columns[2:]] = (df[df.columns[2:]] * 100).round(6)  # Eurostat age/gender files are in percent; the pipeline divides by 100
    path = _path(input_dir, _file_name("age_gender", "age_gender"))
    df.to_csv(path, index=False)
    return [path]
