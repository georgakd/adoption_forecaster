from pandas.testing import assert_frame_equal

from quotas_forecasting.correction.adapters import from_table, to_table

LAST_YEAR = 2024


def _check_table(table, last_year):
    assert list(table.columns) == ["country", "segment", "year", "value", "is_forecast"]
    assert table["year"].dtype == "int64" and table["is_forecast"].dtype == bool
    assert (table["is_forecast"] == (table["year"] > last_year)).all()


def test_roundtrip_education_and_age_gender(captured_frames):
    education = captured_frames["education"][0]
    assert education.isna().any().any()
    table = to_table(education, "education", LAST_YEAR)
    _check_table(table, LAST_YEAR)
    assert set(table["segment"]) == {"secondary", "tertiary"}
    assert table["value"].isna().any()
    assert_frame_equal(from_table(table, education, "education"), education)

    age_gender = captured_frames["age_gender"][0]
    # stage 1: as returned by run_with_data (mixed int/str year labels, NaN missing)
    table = to_table(age_gender, "age_gender", LAST_YEAR)
    _check_table(table, LAST_YEAR)
    assert_frame_equal(from_table(table, age_gender, "age_gender"), age_gender)

    # stage 2: after the processor's rename and fillna(":"), with ":" placeholders; make sure some are present
    filled = age_gender.copy()
    filled.loc[filled.index[0], 2017] = float("nan")
    filled.loc[filled.index[1], "2026"] = float("nan")
    filled = filled.rename(columns={"2025": 2025, "2026": 2026}).astype({2017: object, 2026: object})
    filled = filled.fillna(":")
    assert (filled == ":").any().any()
    table = to_table(filled, "age_gender", LAST_YEAR)
    assert table["value"].isna().sum() == 2
    result = from_table(table, filled, "age_gender")
    assert_frame_equal(result, filled)
    assert result.loc[filled.index[0], 2017] == ":"
