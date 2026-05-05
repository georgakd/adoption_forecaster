"""
Tools for the agent
"""

import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def analyze_country_row_trend_education(df: pd.DataFrame, country: str, country_col: str, projection_cols: list, feedback_retriever=None) -> dict:
    """
    Analyze one country's row: for each projection column, compare it to the previous year
    of the same type. If change exceeds 1%, it's an outlier.
    """
    country_rows = df[df[country_col] == country]
    if len(country_rows) == 0:
        return {"error": f"Country '{country}' not found"}

    country_data = country_rows.iloc[0]

    results = {"country": country, "analyses": {}}

    # analyze each projection column separate
    for proj_col in projection_cols:
        # get previous year column of the same type
        year = proj_col.split("_")[1]
        prev_year = str(int(year) - 1)
        prev_col = proj_col.replace(year, prev_year)

        try:
            prev_value = pd.to_numeric(country_data[prev_col])
            if pd.isna(prev_value):
                results["analyses"][proj_col] = {"error": f"Previous year value ({prev_col}) is NaN"}
                continue
        except (ValueError, TypeError, KeyError) as e:
            results["analyses"][proj_col] = {"error": f"Could not read previous year value from {prev_col}: {e}"}
            continue

        try:
            current_value = pd.to_numeric(country_data[proj_col])
            if pd.isna(current_value):
                results["analyses"][proj_col] = {"error": f"Projection value is NaN for {proj_col}"}
                continue
        except (ValueError, TypeError, KeyError) as e:
            results["analyses"][proj_col] = {"error": f"Could not read projection value for {proj_col}: {e}"}
            continue

        # calc year-over-year change
        change = current_value - prev_value
        change_pct = (change / prev_value * 100) if prev_value != 0 else 0

        # check if exceeds 1% threshold
        is_outlier = abs(change_pct) > 0.5
        if is_outlier:
            logger.info(f"[OUTLIER] {proj_col}: change_pct={change_pct:.4f}% > 0.5%")
        else:
            logger.info(f"[OK] {proj_col}: change_pct={change_pct:.4f}% <= 0.5%")

        parts = proj_col.split("_")
        col_type = "_".join(parts[1:]) if len(parts) >= 2 else "unknown"

        # get feedback for secondary level
        feedback_text = None
        if feedback_retriever and proj_col.startswith("secondary_"):
            feedback_text = feedback_retriever.get_feedback_education(
                country=country,
                level=proj_col,  # e.g., "secondary_2026"
                projection_value=float(current_value),
                previous_value=float(prev_value) if not pd.isna(prev_value) else None,
            )
            if feedback_text:
                logger.info(
                    f"[FEEDBACK] Retrieved feedback for {country} {proj_col}: {feedback_text[:100]}..."
                    if len(feedback_text) > 100
                    else f"[FEEDBACK] Retrieved feedback for {country} {proj_col}: {feedback_text}"
                )
            else:
                logger.debug(f"[FEEDBACK] No feedback found for {country} {proj_col}")

        results["analyses"][proj_col] = {
            "type": col_type,
            "previous_year_column": prev_col,
            "previous_year_value": prev_value,
            "current_value": current_value,
            "change": change,
            "change_pct": change_pct,
            "is_outlier": is_outlier,
            "threshold": 1.0,
            "feedback": feedback_text,
        }

    return results


def analyze_country_trend_education(df: pd.DataFrame, country: str, country_col: str, projection_cols: list, feedback_retriever=None) -> dict:
    """
    Analyze one country's row: for each projection, compare it to the previous year
    of the same type. If change exceeds 1%, it's an outlier.
    Returns a dictionary matching AnalyzeCountryTrendToolOutput schema (pure tool output).
    """
    try:
        trend_info = analyze_country_row_trend_education(df, country, country_col, projection_cols, feedback_retriever=feedback_retriever)

        if "error" in trend_info:
            return {"success": False, "country": country, "analyses": {}, "outliers": [], "error": trend_info["error"]}

        outliers = []
        analyses = {}

        for proj_col, analysis in trend_info["analyses"].items():
            if "error" in analysis:
                analyses[proj_col] = {
                    "column": proj_col,
                    "type": "unknown",
                    "previous_year_column": "",
                    "previous_year_value": 0.0,
                    "current_value": 0.0,
                    "change": 0.0,
                    "change_pct": 0.0,
                    "is_outlier": False,
                    "threshold": 1.0,
                    "error": analysis["error"],
                    "feedback": None,
                }
            else:
                analyses[proj_col] = {
                    "column": proj_col,
                    "type": analysis.get("type", "unknown"),
                    "previous_year_column": analysis.get("previous_year_column", ""),
                    "previous_year_value": float(analysis.get("previous_year_value", 0.0)),
                    "current_value": float(analysis.get("current_value", 0.0)),
                    "change": float(analysis.get("change", 0.0)),
                    "change_pct": float(analysis.get("change_pct", 0.0)),
                    "is_outlier": bool(analysis.get("is_outlier", False)),
                    "threshold": float(analysis.get("threshold", 0.5)),
                    "error": None,
                    "feedback": analysis.get("feedback"),
                }
                if analyses[proj_col]["is_outlier"]:
                    outliers.append(proj_col)

        return {"success": True, "country": country, "analyses": analyses, "outliers": outliers, "error": None}
    except Exception as e:
        return {"success": False, "country": country, "analyses": {}, "outliers": [], "error": f"Error analyzing trend: {e}"}


def adjust_country_projections_education(
    df: pd.DataFrame, country: str, country_col: str, projection_cols: list, adjustments: dict, reason: str = ""
) -> tuple[dict, pd.DataFrame]:
    """
    Adjust projection columns for a specific country.
    IMPORTANT: Only secondary columns are adjusted by AI. Tertiary is automatically calculated as: tertiary = 1 - secondary
    This ensures tertiary + secondary always sum to 1 for the same year.
    Returns a tuple of (result_dict, modified_dataframe) where result_dict matches AdjustCountryProjectionsOutput schema.
    """
    try:
        matching_rows = df[df[country_col] == country]

        if len(matching_rows) == 0:
            available_countries = ", ".join(df[country_col].unique().tolist()[:10])
            return {
                "success": False,
                "country": country,
                "adjustments_made": {},
                "error": (
                    f"Error: Country '{country}' not found. \n"
                    f"Available countries (sample): {available_countries}"
                    f"{'...' if len(df[country_col].unique()) > 10 else ''}"
                ),
            }, df

        country_data = matching_rows.iloc[0]
        adjustments_made = {}

        # Process only secondary adjustments (tertiary adjustments are ignored)
        for col, new_value in adjustments.items():
            if col not in projection_cols:
                continue

            if col not in df.columns:
                continue

            # Only process secondary columns - skip tertiary columns
            if col.startswith("tertiary_"):
                logger.warning(
                    f"[WARNING] Tertiary column '{col}' adjustment ignored. Only secondary columns can be adjusted. Tertiary will be auto-calculated."
                )
                continue

            if col.startswith("secondary_"):
                old_value = float(country_data[col])
                df.loc[df[country_col] == country, col] = new_value

                adjustments_made[col] = {
                    "column": col,
                    "old_value": old_value,
                    "new_value": float(new_value),
                    "reason": reason,
                    "auto_adjusted": False,
                }

                # Auto-adjust corresponding tertiary to maintain sum = 1
                # Column format: "secondary_2025" -> year is at index 1
                year = col.split("_")[1]
                tertiary_col = f"tertiary_{year}"

                if tertiary_col in df.columns:
                    new_tertiary = 1.0 - new_value
                    old_tertiary = float(country_data[tertiary_col])
                    df.loc[df[country_col] == country, tertiary_col] = new_tertiary

                    tertiary_reason = (
                        f"Automatically adjusted tertiary_{year} from {old_tertiary:.4f} to {new_tertiary:.4f} "
                        f"to maintain the constraint that secondary_{year} + tertiary_{year} = 1.0. "
                        f"This ensures data consistency after adjusting secondary_{year} to {new_value:.4f}."
                    )

                    adjustments_made[tertiary_col] = {
                        "column": tertiary_col,
                        "old_value": old_tertiary,
                        "new_value": float(new_tertiary),
                        "reason": tertiary_reason,
                        "auto_adjusted": True,
                    }

        if adjustments_made:
            return {"success": True, "country": country, "adjustments_made": adjustments_made, "error": None}, df
        else:
            return {
                "success": False,
                "country": country,
                "adjustments_made": {},
                "error": f"Warning: No valid projection columns found to adjust for {country}",
            }, df

    except Exception as e:
        return {"success": False, "country": country, "adjustments_made": {}, "error": f"Error adjusting projections: {e}"}, df


def analyze_country_individual_row_trend_age_gender(
    df: pd.DataFrame, country: str, country_col: str, individual: str, individual_col: str, projection_cols: list, feedback_retriever=None
) -> dict:
    """
    Analyze one country's and individual's row: for each projection column, compare it to the previous year
    of the same type. If percentage change exceeds 1%, it's an outlier.
    """
    country_individual_rows = df[(df[country_col] == country) & (df[individual_col] == individual)]
    if len(country_individual_rows) == 0:
        return {"error": f"Country '{country}' and individual '{individual}' not found"}

    country_individual_data = country_individual_rows.iloc[0]

    results = {"country": country, "individual": individual, "analyses": {}}

    for proj_col in projection_cols:
        prev_col = str(int(proj_col) - 1)

        try:
            prev_value = pd.to_numeric(country_individual_data[prev_col])
            if pd.isna(prev_value):
                results["analyses"][proj_col] = {"error": f"Previous year value ({prev_col}) is NaN"}
                continue
        except (ValueError, TypeError, KeyError) as e:
            results["analyses"][proj_col] = {"error": f"Could not read previous year value from {prev_col}: {e}"}
            continue

        try:
            current_value = pd.to_numeric(country_individual_data[proj_col])
            if pd.isna(current_value):
                results["analyses"][proj_col] = {"error": f"Projection value is NaN for {proj_col}"}
                continue
        except (ValueError, TypeError, KeyError) as e:
            results["analyses"][proj_col] = {"error": f"Could not read projection value for {proj_col}: {e}"}
            continue

        # calc year-over-year change
        change = current_value - prev_value
        change_pct = (change / prev_value * 100) if prev_value != 0 else 0

        # check if exceeds 1% threshold
        is_outlier = abs(change_pct) > 1.0

        parts = proj_col.split("_")
        col_type = "_".join(parts[1:]) if len(parts) >= 2 else "unknown"

        feedback_text = None
        if feedback_retriever:
            feedback_text = feedback_retriever.get_feedback_age_gender(
                country=country,
                individual=individual,
                column=proj_col,
                projection_value=float(current_value),
                previous_value=float(prev_value) if not pd.isna(prev_value) else None,
            )
            if feedback_text:
                logger.info(f"[FEEDBACK] Retrieved feedback for {country} {individual} {proj_col}: {feedback_text[:100]}...")

        results["analyses"][proj_col] = {
            "type": col_type,
            "previous_year_column": prev_col,
            "previous_year_value": prev_value,
            "current_value": current_value,
            "change": change,
            "change_pct": change_pct,
            "is_outlier": is_outlier,
            "threshold": 1.0,
        }

    return results


def list_countries(df: pd.DataFrame, country_col: str) -> dict:
    """
    Tool to list all available countries in the DataFrame.
    Returns a dictionary matching ListCountriesOutput schema.
    """
    try:
        countries = df[country_col].unique().tolist()
        return {"success": True, "countries": countries, "count": len(countries), "error": None}
    except Exception as e:
        return {"success": False, "countries": [], "count": 0, "error": f"Error listing countries: {e}"}


def list_gender_age_groups(df: pd.DataFrame, individual_col: str) -> dict:
    """
    Tool to list all distinct individuals in the DataFrame.
    Returns a dictionary matching ListIndividualsOutput schema.
    """
    try:
        individuals = df[individual_col].unique().tolist()
        return {"success": True, "individuals": individuals, "count": len(individuals), "error": None}
    except Exception as e:
        return {"success": False, "individuals": [], "count": 0, "error": f"Error listing individuals: {e}"}


def list_all_combinations(df: pd.DataFrame, country_col: str, individual_col: str) -> dict:
    """
    Tool to list all country-individual combinations that need to be processed.
    This helps track progress and ensures all combinations are analyzed.
    Returns a dictionary with structured combination data.
    """
    try:
        countries = df[country_col].unique().tolist()
        individuals = df[individual_col].unique().tolist()
        total_combinations = len(countries) * len(individuals)

        combinations = []
        for i, country in enumerate(countries, 1):
            for j, individual in enumerate(individuals, 1):
                combo_num = (i - 1) * len(individuals) + j
                combinations.append({"number": combo_num, "country": country, "individual": individual})

        return {
            "success": True,
            "total_combinations": total_combinations,
            "countries_count": len(countries),
            "individuals_count": len(individuals),
            "combinations": combinations,
            "error": None,
        }
    except Exception as e:
        return {
            "success": False,
            "total_combinations": 0,
            "countries_count": 0,
            "individuals_count": 0,
            "combinations": [],
            "error": f"Error listing combinations: {e}",
        }


def analyze_country_trend_age_gender(
    df: pd.DataFrame, country: str, country_col: str, individual: str, individual_col: str, projection_cols: list, feedback_retriever=None
) -> dict:
    """
    Analyze one country's and individual's row: for each projection, compare it to the previous year
    of the same type. If change exceeds 1%, it's an outlier.
    Returns a dictionary matching AnalyzeCountryTrendToolOutput schema (pure tool output).
    """
    try:
        trend_info = analyze_country_individual_row_trend_age_gender(
            df, country, country_col, individual, individual_col, projection_cols, feedback_retriever=feedback_retriever
        )

        if "error" in trend_info:
            return {"success": False, "country": country, "individual": individual, "analyses": {}, "outliers": [], "error": trend_info["error"]}

        outliers = []
        analyses = {}

        for proj_col, analysis in trend_info["analyses"].items():
            if "error" in analysis:
                analyses[proj_col] = {
                    "column": proj_col,
                    "type": "unknown",
                    "previous_year_column": "",
                    "previous_year_value": 0.0,
                    "current_value": 0.0,
                    "change": 0.0,
                    "change_pct": 0.0,
                    "is_outlier": False,
                    "threshold": 1.0,
                    "error": analysis["error"],
                }
            else:
                analyses[proj_col] = {
                    "column": proj_col,
                    "type": analysis.get("type", "unknown"),
                    "previous_year_column": analysis.get("previous_year_column", ""),
                    "previous_year_value": float(analysis.get("previous_year_value", 0.0)),
                    "current_value": float(analysis.get("current_value", 0.0)),
                    "change": float(analysis.get("change", 0.0)),
                    "change_pct": float(analysis.get("change_pct", 0.0)),
                    "is_outlier": bool(analysis.get("is_outlier", False)),
                    "threshold": float(analysis.get("threshold", 1.0)),
                    "error": None,
                }
                if analyses[proj_col]["is_outlier"]:
                    outliers.append(proj_col)

        return {"success": True, "country": country, "individual": individual, "analyses": analyses, "outliers": outliers, "error": None}
    except Exception as e:
        return {
            "success": False,
            "country": country,
            "individual": individual,
            "analyses": {},
            "outliers": [],
            "error": f"Error analyzing trend: {e}",
        }


def adjust_country_projections_age_gender(
    df: pd.DataFrame,
    country: str,
    country_col: str,
    individual: str,
    individual_col: str,
    projection_cols: list,
    adjustments: dict,
    reason: str = "",
) -> tuple[dict, pd.DataFrame]:
    """
    Adjust projection columns for a specific country and individual.
    Returns a tuple of (result_dict, modified_dataframe) where result_dict matches AdjustCountryProjectionsOutput schema.
    """
    try:
        matching_rows = df[(df[country_col] == country) & (df[individual_col] == individual)]

        if len(matching_rows) == 0:
            available_countries_individuals = ", ".join((df[country_col] + df[individual_col]).unique().tolist())
            return {
                "success": False,
                "country": country,
                "individual": individual,
                "adjustments_made": {},
                "error": (
                    f"Error: Country '{country}' and Individual '{individual}' not found."
                    f"Available countries-individuals: {available_countries_individuals}"
                ),
            }, df

        country_individual_data = matching_rows.iloc[0]
        adjustments_made = {}

        for col, new_value in adjustments.items():
            if col not in projection_cols:
                continue

            if col not in df.columns:
                continue

            old_value = float(country_individual_data[col])
            df.loc[(df[country_col] == country) & (df[individual_col] == individual), col] = new_value

            adjustments_made[col] = {"column": col, "old_value": old_value, "new_value": float(new_value), "reason": reason, "auto_adjusted": False}

        if adjustments_made:
            return {"success": True, "country": country, "individual": individual, "adjustments_made": adjustments_made, "error": None}, df
        else:
            return {
                "success": False,
                "country": country,
                "individual": individual,
                "adjustments_made": {},
                "error": f"Warning: No valid projection columns found to adjust for {country} and '{individual}'",
            }, df

    except Exception as e:
        return {
            "success": False,
            "country": country,
            "individual": individual,
            "adjustments_made": {},
            "error": f"Error adjusting projections: {e}",
        }, df


def analyze_country_row_trend(
    df: pd.DataFrame, country: str, country_col: str, individual: str, individual_col: str, projection_cols: list, module: str
) -> dict:
    """
    Analyze one country's row: for each projection column, compare it to the previous year
    of the same type. If change exceeds 1%, it's an outlier.
    """
    if module == "education":
        return analyze_country_row_trend_education(df, country, country_col, projection_cols)
    elif module == "age_gender":
        return analyze_country_individual_row_trend_age_gender(df, country, country_col, individual, individual_col, projection_cols)
    else:
        raise ValueError(f"Invalid module: {module}")


def analyze_country_trend(
    df: pd.DataFrame,
    country: str,
    country_col: str,
    individual: str,
    individual_col: str,
    projection_cols: list,
    module: str,
    feedback_retriever=None,
) -> dict:
    """
    Analyze one country's row: for each projection, compare it to the previous year
    of the same type. If change exceeds 1%, it's an outlier.
    Returns a dictionary matching AnalyzeCountryTrendToolOutput schema (pure tool output).
    """
    if module == "education":
        return analyze_country_trend_education(df, country, country_col, projection_cols, feedback_retriever=feedback_retriever)
    elif module == "age_gender":
        return analyze_country_trend_age_gender(
            df, country, country_col, individual, individual_col, projection_cols, feedback_retriever=feedback_retriever
        )
    else:
        raise ValueError(f"Invalid module: {module}")


def adjust_country_projections(
    df: pd.DataFrame,
    country: str,
    country_col: str,
    individual: str,
    individual_col: str,
    projection_cols: list,
    adjustments: dict,
    module: str,
    reason: str = "",
) -> tuple[dict, pd.DataFrame]:
    """
    Adjust projection columns for a specific country and individual.
    Returns a tuple of (result_dict, modified_dataframe) where result_dict matches AdjustCountryProjectionsOutput schema.
    """
    if module == "education":
        return adjust_country_projections_education(df, country, country_col, projection_cols, adjustments, reason)
    elif module == "age_gender":
        return adjust_country_projections_age_gender(df, country, country_col, individual, individual_col, projection_cols, adjustments, reason)
    else:
        raise ValueError(f"Invalid module: {module}")
