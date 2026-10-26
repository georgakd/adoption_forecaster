"""
Utility functions for data processing
"""

import logging
import pandas as pd
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def get_projection_columns(df: pd.DataFrame, module: str, adjusted_year: int) -> list:
    """Identify columns that are projections (containing adjusted_year or adjusted_year - 1)."""
    if module == "age_gender":
        projection_columns = [f"{adjusted_year}", f"{adjusted_year - 1}"]
    elif module == "education":
        projection_columns = [
            f"secondary_{adjusted_year}",
            f"secondary_{adjusted_year - 1}",
            f"tertiary_{adjusted_year}",
            f"tertiary_{adjusted_year - 1}",
        ]
    else:
        raise ValueError(f"Invalid module: {module}")

    return projection_columns


def get_historical_columns(df: pd.DataFrame, country_col: str, individual_col: str, projection_columns: list) -> list:
    """Get historical columns (non-projection, non-country, non-individual columns)."""
    return [col for col in df.columns if col != country_col and col != individual_col and col not in projection_columns]


def create_tools_schema_age_gender(projection_columns: list, country_col: str, individual_col: str, adjusted_year: int) -> list:
    """Create tool schemas with projection column and country information."""
    projection_cols_str = ", ".join(projection_columns)  # noqa: F841

    return [
        {
            "type": "function",
            "name": "list_countries",
            "description": "List all available countries in the dataset.",
            "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        },
        {
            "type": "function",
            "name": "list_individuals",
            "description": "List all distinct gender-age groups in the dataset.",
            "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        },
        {
            "type": "function",
            "name": "list_all_combinations",
            "description": "List all country-individual combinations that need to be processed.",
            "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        },
        {
            "type": "function",
            "name": "analyze_country_trend",
            "description": "Analyze ONE country's and individual's row: for each projection column, compare it to the previous year of the same type.",
            "parameters": {
                "type": "object",
                "properties": {
                    "country": {"type": "string", "description": "The country name"},
                    "individual": {"type": "string", "description": "The individual name"},
                },
                "required": ["country", "individual"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "adjust_country_projections",
            "description": "Adjust projection columns for ONE country and individual.",
            "parameters": {
                "type": "object",
                "properties": {
                    "country": {
                        "type": "string",
                        "description": "The country name.",
                    },
                    "individual": {
                        "type": "string",
                        "description": "The individual name.",
                    },
                    "adjustments": {
                        "type": "object",
                        "description": ("Dictionary mapping projection column names to new numeric values between 0 and 1. "),
                        "additionalProperties": {"type": "number"},
                    },
                    "reason": {
                        "type": "string",
                        "description": (
                            "Detailed reason for adjustments including: why each specific value was chosen, how feedback was considered (if provided), how historical trends influenced decisions, and the calculation logic behind the values."
                            "Example: 'Adjusting 2025 from 0.65 to 0.538 because feedback suggests conservative approach after 6pp increase in 2023."
                            "Historical avg YoY change is 2%, so reducing to align with trend. 2026 set to 0.598 (1.1% increase from adjusted 2025) to maintain realistic progression."
                        ),
                    },
                },
                "required": ["country", "individual", "adjustments", "reason"],
                "additionalProperties": False,
            },
        },
    ]


def create_tools_schema_education(projection_columns: list, country_col: str, adjusted_year: int) -> list:
    """
    Create tool schemas with projection column and country information.
    """
    projection_cols_str = ", ".join(projection_columns)  # noqa: F841

    return [
        {
            "type": "function",
            "name": "list_countries",
            "description": "List all available countries in the dataset.",
            "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        },
        {
            "type": "function",
            "name": "analyze_country_trend",
            "description": "Analyze ONE country's row: for each projection column, compare it to the previous year of the same type.",
            "parameters": {
                "type": "object",
                "properties": {"country": {"type": "string", "description": "The country name."}},
                "required": ["country"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "adjust_country_projections",
            "description": ("Adjust projection columns for ONE country."),
            "parameters": {
                "type": "object",
                "properties": {
                    "country": {"type": "string", "description": "The country name."},
                    "adjustments": {
                        "type": "object",
                        "description": (" Dictionary mapping projection column names to new numeric values. "),
                        "additionalProperties": {"type": "number"},
                    },
                    "reason": {
                        "type": "string",
                        "description": (
                            "Detailed reason for adjustments including: why each specific value was chosen,/how feedback was considered (if provided), how historical trends influenced decisions,"
                            "and the calculation logic behind the values. Example: 'Adjusting 2025 from 0.65 to 0.538 because feedback suggests conservative approach after 6pp increase in 2023."
                            "Historical avg YoY change is 2%, so reducing to align with trend. 2026 set to 0.598 (1.1% increase from adjusted 2025) to maintain realistic progression."
                        ),
                    },
                },
                "required": ["country", "adjustments", "reason"],
                "additionalProperties": False,
            },
        },
    ]


def create_tools_schema(projection_columns: list, country_col: str, individual_col: str, adjusted_year: int, module: str) -> list:
    """Create tool schemas based on module."""
    if module == "education":
        return create_tools_schema_education(projection_columns, country_col, adjusted_year)
    elif module == "age_gender":
        return create_tools_schema_age_gender(projection_columns, country_col, individual_col, adjusted_year)
    else:
        raise ValueError(f"Invalid module: {module}")


def validate_historical_data(df: pd.DataFrame, df_modified: pd.DataFrame, historical_columns: list) -> None:
    """Verify that historical data was not modified."""
    if not historical_columns:
        return
    historical_unchanged = all(df[col].equals(df_modified[col]) for col in historical_columns)
    if not historical_unchanged:
        logger.warning("\n[WARNING] Historical data appears to have been modified!")
        logger.info("   This should not happen. Please review the output file carefully.")
    else:
        logger.info("\n[OK] Verified: Historical data unchanged")


def log_changes_for_country(df: pd.DataFrame, df_modified: pd.DataFrame, col: str, country: str, country_col: str) -> None:
    """Log changes for a specific country (education module)."""
    country_rows = df[df[country_col] == country]
    if len(country_rows) == 0:
        return

    old_val = country_rows[col].iloc[0]
    new_val = df_modified[df_modified[country_col] == country][col].iloc[0]
    logger.info(f"    • {country}: {old_val:.4f} -> {new_val:.4f}")


def log_changes_for_combination(
    df: pd.DataFrame, df_modified: pd.DataFrame, col: str, country: str, individual: str, country_col: str, individual_col: str
) -> None:
    """Log changes for a specific country-individual combination (age_gender module)."""
    country_rows = df[df[country_col] == country]
    if len(country_rows) == 0:
        return

    old_val = country_rows[col].iloc[0]
    new_val = df_modified[(df_modified[country_col] == country) & (df_modified[individual_col] == individual)][col].iloc[0]
    logger.info(f"    • {country}-{individual}: {old_val:.4f} -> {new_val:.4f}")


def report_changes(
    df: pd.DataFrame, df_modified: pd.DataFrame, projection_columns: list, country_col: str, individual_col: Optional[str], module: str
) -> None:
    """Report all changes made to projection columns."""
    if not projection_columns:
        return

    logger.info("\n[INFO] Changes made (projection columns only):")
    changes_found = False

    for col in projection_columns:
        if col not in df.columns or col not in df_modified.columns:
            continue

        if df[col].equals(df_modified[col]):
            continue

        diff_count = (df[col] != df_modified[col]).sum()
        if diff_count == 0:
            continue

        changes_found = True
        logger.info(f"\n  - {col}: {diff_count} value(s) modified")

        changed_mask = df[col] != df_modified[col]
        changed_countries = df.loc[changed_mask, country_col].unique().tolist()

        if module == "age_gender":
            changed_individuals = df.loc[changed_mask, individual_col].unique().tolist()
            for country in changed_countries:
                for individual in changed_individuals:
                    log_changes_for_combination(df, df_modified, col, country, individual, country_col, individual_col)
        elif module == "education":
            for country in changed_countries:
                log_changes_for_country(df, df_modified, col, country, country_col)

    if not changes_found:
        logger.info("  No changes were made to projection columns.")


def load_feedback(file_path: str, module: str) -> dict:
    """
    Load feedback about ML projections.
    For education: Only secondary has feedback. Tertiary is auto-calculated.
    For age_gender: Country-individual combinations have feedback.
    Returns:
    Education: {country: {"secondary": {"comments": "...", "adjusted_value": ...}}}
    Age_gender: {country: {"age_gender": {individual: {"comments": "...", "adjusted_value": ...}}}}
    """
    # Try different encodings and CSV parsing options to handle various formats
    encodings = ["utf-8", "latin-1", "iso-8859-1", "cp1252"]
    df = None
    encoding_used = None

    for encoding in encodings:
        try:
            # Try with proper CSV parsing (handles quoted fields with commas)
            df = pd.read_csv(
                file_path,
                encoding=encoding,
                quotechar='"',
                skipinitialspace=True,
                on_bad_lines="skip",  # Skip malformed lines instead of failing
                engine="python",  # More lenient Python parser
            )
            encoding_used = encoding
            logger.info(f"[FEEDBACK] Successfully loaded feedback CSV using {encoding} encoding")
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.debug(f"[FEEDBACK] Failed to load with {encoding} using standard parser: {e}")
            # Try with even more lenient parsing - just skip bad lines
            try:
                df = pd.read_csv(file_path, encoding=encoding, on_bad_lines="skip", engine="python", sep=",", quotechar='"')
                encoding_used = encoding  # noqa: F841
                logger.warning(f"[FEEDBACK] Loaded feedback CSV with lenient parsing using {encoding} encoding. Some rows may have been skipped.")
                break
            except Exception as e2:
                logger.debug(f"[FEEDBACK] Failed to load with {encoding} using lenient parser: {e2}")
                continue

    if df is None:
        raise ValueError(f"Could not decode/parse feedback CSV file {file_path} with any of the attempted encodings: {encodings}")

    feedback_data = {}
    countries_with_feedback = 0

    # Try to find column names (case-insensitive and handle variations)
    country_col = None
    individual_col = None
    education_col = None
    comments_col = None
    manual_adj_col = None

    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in ["country", "countries", "country_name"]:
            country_col = col
        elif col_lower in ["individual", "individuals", "age_gender", "age_group", "gender"]:
            individual_col = col
        elif col_lower in ["education", "education_level", "level", "edu_level"]:
            education_col = col
        elif col_lower in ["comments", "comment", "feedback", "notes", "note"]:
            comments_col = col
        elif col_lower in ["manual adjustment", "manual_adjustment", "adjustment", "manual_adj", "adjusted_value"]:
            manual_adj_col = col

    if not country_col:
        available_cols = ", ".join(df.columns.tolist())
        raise ValueError(f"Could not find 'Country' column in feedback CSV. Available columns: {available_cols}")

    if not education_col:
        logger.warning("[FEEDBACK] Could not find 'Education' column. Assuming all rows are Secondary.")
        education_col = None

    if not comments_col:
        logger.warning(f"[FEEDBACK] Could not find 'Comments' column. Available columns: {', '.join(df.columns.tolist())}")
        comments_col = None

    for _, row in df.iterrows():
        try:
            country = str(row[country_col]).strip() if country_col else None
            if not country or country == "nan":
                continue

            comments = str(row.get(comments_col, "")).strip() if comments_col else ""
            manual_adj = row.get(manual_adj_col, None) if manual_adj_col else None

            adjusted_value = None
            if manual_adj is not None and pd.notna(manual_adj):
                try:
                    adjusted_value = float(manual_adj)
                except (ValueError, TypeError):
                    adjusted_value = None

            if module == "education":
                # education logic
                education_level = str(row[education_col]).strip() if education_col else "Secondary"
                if education_level.lower() in ["secondary", "2", "2.0"] or education_col is None:
                    if country not in feedback_data:
                        feedback_data[country] = {}
                        countries_with_feedback += 1
                    feedback_data[country]["secondary"] = {"comments": comments, "adjusted_value": adjusted_value}

            elif module == "age_gender":
                # age-gender logic
                individual = str(row[individual_col]).strip() if individual_col else None
                if not individual or individual == "nan":
                    continue

                if country not in feedback_data:
                    feedback_data[country] = {}
                    countries_with_feedback += 1

                if "age_gender" not in feedback_data[country]:
                    feedback_data[country]["age_gender"] = {}

                feedback_data[country]["age_gender"][individual] = {"comments": comments, "adjusted_value": adjusted_value}

                if comments and comments != "OK" and comments.lower() != "nan":
                    logger.info(
                        f"[FEEDBACK] Loaded feedback for {country} {individual}: {comments[:80]}..."
                        if len(comments) > 80
                        else f"[FEEDBACK] Loaded feedback for {country} {individual}: {comments}"
                    )

        except Exception as e:
            logger.warning(f"[FEEDBACK] Error processing row {_}: {e}. Skipping.")
            continue

    logger.info(f"[FEEDBACK] Loaded feedback data for {countries_with_feedback} countries from {file_path}")
    return feedback_data


def get_feedback_csv_path(input_dir: Path, module: str, adjusted_year: int) -> Optional[Path]:
    if module in ("education", "age_gender"):
        return input_dir / f"feedback_{adjusted_year - 1}_{module}.csv"
    logger.warning("Provided module does not exist. Module should be age_gender or education.")
    return None
