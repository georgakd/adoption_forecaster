"""Pipeline hook for the correction agent, behind ``correction.enabled``.

With the flag off nothing here builds a client, a store or a table: ``maybe_correct`` returns its input unchanged.
With the flag on, ``prepare_correction`` builds the LLM client and source store at startup (so a missing ``NVIDIA_API_KEY`` or a
missing store fails before any forecasting), and ``maybe_correct`` corrects one predictor frame. Agent failures never raise: the
baseline, plus any corrections accepted before the failure, is kept and the error is logged (``correction.on_failure``).
"""

import pandas as pd

from adoption_forecaster.config.logging_config import get_logger
from adoption_forecaster.correction.adapters import from_table, to_table
from adoption_forecaster.correction.agent import run_correction
from adoption_forecaster.correction.caps import validate_yoy_cap_settings
from adoption_forecaster.correction.llm import LLMClient, get_llm_client
from adoption_forecaster.correction.sources import SourceStore
from adoption_forecaster.correction.types import correction_record

logger = get_logger(__name__)


class NoSourceStoreError(RuntimeError):
    """Correction is enabled but no real source store exists yet."""


def correction_enabled(config: dict) -> bool:
    return bool(config.get("correction", {}).get("enabled", False))


def get_source_store(config: dict) -> SourceStore:
    raise NoSourceStoreError("no source store is configured; real sources arrive in milestone 2")


def prepare_correction(config: dict, store: SourceStore | None, llm: LLMClient | None) -> tuple[SourceStore | None, LLMClient | None]:
    """At startup: with the flag on, build whatever was not injected (the LLM client first, so a missing key is reported first)."""
    if not correction_enabled(config):
        return store, llm
    validate_yoy_cap_settings(config["correction"]["yoy_cap"])
    if llm is None:
        llm = get_llm_client(config)
    if store is None:
        store = get_source_store(config)
    return store, llm


def maybe_correct(df: pd.DataFrame, module: str, last_year: int, config: dict, store: SourceStore, llm: LLMClient | None = None) -> pd.DataFrame:
    """Return ``df`` unchanged with the flag off; otherwise the agent-corrected frame (same shape, index, columns and dtypes)."""
    if not correction_enabled(config):
        return df
    if llm is None:
        llm = get_llm_client(config)
    try:
        result = run_correction(to_table(df, module, last_year), module, store, llm, config)
        corrected = from_table(result.corrected, template=df, module=module)
    except AssertionError:
        raise  # a too-short ScriptedClient script in a test, not an agent failure (same rule as run_correction)
    except Exception as exc:  # noqa: BLE001 - on_failure: fallback_to_baseline
        logger.error(f"Correction agent failed for {module} ({type(exc).__name__}); keeping the baseline")
        return df
    logger.info(f"Correction {module}: accepted={len(result.corrections)}, rejected={len(result.rejections)}, error={result.error}")
    for correction in result.corrections:
        r = correction_record(correction, module)
        logger.info(
            f"Accepted correction {module} {r['country']}/{r['segment']}/{r['year']}: {r['old_value']:.4f} -> {r['new_value']:.4f}, "
            f"doc_id={r['doc_id']}, confidence={r['confidence']}, quote={r['quote']!r}"
        )
    if result.error:
        kept = len(result.corrections)
        logger.error(f"Correction agent failed for {module}: {result.error}; keeping the baseline plus {kept} accepted correction(s)")
    return corrected if result.corrections else df
