# AI projection corrections (`AI_adjustments`)

Conceptual and architectural reference for the **LLM-based post-processing** package under `adoption_forecaster/AI_adjustments/`.

**Operational detail** (exact CLI commands, install steps, directory tree, per-file APIs, adding tools): **[`adoption_forecaster/AI_adjustments/README.md`](../README.md)** — that README is written to stand alone next to the code.

## Summary

The **AI adjustments** module provides a **`CorrectionAgent`** that post-processes **forecast CSVs** using OpenAI’s **Responses API** and **tool (function) calling**. The model does not edit files directly; it calls validated tools that read and update a `pandas` DataFrame under rules in **`prompts.py`**, **`rules.py`**, and **`workflow_enforcer.py`**.

---

## Problem

1. **Model outputs** (internet penetration / quota-style projections) can show **unrealistic year-over-year moves**, violate **domain constraints** (e.g. education levels must sum to 1), or drift from **historical shape**.
2. **Manual review** by researchers does not scale across countries and segments.
3. The business needs **repeatable**, **logged** adjustments that still allow **expert feedback** to influence decisions.

---

## Solution

- **Agent loop:** The LLM receives a system prompt and optional user instructions, then repeatedly calls tools such as listing entities, **analyzing** trends (including outlier flags), and **adjusting** projections.
- **Tool boundary:** All mutations go through **`tools.py`** (e.g. `adjust_country_projections`), which can **enforce invariants** (e.g. for Education, only **`secondary_*`** columns are changed; **`tertiary_*`** is set to `1 - secondary`).
- **Validation:** **Pydantic** schemas in **`schemas.py`** validate tool arguments and outputs.
- **Workflow:** **`workflow_enforcer.py`** and **`message_history.py`** enforce ordering (e.g. one function call per model response, complete coverage before finish).
- **Optional RAG:** Only if a file exists at the **exact path** resolved by **`get_feedback_csv_path()`** in **`utils.py`**: **`AI_adjustments/input/feedback_{adjusted_year - 1}_{module}.csv`** (e.g. `--adjusted-year 2026` and `--module education` → `input/feedback_2025_education.csv`). **`main.py`** does not scan `input/` for arbitrary names; any other filename is ignored and the run **continues without researcher feedback** while still adjusting projections (see **Limitations**). When loaded, **`FeedbackRAGRetriever`** (`feedback_rag.py`) embeds comments (**OpenAI `text-embedding-3-small`**) and retrieves on **exact match** or **cosine similarity** above a threshold (see **[ARCHITECTURE_RAG.md](./ARCHITECTURE_RAG.md)**; default similarity gate in code is **0.75**).

**High-level flow**

```
CSV → AI_adjustments/main.py → load_config + Secret Manager (OpenAI key)
     → CorrectionAgent.run() ↔ OpenAI (tools)
     → modified CSV + optional adjustment_log JSON
```

---

## Value

| Benefit | Notes |
|--------|--------|
| **Speed** | Automates first-pass review and correction of many rows. |
| **Consistency** | Same prompts/rules applied across runs (modulo LLM sampling). |
| **Auditability** | Changes can be traced via adjustment logging and before/after reporting. |
| **Expert grounding** | RAG surfaces historical researcher comments when relevant. |
| **Separation of concerns** | Policy lives in `rules.py` / `prompts.py`; mechanics in agent/tools/schemas. |

---

## Functionalities

### Modes / modules

- **`education`**: Country-level **secondary / tertiary** projections. **Only `secondary_<year>`** is adjusted by the agent; **`tertiary_<year>`** is derived so **secondary + tertiary = 1** per year. **Column names, sequential-year rules, and tool behavior** are documented under **Module Structure** in the [module README](../README.md#module-structure).
- **`age_gender`**: Country × individual (age/gender) combinations; projections stay in **[0, 1]**; rules favor **conservative downward** adjustments. **Projection column naming** for this module is in the same README section.

### CLI (standalone)

Entry point **`python -m adoption_forecaster.AI_adjustments.main`** (not **`adoption_forecaster/main.py`**). Flags: `--module`, `--mode`, `--adjusted-year`; interactive CSV selection; outputs under `output/` including optional **`adjustment_log_*.json`**. **Copy-paste commands and argument reference:** [module README — Usage](../README.md#usage).

### Configuration & secrets

- **`adoption_forecaster/config/config.yaml`**: `defaults.project_id`, **`secrets.openai_api_key`** (Secret Manager secret name).
- Runtime uses **`get_openai_api_key`** (`gcp_secrets.py`); callers need credentials that can read Secret Manager (e.g. ADC or service account), unless `OPENAI_API_KEY` is exported, in which case it takes precedence and Secret Manager is not contacted. **Local vs prod notes:** [module README — OpenAI API key](../README.md#openai-api-key-local-and-prod).

### Researcher feedback file (RAG)

**`main.py`** loads feedback only from:

`adoption_forecaster/AI_adjustments/input/feedback_{adjusted_year - 1}_{module}.csv`

| `--adjusted-year` | `--module`   | Expected feedback filename      |
|-------------------|--------------|----------------------------------|
| `2026`            | `education`  | `feedback_2025_education.csv`    |
| `2026`            | `age_gender` | `feedback_2025_age_gender.csv` |

Implementation: **`get_feedback_csv_path(input_dir, module, adjusted_year)`** in **`utils.py`**; check in **`main.py`** (~line 162). If that path is missing, the process logs that no feedback file was found and **proceeds without RAG**—corrections still run, but **without** embedded researcher comments, which can **materially change** outcomes compared to a run with feedback. Rename or copy exports to match this pattern when you intend to use feedback.

### Source layout

Files live under **`adoption_forecaster/AI_adjustments/`**. **Directory tree, roles, and how to add a tool** are maintained in one place: [module README — Architecture and Key Components](../README.md#architecture).

### Changing behavior safely

- Edits to **`rules.py`** or **`prompts.py`** should follow the repo **Prompt & Rules Change Guide**:  
  **`.cursor/rules/prompt-rules-change-guide.mdc`**

---

## Limitations

1. **LLM behavior:** Non-deterministic (unless temperature and sampling are tightly controlled); may misinterpret edge cases; **cost and latency** scale with iterations and data size.
2. **Not a forecaster:** This module **does not** replace `adoption_forecaster`’s statistical forecasting; it **adjusts supplied CSVs** after the fact.
3. **Not wired into default forecasting `main.py`:** The core pipeline does **not** call `AI_adjustments`; operations must run the **separate CLI** (or integrate explicitly if product requires it).
4. **Dependencies:** OpenAI and pydantic are declared in the main **`pyproject.toml`**; no separate install step is needed.
5. **RAG quality:** Retrieval depends on feedback CSV coverage, embedding quality, and the **similarity threshold**; no semantic match ⇒ no comment for that lookup (graceful degradation).
6. **Feedback filename:** A misnamed feedback file (anything other than **`feedback_{adjusted_year - 1}_{module}.csv`** under **`input/`**) is **not** loaded; the CLI **does not fail** and may look like a normal run. Always confirm the log (e.g. feedback loaded vs “No feedback file found”) or the exact path above.
7. **Safety caps:** **`max_iterations`** in the agent prevents infinite loops; very large datasets may hit limits or require batching strategies (not built-in).
8. **Distinct from `age_groups_tweak`:** Deterministic **`age_groups_tweak`** in **`forecasting/utils.py`** applies a **rule-based** age-band fix **inside** the Age/Gender pipeline; it is **not** this LLM post-processor.

---

## Related documentation

| Doc | Purpose |
|-----|---------|
| [AI_adjustments/README.md](../README.md) | Install, CLI, directory tree, per-file APIs, module constraints, extending tools |
| [ARCHITECTURE_RAG.md](./ARCHITECTURE_RAG.md) | Feedback embeddings and retrieval |
| [ARCHITECTURE.md](../../../docs/ARCHITECTURE.md) | Main service architecture (includes optional `AI_adjustments`) |
| [DESIGN_PATTERNS.md](../../../docs/DESIGN_PATTERNS.md) | Broader codebase patterns; links back here for the LLM module |
| [README.md](../../../README.md) | Repo overview → **AI projection corrections** |

---

## Document history

| Version | Date | Notes |
|---------|------|--------|
| 1.0 | 2026-03-19 | Initial in-repo technical reference |
| 1.1 | 2026-03-19 | De-duplicate file/CLI detail; defer to module README |
| 1.2 | 2026-03-19 | Document exact feedback CSV path; silent no-RAG if misnamed |
