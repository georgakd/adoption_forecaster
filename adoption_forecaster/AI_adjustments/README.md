# AI Adjustments Module

**Hands-on reference** for this package: install, **CLI**, folder layout, **Education / Age_Gender** data expectations, and how to change prompts, rules, or tools.

**Why this exists, trade-offs, and limitations:** [`docs/AI_PROJECTION_CORRECTIONS.md`](../../docs/AI_PROJECTION_CORRECTIONS.md). **Researcher-feedback RAG (embeddings, retrieval):** [`docs/ARCHITECTURE_RAG.md`](../../docs/ARCHITECTURE_RAG.md).

## Overview

The **`main.py` in this directory** drives an OpenAI model (default **GPT-4o**) with **function calling** over your CSV: it inspects projections, flags unrealistic moves using thresholds and wording in **`rules.py`** (for example year-over-year change rules), and applies corrections through **`tools.py`** while **`workflow_enforcer.py`** keeps the agent on sequence. Modules: **`education`** and **`age_gender`**.

## Relationship to the main forecasting pipeline

- **`adoption_forecaster/main.py`** (repo root package) runs **Eurostat fetch + quota forecasting** only. It does **not** call this module.
- **`AI_adjustments/main.py`** is a **separate CLI** for **post-processing CSVs** you place under `input/`. Use it when you want LLM-guided corrections after forecasts exist.

**Not the same as** **`age_groups_tweak`** in **`forecasting/utils.py`**: that function applies a **deterministic** age-band rule inside the core Age/Gender pipeline. This module is an **LLM + tools** post-processor.

## Architecture

The system uses a multi-component architecture:

```
AI_adjustments/
├── main.py              # Entry point and CLI interface
├── agent.py             # CorrectionAgent + apply_corrections()
├── client.py            # OpenAI client wrapper
├── tools.py             # Tool functions for data analysis and adjustments
├── tool_registry.py     # Registers tools and binds handlers
├── schemas.py           # Pydantic schemas for validation
├── prompts.py           # System prompts for AI agent
├── rules.py             # Correction rules and instructions
├── workflow_enforcer.py # Workflow ordering / violations
├── message_history.py   # Conversation state for the agent loop
├── feedback_rag.py      # Optional researcher-feedback RAG
├── gcp_secrets.py       # OpenAI API key from Secret Manager
├── utils.py             # Projection columns, tool schemas, helpers
├── input/               # Input CSV files (+ optional feedback CSV)
└── output/              # Output CSV files with corrections + logs
```

## Installation

### Prerequisites

- **Python 3.11** (aligned with repo `pyproject.toml`)
- **GCP access** to the Secret Manager secret configured in `adoption_forecaster/config/config.yaml` (`secrets.openai_api_key`)
- OpenAI API usage (via that key)

### Setup

1. From the **repository root**, install the main project with Poetry (`poetry install`). `openai` and `pydantic` are regular dependencies in `pyproject.toml`; no separate requirements file is needed.

1. **API key:** Loaded at runtime from **GCP Secret Manager** (see below). A `.env` `OPENAI_API_KEY` is **not** read by the current `main.py` path.

2. Prepare input data: place prediction CSV files in **`input/`** under the directory you run the CLI from (paths are relative to the current working directory; typically the repository root).

### OpenAI API key (local and prod)

The AI adjustments module needs an OpenAI API key for the correction agent and optional RAG feedback.

- **Prod and local mode**  
  The key is always loaded from **GCP Secret Manager**. The secret name is set in `config.yaml` under `secrets.openai_api_key` (e.g. `adoption-forecaster-openai-api-key`). The project is taken from `defaults.project_id`.  
  At runtime, `get_openai_api_key(config)` is called and the key is used by `AIClient` and `FeedbackRAGRetriever`. For this to work in prod (or when running “locally” against GCP), the environment must have credentials that can access Secret Manager (e.g. the forecasting service account or Application Default Credentials).

- **Local development**  
  In the current implementation, local runs use the same GCP Secret Manager path as prod. Ensure you are authenticated (e.g. `gcloud auth application-default login`) and that your identity can access the secret in the configured project.

## Usage

### Command Line Interface

Run the CLI as a module from the **repository root**:

```bash
poetry run python -m adoption_forecaster.AI_adjustments.main --module <module> --mode <mode> --adjusted-year <year>
```

If you do not use Poetry, activate a venv with the project dependencies installed and run `python -m adoption_forecaster.AI_adjustments.main ...` instead.

#### Arguments

- `--module` (**required**): `education` | `age_gender`
  - `education`: secondary/tertiary projections (**only secondary** is adjusted by the agent; tertiary is derived)
  - `age_gender`: country × individual (age/gender) projections

- `--mode` (**required**): `local` | `prod` (reserved for future behavior; pass either value today)

- `--adjusted-year` (**required**): target forecast year, e.g. `2026`

#### Example

```bash
poetry run python -m adoption_forecaster.AI_adjustments.main --module education --mode local --adjusted-year 2026
```

### Workflow

1. **File Selection**: The script lists available CSV files in `input/` and prompts for selection
2. **Data Loading**: Loads the selected CSV file and identifies projection columns
3. **Rules Application**: Uses rules from `rules.py` (or custom instructions if provided)
4. **AI Processing**: The AI agent systematically processes each entity:
   - Analyzes trends for outliers
   - Applies corrections when needed
   - Maintains workflow constraints
5. **Output**: Saves corrected data to `output/` directory and may write **`adjustment_log_<stem>.json`**

## Module Structure

Concrete **constraints and column names** below; **design rationale** (problem/solution) is in [`docs/AI_PROJECTION_CORRECTIONS.md`](../../docs/AI_PROJECTION_CORRECTIONS.md).

### Education Module

Processes country-level projections with tertiary and secondary education levels.

**Constraints:**

- For each year, **`secondary_<year>` + `tertiary_<year>` = 1.0**
- The agent must **only propose changes to `secondary_*` projection columns**. **`tools.py`** ignores direct tertiary adjustments and sets **`tertiary_<year> = 1.0 - secondary_<year>`** after a secondary change.
- Rules in **`rules.py`** require **sequential** year fixes (e.g. stabilize `secondary_2025` before `secondary_2026` relative to the updated baseline).

**Projection Columns:**

- `secondary_<year>` and `tertiary_<year>` (e.g. `secondary_2026`, `tertiary_2026`) — as detected by `get_projection_columns()`; the agent adjusts **secondary** only.

### Age/Gender Module

Processes country-individual combination projections.

**Constraints:**

- Projections must remain within [0, 1] bounds
- Conservative downward adjustments preferred

**Projection Columns:**

- `{year}` format
- Example: `2026`, `2025`

## Key Components

### `agent.py`

- **`CorrectionAgent`**: Orchestrates the tool-calling loop, message history, and workflow enforcement for a given `module` (`education` or `age_gender`).
- **`apply_corrections(...)`**: Public entry used by `main.py`; constructs `CorrectionAgent` and returns **`(df_working, adjustment_log)`**.

**Features:**

- Workflow enforcement (e.g. one function call per response)
- Pending adjustment tracking
- Progress monitoring
- **`max_iterations`** cap to avoid infinite loops

### `client.py`

OpenAI client wrapper (`AIClient`):

- `create_response()`: Creates API responses
- `process_response_output()`: Processes response items
- `limit_function_calls()`: Enforces function call limits to avoid bypassing any adjustments required

### `tools.py`

Tool functions available to the AI agent:

- `list_countries()`: Lists available countries
- `list_gender_age_groups()`: Lists available individuals (age/gender)
- `list_all_combinations()`: Lists all country-individual combinations
- `analyze_country_trend()`: Analyzes trends and identifies outliers
- `adjust_country_projections()`: Applies corrections to projections

### `schemas.py`

Pydantic schemas for validation:

- Input validation for tool calls
- Output schemas for tool responses
- Type safety and error handling

### `prompts.py`

System prompts for the AI agent:

- `get_education_system_prompt()`: Education module system prompt
- `get_age_gender_system_prompt()`: Age/Gender module system prompt

Contains detailed workflow instructions, constraints, and task descriptions.

### `rules.py`

Correction rules and instructions:

- `get_education_rules()`: Education module rules
- `get_age_gender_rules()`: Age/Gender module rules
- `get_rules()`: Unified rule getter

**Customization:**
Edit `rules.py` to modify correction behavior without changing core logic.

### `utils.py`

Utility functions:

- `get_projection_columns()`: Identifies projection columns
- `get_historical_columns()`: Identifies historical data columns
- `create_tools_schema()`: Creates tool schemas for function calling

### `tool_registry.py`

Centralized tool registry for managing all tools:

- `ToolRegistry`: Class for registering and managing tools
- `register_all_tools()`: Registers all available tools with their handlers
- `get_registry()`: Gets the global registry instance

**Adding a New Tool:**

1. Create the tool function in `tools.py`
2. Create input/output schemas in `schemas.py`
3. Add handler method to `CorrectionAgent` class
4. Register the tool in `register_all_tools()` function in `tool_registry.py`
5. Add tool schema to `create_tools_schema()` in `utils.py`

## Changing prompts and rules

When editing `rules.py` or `prompts.py`, follow the **Prompt & Rules Change Guide** so that agent behavior, invariants, and tool-call schemas are preserved.

- **Location:** [`.cursor/rules/prompt-rules-change-guide.mdc`](../../.cursor/rules/prompt-rules-change-guide.mdc) (in the repo root)
- **Scope:** Applies to `rules.py` and `prompts.py` only; excludes refactors unrelated to agent behavior, dependency changes, and tool implementation.

It is applied manually, which means, it is explicitly called in the cursor agent chat prompt, like "“Apply all active MDC rules (to the active files).”"

## Configuration

### Custom Rules

Modify `rules.py` to customize correction behavior, e.g. edit the string returned by `get_education_rules()` / `get_age_gender_rules()`, or extend `get_rules(module, count=None)`.

### Custom Instructions

You can provide custom instructions at runtime:

- When prompted, enter custom instructions
- Or press Enter to use default rules from `rules.py`

## Output

### Output Files

Corrected data is saved to `output/` directory with prefix `modified_`:

- Original: `input/2026_Education_16-74_predictions.csv`
- Output: `output/modified_2026_Education_16-74_predictions.csv`

### Output Validation

The system automatically validates:

- Historical data remains unchanged
- Only projection columns are modified
- Changes are logged with before/after values

## Workflow Rules

The AI agent follows strict workflow rules:

1. **One Function Call Per Response**: Prevents parallel processing issues
2. **Sequential Processing**: Processes entities one at a time
3. **Complete Analysis**: Must analyze all entities before concluding
4. **Pending Adjustments**: Cannot analyze new entity if previous has pending adjustments
5. **Workflow Enforcement**: System enforces rules with warnings and messages

## Error Handling

- **Validation**: Tool call inputs are validated using Pydantic schemas
- **Workflow Violations**: Detected and corrected with enforcement messages
- **API Errors**: Handled with error messages
- **Data Integrity**: Historical data is protected from modifications

## Dependencies

- `pandas>=2.0.0`: Data manipulation
- `numpy>=1.24.0`: Numerical operations
- `openai>=1.0.0`: OpenAI API client

## Notes

- The system uses OpenAI's `responses.create()` API (not standard chat completions)
- Function calling is used for structured tool interactions
- Maximum iterations are limited to prevent infinite loops
- All corrections are logged for audit purposes
