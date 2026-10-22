# Quotas Forecasting

This project provides tools and pipelines for quota-based forecasting of internet usage and demographic trends, supporting both Education and Age_Gender modules. It is designed to run both locally in a poetry environment and in production/cloud environments.

## Project Structure

```
quotas_forecasting/
│
├── quotas_forecasting/           # Main package directory
│   ├── main.py                   # Entry point for running forecasting modules
│   │
│   ├── config/                   # Configuration management
│   │   ├── config_loader.py      # Loads configuration files
│   │   ├── config.yaml           # Main configuration file
│   │   └── logging_config.py     # Logging configuration
│   │
│   ├── data_handlers/            # Data loading and processing utilities
│   │   ├── data_validators.py    # Data validation logic
│   │   ├── file_writer.py        # Handles output file writing
│   │   ├── loader.py             # Handles data loading
│   │   ├── missing_data_handlers.py # Handles missing data
│   │   └── proxy_handlers.py     # Handles countries usead as proxy data sources to other countries
│   │
│   ├── fetch_data/               # External data fetching
│   │   └── fetch_data_from_Eurostat.py
│   │
│   ├── forecasting/              # Forecasting logic
│   │   ├── predictors.py         # Forecasting model definitions and utilities
│   │   ├── processors.py         # Data processing and pipeline orchestration
│   │   ├── quotas_forecasting.py # Core forecasting implementation
│   │   └── utils.py              # Forecasting utilities
│   │
│   ├── AI_adjustments/           # LLM-based projection correction (post-processing CSVs)
│   │   ├── main.py               # Standalone CLI for post-processing forecast CSVs
│   │   ├── agent.py              # CorrectionAgent orchestration (tool-calling loop)
│   │   ├── tools.py / prompts.py / rules.py
│   │   └── README.md             # Module developer documentation
│   │
│   ├── services/                 # External service integrations
│   │   ├── message_serializer.py # Message serialization
│   │   └── modelling_files_pb2.py # Protobuf definitions
│   │
│   ├── testing/                  # Output validation tools
│   │   ├── validation_core.py
│   │   ├── validation_ga.py
│   │   └── validation_local.py
│   │
│   ├── Age_Gender/               # Age/Gender module data (input/output)
│   └── Education/                # Education module data (input/output)
│
├── docs/                        # Documentation files
│   ├── AI_PROJECTION_CORRECTIONS.md  # Technical reference: LLM post-processing (AI_adjustments)
│   ├── ARCHITECTURE_RAG.md           # RAG / researcher-feedback architecture for AI_adjustments
│
├── Dockerfile                   # Docker image definition for cloud deployment
├── docker-run.sh                # Script to build and run Docker container
├── requirements.txt             # Python dependencies
├── pyproject.toml               # Project metadata and dependencies (for Poetry)
├── poetry.lock                  # Poetry lock file
├── CODEOWNERS                   # Code ownership definitions
└── README.md                    # Project documentation (this file)
```

## Main Functionality of Each Folder

- **quotas_forecasting/**: Core package containing all main logic for data loading, processing, forecasting, and output. The `main.py` script is the entry point for running forecasts.
  - **config/**: Configuration loading and logging setup.
  - **data_handlers/**: Data loading, validation, file writing, and handling missing data/proxies.
  - **fetch_data/**: Fetching data from external sources (Eurostat).
  - **forecasting/**: Forecasting models, processors, and pipeline orchestration.
  - **AI_adjustments/**: Optional **post-processing** step that uses an LLM with tool calling to review and adjust forecast CSVs (outliers, domain constraints). **Not** invoked currently by the main `main.py` forecasting CLI; run separately. See [docs/AI_PROJECTION_CORRECTIONS.md](docs/AI_PROJECTION_CORRECTIONS.md) and [AI_adjustments/README.md](quotas_forecasting/AI_adjustments/README.md).
  - **services/**: External service integrations (Pub/Sub, protobuf serialization).
  - **testing/**: Output validation tools for local and GA environments.
  - **Age_Gender/** and **Education/**: Module-specific input/output data folders.
- **docs/**: Project documentation (**`AI_PROJECTION_CORRECTIONS.md`**, **`ARCHITECTURE_RAG.md`**, etc.).

## AI projection corrections

The **AI adjustments** subsystem corrects exported **prediction CSVs** using OpenAI (e.g. GPT-4o) and **function calling**: an agent calls registered tools to analyze trends, detect unrealistic changes, and write adjustments while enforcing workflow and validation rules.

| Topic | Where to read |
|--------|----------------|
| Problem, solution, value, features, limitations (technical overview) | [docs/AI_PROJECTION_CORRECTIONS.md](docs/AI_PROJECTION_CORRECTIONS.md) |
| How to run the CLI, layout, components | [quotas_forecasting/AI_adjustments/README.md](quotas_forecasting/AI_adjustments/README.md) |
| Researcher feedback + embeddings (RAG) | [docs/ARCHITECTURE_RAG.md](docs/ARCHITECTURE_RAG.md) |

**Relationship to the main pipeline:** `quotas_forecasting/main.py` runs Eurostat loading and quota forecasting only. **AI_adjustments** is a **separate** CLI: run from the repository root with `python -m quotas_forecasting.AI_adjustments.main ...` (see that README). It loads `config.yaml` for GCP Secret Manager (`secrets.openai_api_key`) like the rest of the project.

**Not the same as** deterministic `age_groups_tweak` in `forecasting/utils.py`, which applies a fixed age-band rule inside the core Age/Gender pipeline.

## Basic Classes

- EurostatFetcher: Low-level API client (utility layer)
- EurostatApp: Business logic processor (service layer)
- EurostatDataFetcher: High-level orchestrator (application layer)
end-to-end flow: `main.py → EurostatDataFetcher → EurostatApp → EurostatFetcher → Eurostat API`

## Running the Main Forecasting Script

You can run the main forecasting pipeline either locally or in production/cloud environments. Example usage:

### Local Execution

- Install poetry
- Install poetry shell with: `poetry self add poetry-plugin-shell`
- Switch to python3.11 if you have another default venv with: `poetry env use python3.11`
- Execute `poetry install` or `- poetry update`
- Switch inside the venv with: `poetry shell`
- Inside the shell, type: `export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcloud/application_default_credentials.json"`
- Copy the correct input files from `ml-quotas-forecasting/Eurostat/` bucket to the same level as the `main.py` file.
- Run:

```sh
python main.py --module both --mode local --pred-year 2026 --last-year 2024
```

- `--module`: Which module to run (`education`, `age_gender`, or `both`).
- `--mode`: Execution mode (`local` or `prod`).
- `--pred-year`: Target year for predictions.
- `--last-year`: Last year of available data.

## Pre-commit Hooks

This project uses pre-commit hooks to ensure code quality and consistency. Hooks automatically format code, check for linting issues, and prevent committing debug statements.

### Setup (One-time per developer)

After cloning the repository and installing dependencies:

### Install dependencies (includes pre-commit)

```sh
poetry install
```

### Set up git hooks (downloads Black, Ruff, etc. automatically)

```sh
poetry run pre-commit install
```

They run automatically on each commit. In case hooks have changed your file you need to stage & commit your file again.

### Execution via Docker (Local & Cloud Shell)

#### 1. Make sure Docker is running

- **Locally**: Launch Docker Desktop
- **In Cloud Shell**: Docker should be available immediately in Cloud Shell.
You can confirm it's ready by running:

```bash
docker --version
```

If you see an error like `Cannot connect to the Docker daemon`or `permission denied`, start it with

  ```bash
  sudo service docker start
  ```

#### 2. Run .`/docker-run.sh` with main args

Run the script with the arguments you use for main. For example:

```bash
./docker-run.sh --module education --mode prod --pred-year 2026 --last-year 2024
  ```

#### The script

- Builds the Docker image (quotas-forecasting)
- Fetches the latest service account key from Secret Manager
- Mounts the key inside the container as a temporary file (No need to manually download or manage credentials)
- Sets the required GOOGLE_APPLICATION_CREDENTIALS env variable
- Runs main.py script with the given arguments

## Modules

- **Education**: Handles forecasting for education-related demographic segments.
- **Age_Gender**: Handles forecasting for age and gender demographic segments.

## Configuration

- Configuration is loaded from YAML files via `config_loader.py`.
- Key parameters include bucket names, project IDs, and validation rules (such as required year difference).

## CI Pipelines (GitHub Actions)

Three GitHub Actions workflows for now, two to validate the forecasting pipeline at different stages and one to trigger a Cloud Run Job.

### 1. Pre-Merge: Docker Build & CLI Check (`pull_request`)

**File:** `.github/workflows/pr-check-docker-build.yml`
**Trigger:** Automatically runs on every **PR targeting** `main`

**What it does:**

- 🐳 Builds the Docker image from the latest main code
- 🚦 Runs `--help` to confirm the CLI starts without errors

This is a fast, lightweight check that prevents broken Dockerfiles or Python startup issues from being merged.

### 2. Post-Merge: Full Pipeline Run + Image Push  (push to main)

**File:** `.github/workflows/post-merge-run-full-pipeline.yml`
**Trigger:**  Runs once after a **PR is** **merged** into `main`

**What it does:**

- 🐳 Builds the Docker image from the latest main code
- 🔐 Authenticates to GCP using the GitHub secret `SA_GWI_HOST_NET`(*)
- 🔑 Fetches the forecasting service account key from Secret Manager
- 🧪 Runs the pipeline using real CLI arguments for both modules
- 🚀 Pushes the built Docker image to`eu.gcr.io/ml-cloud-374512/quotas-forecasting-image` if all  above tests pass

(*) `SA_GWI_HOST_NET`: A GitHub Actions secret that contains the deployer service account key. Used to authenticate GitHub to GCP and fetch the real forecasting SA key from Secret Manager.

### 3. Post-Merge: Trigger Cloud Run Job (push to main)

**File:** `.github/workflows/post-merge-trigger-cloud-run.yml`
**Trigger:** Runs once after a **PR is** **merged** into `main`

**What it does:**

- 🗳️ downloads the repository code to the runner
- 🔒 authenticates with Google Cloud using service account credentials
- 🚀 executes the Cloud Run job and waits for completion

---

For more details on forecasting methodologies and configuration, see the code and comments in the respective modules.

## Modes & flags: quick guide

- **Core CLI flags**:
  - `--mode`: `local` or `prod`
  - `--module`: `education`, `age_gender`, or `both`
  - `--pred-year` and `--last-year`: must satisfy the configured year difference
- **Key config toggles** (in `quotas_forecasting/config/config.yaml`):
  - `defaults.auto_fetch`: whether to auto-fetch Eurostat data
  - `pubsub.enable_notifications`: whether to publish Pub/Sub notifications after uploads

### Run locally (no notifications)

- **CLI**:

```bash
python main.py --module both --mode local --pred-year 2026 --last-year 2024
```

- **Config**:

```yaml
defaults:
  auto_fetch: false
pubsub:
  enable_notifications: false
```

- Notes: Place input CSVs locally as described above; set `GOOGLE_APPLICATION_CREDENTIALS` only if you interact with GCS.

### Run in production without sending notifications

- **CLI**:

```bash
python main.py --module both --mode prod --pred-year 2026 --last-year 2024
```

- **Config**:

```yaml
defaults:
  auto_fetch: true
pubsub:
  enable_notifications: false
```

- Notes: The pipeline fetches Eurostat data and writes to GCS but does not publish Pub/Sub messages.

### Run in production and send notifications

- **CLI**:

```bash
python main.py --module both --mode prod --pred-year 2026 --last-year 2024
```

- **Config**:

```yaml
gcs:
  data_source: "Eurostat" #Should be "Eurostat" for pubsub

defaults:
  auto_fetch: true
pubsub:
  enable_notifications: true
  topic_name: "population-modelling-files-notifications"
  sub_project_id: "<ANDROMEDA_PROJECT_ID>"
  service_account: "<PUBLISHER_SA_EMAIL>"
```

- Notes: Ensure Secret Manager contains the publisher key at `projects/<project_id>/secrets/quotas-forecasting-sa-key`, and `bucket_name_external`/`dir_external` are set for Andromeda.
