# Adoption Forecaster

This project provides tools and pipelines for quota-based forecasting of internet usage and demographic trends, supporting both Education and Age_Gender modules. It is designed to run both locally in a poetry environment and in production/cloud environments.

## Project Structure

```
adoption_forecaster/
│
├── adoption_forecaster/           # Main package directory
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
│   │   ├── adoption_forecaster.py # Core forecasting implementation
│   │   └── utils.py              # Forecasting utilities
│   │
│   ├── AI_adjustments/           # LLM-based projection correction (post-processing CSVs)
│   │   ├── main.py               # Standalone CLI for post-processing forecast CSVs
│   │   ├── agent.py              # CorrectionAgent orchestration (tool-calling loop)
│   │   ├── tools.py / prompts.py / rules.py
│   │   ├── README.md             # Module developer documentation
│   │   ├── docs/                 # AI_PROJECTION_CORRECTIONS.md (technical reference), ARCHITECTURE_RAG.md (researcher-feedback RAG)
│   │   └── tests/                # Import and CLI smoke tests
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
│
├── Dockerfile                   # Docker image definition for cloud deployment
├── docker-run.sh                # Script to build and run Docker container
├── requirements.txt             # Python dependencies
├── pyproject.toml               # Project metadata and dependencies (for Poetry)
├── poetry.lock                  # Poetry lock file
└── README.md                    # Project documentation (this file)
```

## Main Functionality of Each Folder

- **adoption_forecaster/**: Core package containing all main logic for data loading, processing, forecasting, and output. The `main.py` script is the entry point for running forecasts.
  - **config/**: Configuration loading and logging setup.
  - **data_handlers/**: Data loading, validation, file writing, and handling missing data/proxies.
  - **fetch_data/**: Fetching data from external sources (Eurostat).
  - **forecasting/**: Forecasting models, processors, and pipeline orchestration.
  - **AI_adjustments/**: Optional **post-processing** step that uses an LLM with tool calling to review and adjust forecast CSVs (outliers, domain constraints). **Not** invoked currently by the main `main.py` forecasting CLI; run separately. See [adoption_forecaster/AI_adjustments/docs/AI_PROJECTION_CORRECTIONS.md](adoption_forecaster/AI_adjustments/docs/AI_PROJECTION_CORRECTIONS.md) and [AI_adjustments/README.md](adoption_forecaster/AI_adjustments/README.md).
  - **services/**: External service integrations (Pub/Sub, protobuf serialization).
  - **testing/**: Output validation tools for local and GA environments.
  - **Age_Gender/** and **Education/**: Module-specific input/output data folders.
- **docs/**: Project documentation (architecture, design patterns, maths review, correction agent). The `AI_adjustments` docs live in `adoption_forecaster/AI_adjustments/docs/`.

## AI projection corrections

The **AI adjustments** subsystem corrects exported **prediction CSVs** using OpenAI (e.g. GPT-4o) and **function calling**: an agent calls registered tools to analyze trends, detect unrealistic changes, and write adjustments while enforcing workflow and validation rules.

| Topic | Where to read |
|--------|----------------|
| Problem, solution, value, features, limitations (technical overview) | [adoption_forecaster/AI_adjustments/docs/AI_PROJECTION_CORRECTIONS.md](adoption_forecaster/AI_adjustments/docs/AI_PROJECTION_CORRECTIONS.md) |
| How to run the CLI, layout, components | [adoption_forecaster/AI_adjustments/README.md](adoption_forecaster/AI_adjustments/README.md) |
| Researcher feedback + embeddings (RAG) | [adoption_forecaster/AI_adjustments/docs/ARCHITECTURE_RAG.md](adoption_forecaster/AI_adjustments/docs/ARCHITECTURE_RAG.md) |
| Autonomous correction agent (cited evidence, deterministic gate, synthetic benchmark runner) | [docs/CORRECTION_AGENT.md](docs/CORRECTION_AGENT.md) |

**Relationship to the main pipeline:** `adoption_forecaster/main.py` runs Eurostat loading and quota forecasting only. **AI_adjustments** is a **separate** CLI: run from the repository root with `python -m adoption_forecaster.AI_adjustments.main ...` (see that README). It loads `config.yaml` for GCP Secret Manager (`secrets.openai_api_key`) like the rest of the project.

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
- Copy the correct input files from `adoption-forecaster-data/Eurostat/` bucket to the same level as the `main.py` file.
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

### Set up git hooks (downloads Ruff, etc. automatically)

```sh
poetry run pre-commit install
```

They run automatically on each commit. In case hooks have changed your file you need to stage & commit your file again.

The hooks are defined in `.pre-commit-config.yaml` (trailing whitespace, end-of-file newline, YAML/TOML validity, merge-conflict markers, large files, debug statements and Ruff; it reads its settings from `pyproject.toml`, line length 149, with the correctness rules `E4,E7,E9,F`). They only check the files in a commit. There is no code formatter configured, only Ruff's correctness lint. Run `poetry run pre-commit run --all-files` to check the whole repo; the whitespace and end-of-file hooks may touch existing files, so commit that on its own.

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

- Builds the Docker image (adoption-forecaster)
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

## Testing

Run everything from the repository root inside the Poetry environment. No cloud credentials are needed for the default run (about 100 s):

```sh
pytest                                  # whole repo, including the AI_adjustments tests
pytest tests                            # same, without the AI_adjustments tests
pytest -m "not emulators and not realdata"   # explicit form of the default offline run
```

Test layers, from cheapest to most realistic:

| Layer | Command | Needs | What it proves |
|---|---|---|---|
| Unit and golden tests | `pytest tests` | nothing | module logic, and that the real pipeline on a seeded synthetic world reproduces the golden CSVs in `tests/golden/` |
| Pipeline end to end (local) | `pytest tests/e2e/test_pipeline_local.py` | nothing | the full forecasting pipeline, input files to output files |
| Correction agent in the pipeline | `pytest tests/e2e/test_pipeline_with_correction.py` | nothing | the agent behind `correction.enabled`, with a scripted fake LLM and the synthetic source store |
| Production path | `pytest -m emulators` | local emulators (next section) | reads and writes through fake GCS and publishes to Pub/Sub |
| Real Eurostat backtest | `QF_REALDATA=1 pytest -m realdata` | emulators for the first fetch, then the cache in `.cache/eurostat/` | the real pipeline cut at an earlier year; asserts shapes and ranges, not exact numbers |
| Agent benchmark (stub) | `python -m adoption_forecaster.correction.benchmark_cli --module education --seeds 0-4` | nothing | scores corrections against the synthetic truth with an oracle standing in for the model |
| Agent benchmark (live LLM) | same CLI plus `--live --model <id>` | `NVIDIA_API_KEY` exported in your shell; about 17 minutes for 3 seeds x 5 countries | whether a real model actually corrects; see [docs/CORRECTION_AGENT.md](docs/CORRECTION_AGENT.md) |

Notes:
- The emulator and real-data tests are skipped when their environment variables are not set, so a plain `pytest` stays offline.
- Running `main.py` on real production inputs needs the input CSVs from your own bucket (see "Local Execution"); a full run on real data goes through the real-data backtest instead.
- The real CLI cannot enable `correction.enabled` yet: it stops with `NoSourceStoreError` until a real source store exists (see "Before enabling `correction.enabled`" in [docs/CORRECTION_AGENT.md](docs/CORRECTION_AGENT.md)).
- Not testable locally: Secret Manager, IAM, the Cloud Run job and the image push. These need a real GCP project (see "Cloud resources").

### Testing with local emulators

You can run the production path (GCS reads and writes, Pub/Sub messages) without a GCP project by using local emulators. Docker must be running.

```sh
docker compose -f docker-compose.emulators.yml up -d
export STORAGE_EMULATOR_HOST=http://localhost:4553   # fake GCS (fsouza/fake-gcs-server)
export PUBSUB_EMULATOR_HOST=localhost:8685           # Pub/Sub emulator
pytest -m emulators -v
docker compose -f docker-compose.emulators.yml down -v
```

- When `STORAGE_EMULATOR_HOST` is set, `services/gcp_clients.py` uses anonymous credentials instead of reading Secret Manager, so no key or `gcloud` login is needed.
- Without these variables the emulator tests are skipped, so a plain `pytest tests` stays offline.
- Real Eurostat data for the backtest is fetched through the GCS emulator once and cached in `.cache/eurostat/`. Delete that folder and run `QF_REALDATA=1 pytest -m realdata` (emulators up) to refresh it; with the cache present those tests run offline.
- Not covered: Secret Manager, IAM and Cloud Run are not emulated, and the fake GCS server does not enforce permissions. The deploy path and bucket roles can only be tested in a real project (see Cloud resources below).

## Cloud resources

The code, Dockerfile and workflows expect the resources below. They are not shipped with this repo: create them in your own GCP project, then replace `YOUR_GCP_PROJECT_ID` (Dockerfile, `config.yaml`, workflows) with your project ID. If you pick different names, change them in the places listed.

| Resource | Default name | Used for | Set in |
|---|---|---|---|
| GCS bucket (internal) | `adoption-forecaster-data` | Inputs and outputs (`<data_source>/<module>/input,output`) | `config.yaml` `defaults.bucket_name`, Dockerfile `GCS_BUCKET_NAME`, `pr-validation.yml`, `tests/e2e/conftest.py` |
| Secret (service-account key) | `adoption-forecaster-sa-key` | Runtime credentials for GCS and Pub/Sub | `services/gcp_clients.py`, `docker-run.sh`, workflows |
| Secret (LLM key) | `adoption-forecaster-openai-api-key` | Key for the AI corrections | `config.yaml` `secrets.openai_api_key` |
| Service account | `adoption-forecaster@YOUR_GCP_PROJECT_ID.iam.gserviceaccount.com` | Identity the pipeline runs as | `config.yaml` `pubsub.service_account` |
| Container image | `eu.gcr.io/YOUR_GCP_PROJECT_ID/adoption-forecaster-image` | Pushed by the post-merge workflow | `post-merge-run-full-pipeline.yml` |
| Cloud Run job | `adoption-forecaster-run` (region `europe-west1`) | Runs the pipeline in the container | `post-merge-trigger-cloud-run.yml` |

Create them once (bucket names are global, so `adoption-forecaster-data` may be taken: choose your own and update the files above):

```sh
gcloud config set project YOUR_GCP_PROJECT_ID
gcloud storage buckets create gs://adoption-forecaster-data --location=europe-west1
gcloud iam service-accounts create adoption-forecaster
gcloud storage buckets add-iam-policy-binding gs://adoption-forecaster-data \
  --member=serviceAccount:adoption-forecaster@YOUR_GCP_PROJECT_ID.iam.gserviceaccount.com \
  --role=roles/storage.objectAdmin
gcloud iam service-accounts keys create /tmp/key.json \
  --iam-account=adoption-forecaster@YOUR_GCP_PROJECT_ID.iam.gserviceaccount.com
gcloud secrets create adoption-forecaster-sa-key --data-file=/tmp/key.json && rm /tmp/key.json
printf '%s' "$LLM_API_KEY" | gcloud secrets create adoption-forecaster-openai-api-key --data-file=-
```

The Cloud Run job and the image push are created by the workflows once the image exists; the job needs the service account above.

Other values in this repo still point at systems that were not created here and must be replaced or switched off:
- `config.yaml` `defaults.project_id` and `pubsub.sub_project_id` are numeric project numbers of the previous owner. Set `project_id` to your own project number (it is used to read the secret).
- `bucket_name_external`, `dir_external` and `pubsub.topic_name` describe another team's bucket and topic. They are only used when `pubsub.enable_notifications` is `true`; leave it `false` unless you provide your own.
- The workflows run in the runner image `eu.gcr.io/gwi-host-net/ubuntu:act-latest`, which belongs to the previous owner. Replace it with an image you can pull (for example `ubuntu:22.04` with the tools installed).

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
- 🚀 Pushes the built Docker image to`eu.gcr.io/YOUR_GCP_PROJECT_ID/adoption-forecaster-image` if all  above tests pass

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
- **Key config toggles** (in `adoption_forecaster/config/config.yaml`):
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

- Notes: Ensure Secret Manager contains the publisher key at `projects/<project_id>/secrets/adoption-forecaster-sa-key`, and `bucket_name_external`/`dir_external` are set for Andromeda.
