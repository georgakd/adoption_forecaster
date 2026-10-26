# Adoption Forecaster Architecture

This document describes the architecture of the `adoption_forecaster` service, including how components interact and how data flows to the Universe service (Andromeda) via Pub/Sub.

---

## High-Level Architecture Overview

```
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│                               QUOTAS FORECASTING SERVICE                                  │
│                                                                                           │
│  ┌──────────────┐    ┌────────────────┐    ┌──────────────────┐    ┌───────────────────┐  │
│  │   main.py    │───▶│   Processors   │───▶│    Predictors    │───▶│    FileWriter     │  │
│  │   (Entry)    │    │ (Pipeline Orch)│    │ (ML Forecasting) │    │ (Storage + Notify)│  │
│  └──────────────┘    └────────────────┘    └──────────────────┘    └─────────┬─────────┘  │
│         │                                                                    │            │
│         │ config                                                             │            │
│         ▼                                                                    ▼            │
│  ┌──────────────┐                                                 ┌───────────────────┐   │
│  │ ConfigLoader │                                                 │ MessageSerializer │   │
│  │(config.yaml) │                                                 │(Protobuf + PubSub)│   │
│  └──────────────┘                                                 └─────────┬─────────┘   │
│                                                                             │             │
└─────────────────────────────────────────────────────────────────────────────┼─────────────┘
```

---

## Component Breakdown

### 1. Entry Point (`main.py`)

The main orchestrator that:

- Parses CLI arguments (`--module`, `--mode`, `--pred-year`, `--last-year`)
- Loads configuration from `config.yaml`
- Optionally fetches fresh data from Eurostat
- Instantiates and runs the appropriate modules (Education, Age_Gender, or both)

### 2. Processors (`forecasting/processors.py`)

Two processor classes handle module-specific pipelines:

| Processor              | Responsibility                                                               |
| ---------------------- | ---------------------------------------------------------------------------- |
| `EducationProcessor` | Processes education-level internet usage data for 16-64 and 16-74 age groups |
| `AgeGenderProcessor` | Processes age/gender demographic internet penetration data                   |

**Factory Pattern**: `get_processor(module, config)` returns the appropriate processor.

### 3. Predictors (`forecasting/predictors.py`)

Handles the actual forecasting using multiple strategies:

- **Theta Forecasting** - Time series prediction
- **Random Walk** - Drift-based forecasting
- **Logistic Regression** - For saturation modeling
- **LASSO Regression** - For proxy country interpolation

**Factory Pattern**: `get_predictor(module_type, ...)` returns the appropriate predictor.

### 4. Data Handlers

| Component              | File                                       | Responsibility                                                      |
| ---------------------- | ------------------------------------------ | ------------------------------------------------------------------- |
| `Loader`             | `data_handlers/loader.py`                | Loads CSV data from local filesystem or GCS                         |
| `FileWriter`         | `data_handlers/file_writer.py`           | Saves predictions locally or to GCS, triggers Pub/Sub notifications |
| `ProxyHandler`       | `data_handlers/proxy_handlers.py`        | Handles proxy country logic and LASSO interpolation                 |
| `MissingDataHandler` | `data_handlers/missing_data_handlers.py` | Handles missing data in datasets                                    |
| `DataValidator`      | `data_handlers/data_validators.py`       | Validates data integrity                                            |

### 5. Services

| Component               | File                                | Responsibility                                             |
| ----------------------- | ----------------------------------- | ---------------------------------------------------------- |
| `MessageSerializer`   | `services/message_serializer.py`  | Creates protobuf messages and publishes to Pub/Sub         |
| `modelling_files_pb2` | `services/modelling_files_pb2.py` | Generated protobuf definitions for `ModellingOutputFile` |

### 6. Optional: AI projection corrections (`AI_adjustments/`)

Not part of the default forecasting flow above. A **separate CLI** (`python -m adoption_forecaster.AI_adjustments.main`) can post-process exported prediction CSVs using an LLM with tool calling (outlier review, domain constraints, optional researcher-feedback RAG). See **[AI_PROJECTION_CORRECTIONS.md](./AI_PROJECTION_CORRECTIONS.md)** and **[ARCHITECTURE_RAG.md](./ARCHITECTURE_RAG.md)**.

---

## Data Flow: From Input to Universe Service

```
                        ┌─────────────────┐
                        │    EUROSTAT     │
                        │  (External API) │
                        └────────┬────────┘
                                 │ fetch (if auto_fetch=true)
                                 ▼
                        ┌─────────────────┐     ┌─────────────────┐
                        │  Eurostat Data  │────▶│     Loader      │
                        │   (CSV files)   │     │                 │
                        └─────────────────┘     └────────┬────────┘
                                                         │
                 ┌───────────────────────────────────────┼────────────────────────────────────────┐
                 │                                       │                                        │
                 ▼                                       ▼                                        ▼
        ┌─────────────────┐                     ┌─────────────────┐                      ┌─────────────────┐
        │   MissingData   │                     │  ProxyHandler   │                      │  DataValidator  │
        │     Handler     │                     │                 │                      │                 │
        └────────┬────────┘                     └────────┬────────┘                      └────────┬────────┘
                 │                                       │                                        │
                 └───────────────────────────────────────┼────────────────────────────────────────┘
                                                         │
                                                         ▼
                                                ┌─────────────────┐
                                                │   Processors    │
                                                │  (Education /   │
                                                │   AgeGender)    │
                                                └────────┬────────┘
                                                         │
                                                         ▼
                                                ┌─────────────────┐
                                                │   Predictors    │
                                                │  (Theta, RW,    │
                                                │   Logistic,     │
                                                │   LASSO)        │
                                                └────────┬────────┘
                                                         │
                                                         ▼
                                                ┌─────────────────┐
                                                │   FileWriter    │
                                                │                 │
                                                └────────┬────────┘
                                                         │
                 ┌───────────────────────────────────────┼────────────────────────────────────────┐
                 │                                       │                                        │
                 ▼                                       ▼                                        ▼
        ┌─────────────────┐                     ┌─────────────────┐                      ┌──────────────────┐
        │  Internal GCS   │                     │  External GCS   │                      │ MessageSerializer│
        │   Bucket (ML)   │                     │   (Andromeda)   │                      │                  │
        │                 │                     │                 │                      │   Protobuf +     │
        │  ml-quotas-     │                     │ universe-bucket │                      │     Pub/Sub      │
        │  forecasting    │                     │ -staging        │                      │                  │
        └─────────────────┘                     └─────────────────┘                      └────────┬─────────┘
                                                                                                  │
                                                                                                  ▼
                                                                                         ┌─────────────────┐
                                                                                         │    Pub/Sub      │
                                                                                         │     Topic       │
                                                                                         │                 │
                                                                                         │  population-    │
                                                                                         │ modelling-files │
                                                                                         │ -notifications  │
                                                                                         └────────┬────────┘
                                                                                                  │
                                                                                                  ▼
                                                                                         ┌─────────────────┐
                                                                                         │ UNIVERSE SERVICE│
                                                                                         │   (Andromeda)   │
                                                                                         └─────────────────┘
```

---

## Pub/Sub Notification Flow

When predictions are saved in production mode with `enable_notifications: true`, the following sequence occurs:

### Sequence Diagram

```
┌──────────┐    ┌───────────┐    ┌──────────────────┐    ┌─────────────┐    ┌──────────┐    ┌──────────┐
│Processor │    │FileWriter │    │MessageSerializer │    │ Secret Mgr  │    │ Pub/Sub  │    │ Universe │
└────┬─────┘    └─────┬─────┘    └────────┬─────────┘    └──────┬──────┘    └────┬─────┘    └────┬─────┘
     │                │                    │                    │                │               │
     │save_predictions│                    │                    │                │               │
     │───────────────▶│                    │                    │                │               │
     │                │                    │                    │                │               │
     │                │ Upload to          │                    │                │               │
     │                │ Internal GCS       │                    │                │               │
     │                │─────────┐          │                    │                │               │
     │                │         │          │                    │                │               │
     │                │◀────────┘          │                    │                │               │
     │                │                    │                    │                │               │
     │                │ Upload to          │                    │                │               │
     │                │ External GCS       │                    │                │               │
     │                │ (Andromeda bucket) │                    │                │               │
     │                │─────────┐          │                    │                │               │
     │                │         │          │                    │                │               │
     │                │◀────────┘          │                    │                │               │
     │                │                    │                    │                │               │
     │                │ send_modelling_    │                    │                │               │
     │                │ output_notification│                    │                │               │
     │                │───────────────────▶│                    │                │               │
     │                │                    │                    │                │               │
     │                │                    │ Get SA credentials │                │               │
     │                │                    │───────────────────▶│                │               │
     │                │                    │                    │                │               │
     │                │                    │◀───────────────────│                │               │
     │                │                    │                    │                │               │
     │                │                    │ Create protobuf    │                │               │
     │                │                    │ message            │                │               │
     │                │                    │─────────┐          │                │               │
     │                │                    │         │          │                │               │
     │                │                    │◀────────┘          │                │               │
     │                │                    │                    │                │               │
     │                │                    │ Publish to topic   │                │               │
     │                │                    │────────────────────────────────────▶│               │
     │                │                    │                    │                │               │
     │                │                    │◀─────────────────────────────────── │               │
     │                │                    │     message_id     │                │               │
     │                │                    │                    │                │               │
     │                │◀───────────────────│                    │                │ Consume event │
     │                │     success        │                    │                │──────────────▶│
     │◀───────────────│                    │                    │                │               │
     │                │                    │                    │                │               │
```

### Protobuf Message Structure

The `ModellingOutputFile` protobuf message contains:

| Field               | Type          | Description                           |
| ------------------- | ------------- | ------------------------------------- |
| `time`            | Timestamp     | UTC timestamp when event occurred     |
| `sender`          | string        | Service name (`adoption_forecaster`) |
| `file_type`       | FileType enum | Type of output file (see below)       |
| `gcs_path`        | string        | Full GCS path to the uploaded file    |
| `projection_year` | int           | Year of the population projection     |

**FileType Enum Values:**

- `FILE_TYPE_INVALID` (0) - Default/error value
- `AGE_AND_GENDER` (1) - Age and gender predictions
- `EDUCATION_16_TO_64` (2) - Education predictions for 16-64 age group
- `EDUCATION_16_TO_74` (3) - Education predictions for 16-74 age group

---

## Configuration-Driven Behavior

The service behavior is controlled by `config.yaml`:

```yaml
# Enable/disable Pub/Sub notifications
pubsub:
  enable_notifications: true  # Set to true to send notifications
  topic_name: "population-modelling-files-notifications"
  service_account: "quotas-forecasting@ml-cloud-374512.iam.gserviceaccount.com"
  sub_project_id: "464955218160"  # Andromeda's project ID

# External bucket configuration (Andromeda)
defaults:
  bucket_name_external: "universe-bucket-staging"
  dir_external: "population_modelling_projections"

# Data source must be "Eurostat" for pub/sub
gcs:
  data_source: "Eurostat"
```

### Notification Trigger Conditions

Notifications are sent only when ALL of these conditions are met:

1. `mode == "prod"` (production mode)
2. `pubsub.enable_notifications == true`
3. `gcs.data_source == "Eurostat"` (case-insensitive)
4. File is a `past_data` file (contains historical + predictions)

---

## GCS Bucket Structure

### Internal Bucket (`ml-quotas-forecasting`)

```
ml-quotas-forecasting/
├── Eurostat/
│   ├── Education/
│   │   ├── input/
│   │   │   └── internet_penetration_ratios_by_education_*.csv
│   │   └── output/
│   │       ├── 2026_Education_16-64_predictions_both_2025-01-21.csv
│   │       └── 2026_Education_16-64_predictions_with_past_data_both_2025-01-21.csv
│   │
│   └── Age_Gender/
│       ├── input/
│       │   └── internet_penetration_by_age_group_and_gender_*.csv
│       └── output/
│           ├── 2026_Age_Gender_16-74_predictions_both_2025-01-21.csv
│           └── 2026_Age_Gender_16-74_predictions_with_past_data_both_2025-01-21.csv
```

### External Bucket (`universe-bucket-staging`)

```
universe-bucket-staging/
└── population_modelling_projections/
    ├── 2026_Age_Gender_16-74_predictions_with_past_data_both.csv
    ├── 2026_Education_16-64_predictions_with_past_data_both.csv
    └── 2026_Education_16-74_predictions_with_past_data_both.csv
```

---

## Authentication & Security

### Service Account Flow

```
┌──────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│ Adoption Forecaster  │     │    Secret Manager    │     │   Service Account    │
│       Service        │────▶│                      │────▶│    Credentials       │
│                      │     │ quotas-forecasting-  │     │                      │
│                      │     │ sa-key               │     │ Used for:            │
│                      │     │                      │     │ - GCS access         │
│                      │     │                      │     │ - Pub/Sub publishing │
└──────────────────────┘     └──────────────────────┘     └──────────────────────┘
```

The service:

1. Retrieves SA key from Secret Manager at runtime
2. Uses credentials for both GCS operations and Pub/Sub publishing
3. Publishes to Andromeda's Pub/Sub topic in a different GCP project

---

## Module Pipeline Flows

### Education Module Pipeline

```
Input CSVs                                Processing                              Output
───────────                               ──────────                              ──────

population_ratios_by_education       ┌───────────────────┐
internet_penetration_by_education ──▶│ Load & Standardize│
total_population_by_age_range        │     Countries     │
                                     └─────────┬─────────┘
                                               │
                                               ▼
                                     ┌───────────────────┐
                                     │  Handle Missing   │
                                     │       Data        │
                                     └─────────┬─────────┘
                                               │
                                               ▼
                                     ┌───────────────────┐
                                     │  LASSO for Proxy  │
                                     │    Countries      │
                                     └─────────┬─────────┘
                                               │
                                               ▼
                                     ┌───────────────────┐
                                     │   Interpolation   │
                                     └─────────┬─────────┘
                                               │
                                               ▼
                                     ┌───────────────────┐
                                     │ Theta + RW Blend  │───▶ Education_16-64_predictions.csv
                                     │    Forecasting    │───▶ Education_16-74_predictions.csv
                                     └───────────────────┘
```

### Age/Gender Module Pipeline

```
Input CSV                                 Processing                              Output
─────────                                 ──────────                              ──────

internet_penetration_by_age_group    ┌───────────────────┐
    _and_gender                 ────▶│ Load & Standardize│
                                     │     Countries     │
                                     └─────────┬─────────┘
                                               │
                                               ▼
                                     ┌───────────────────┐
                                     │Extract Age/Gender │
                                     │    from rows      │
                                     └─────────┬─────────┘
                                               │
                                               ▼
                                     ┌───────────────────┐
                                     │    Logistic +     │
                                     │   Theta + RW      │───▶ Age_Gender_16-74_predictions.csv
                                     │  Blend Forecast   │
                                     └───────────────────┘
```

---

## Key Design Patterns Used

| Pattern                        | Where                                    | Purpose                                          |
| ------------------------------ | ---------------------------------------- | ------------------------------------------------ |
| **Factory**              | `get_processor()`, `get_predictor()` | Decoupled object creation                        |
| **Strategy**             | Local vs GCS storage modes               | Runtime algorithm selection                      |
| **Composition**          | Processors with handlers                 | Flexible, reusable components                    |
| **Adapter**              | FileWriter                               | Unified interface for different storage backends |
| **Pipeline**             | Processor methods                        | Clear, sequential data processing                |
| **Layered Architecture** | Eurostat fetching                        | Separation of concerns (API → Service → App)   |

---

## Cloud Run Deployment & CI/CD

The service is containerized and deployed to Google Cloud Run as a **Cloud Run Job**. The deployment is fully automated through GitHub Actions workflows.

### Deployment Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                                    GITHUB ACTIONS                                    │
│                                                                                      │
│  ┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐     │
│  │  PR Check Workflow  │     │ Post-Merge Pipeline │     │  Cloud Run Trigger  │     │
│  │                     │     │                     │     │                     │     │
│  │ • Docker build test │     │ • Full pipeline run │     │ • Execute Cloud Run │     │
│  │ • CLI validation    │     │ • Push image to GCR │     │   Job               │     │
│  └─────────────────────┘     └──────────┬──────────┘     └──────────┬──────────┘     │
│                                         │                           │                │
└─────────────────────────────────────────┼───────────────────────────┼────────────────┘
                                          │                           │
                                          ▼                           ▼
                            ┌─────────────────────────┐   ┌─────────────────────────┐
                            │    Google Container     │   │     Cloud Run Job       │
                            │    Registry (GCR)       │   │                         │
                            │                         │   │ quotas-forecasting-run  │
                            │ eu.gcr.io/ml-cloud-     │   │                         │
                            │ 374512/quotas-          │──▶│ Runs forecasting        │
                            │ forecasting-image       │   │ pipeline in container   │
                            └─────────────────────────┘   └─────────────────────────┘
```

### GitHub Actions Workflows

| Workflow                             | Trigger          | Purpose                                 |
| ------------------------------------ | ---------------- | --------------------------------------- |
| `pr-check-docker-build.yml`        | Pull Request     | Validates Docker build and CLI startup  |
| `post-merge-run-full-pipeline.yml` | Push to `main` | Runs full pipeline, pushes image to GCR |
| `post-merge-trigger-cloud-run.yml` | Push to `main` | Triggers Cloud Run Job execution        |

### CI/CD Pipeline Flow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   PR Open   │───▶│Docker Build │───▶│  PR Merged  │───▶│  Full Test  │───▶│ Push to GCR │
│             │    │   + CLI     │    │   to main   │    │  Pipeline   │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └──────┬──────┘
                                                                                   │
                                                                                   ▼
                                                                          ┌───────────────┐
                                                                          │ Trigger Cloud │
                                                                          │    Run Job    │
                                                                          └───────┬───────┘
                                                                                  │
                                                                                  ▼
                                                                          ┌───────────────┐
                                                                          │    Execute    │
                                                                          │  Forecasting  │
                                                                          │   Pipeline    │
                                                                          └───────────────┘
```

### Docker Image

The service is packaged in a Docker container based on `python:3.11-slim`:

```
┌───────────────────────────────────────────┐
│              Docker Image                 │
│                                           │
│  Base: python:3.11-slim                   │
│                                           │
│  ├── Poetry (dependency management)       │
│  ├── adoption_forecaster/ (source code)    │
│  ├── pyproject.toml + poetry.lock         │
│  │                                        │
│  └── ENTRYPOINT:                          │
│      python adoption_forecaster/main.py    │
│                                           │
│  User: appuser (non-root, UID 8002)       │
└───────────────────────────────────────────┘
```

**Key Environment Variables:**

| Variable                 | Value                     | Purpose                   |
| ------------------------ | ------------------------- | ------------------------- |
| `GOOGLE_CLOUD_PROJECT` | `ml-cloud-374512`       | GCP project ID            |
| `GCS_BUCKET_NAME`      | `ml-quotas-forecasting` | Internal storage bucket   |
| `PYTHONPATH`           | `/app`                  | Python package resolution |

### Cloud Run Job Configuration

| Setting  | Value                                                         |
| -------- | ------------------------------------------------------------- |
| Job Name | `quotas-forecasting-run`                                    |
| Region   | `europe-west1`                                              |
| Project  | `ml-cloud-374512`                                           |
| Image    | `eu.gcr.io/ml-cloud-374512/quotas-forecasting-image:latest` |

### Triggering the Job

The Cloud Run Job is triggered automatically after code is merged to `main`:

```bash
gcloud run jobs execute quotas-forecasting-run \
  --region=europe-west1 \
  --project=ml-cloud-374512 \
  --wait
```

The `--wait` flag ensures the workflow waits for job completion before marking success/failure.

### Secrets & Authentication

| Secret                        | Source         | Purpose                                      |
| ----------------------------- | -------------- | -------------------------------------------- |
| `SA_ML_CLOUD`               | GitHub Secret  | Authenticate to GCP for Cloud Run execution  |
| `SA_GWI_HOST_NET`           | GitHub Secret  | Authenticate for pipeline testing & GCR push |
| `quotas-forecasting-sa-key` | Secret Manager | Runtime credentials for GCS and Pub/Sub      |

---

## Summary

The `adoption_forecaster` service:

1. **Deploys** automatically via GitHub Actions → GCR → Cloud Run Job
2. **Fetches** demographic data from Eurostat (optional)
3. **Processes** data through Education and/or Age_Gender pipelines
4. **Generates** predictions using ensemble forecasting methods
5. **Stores** results to both internal (ML) and external (Andromeda) GCS buckets
6. **Notifies** Universe service via Pub/Sub with protobuf messages

Optional **AI projection corrections** (`AI_adjustments/`) are documented in **[AI_PROJECTION_CORRECTIONS.md](./AI_PROJECTION_CORRECTIONS.md)** and are **not** invoked by the main service entry point by default.

The CI/CD pipeline ensures that every merge to `main` triggers a full test run, pushes an updated Docker image to GCR, and executes the Cloud Run Job. The Pub/Sub integration enables Andromeda to automatically consume new projection files without polling, creating an event-driven data pipeline between the ML team and the Universe service.
