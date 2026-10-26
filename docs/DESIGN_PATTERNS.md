# SOLID Principles & Design Patterns

This document explains the SOLID principles and design patterns implemented in the `adoption_forecaster` codebase.

---

## Table of Contents

1. [SOLID Principles](#solid-principles)
   - [Single Responsibility Principle (SRP)](#single-responsibility-principle-srp)
   - [Open/Closed Principle (OCP)](#openclosed-principle-ocp)
   - [Liskov Substitution Principle (LSP)](#liskov-substitution-principle-lsp)
   - [Interface Segregation Principle (ISP)](#interface-segregation-principle-isp)
   - [Dependency Inversion Principle (DIP)](#dependency-inversion-principle-dip)
2. [Design Patterns](#design-patterns)
   - [Factory Pattern](#factory-pattern)
   - [Strategy Pattern](#strategy-pattern)
   - [Composition over Inheritance](#composition-over-inheritance)
   - [Adapter Pattern](#adapter-pattern)
   - [Layered Architecture](#layered-architecture)
   - [Pipeline Pattern](#pipeline-pattern)

---

## SOLID Principles

### Single Responsibility Principle (SRP)

> *A class should have only one reason to change.*

The codebase separates concerns into focused, single-purpose classes:

| Class | Single Responsibility |
|-------|----------------------|
| `Loader` | Data loading from local/GCS |
| `FileWriter` | Saving predictions to local/GCS |
| `ProxyHandler` | Finding proxy countries and interpolation |
| `MissingDataHandler` | Handling missing data in education datasets |
| `DataValidator` | Validating data integrity |
| `MessageSerializer` | Protobuf message creation and Pub/Sub publishing |

**Example: Handler separation in processors.py**

```python
class EducationProcessor:
    def __init__(self, module_type="education", config=None):
        # Each handler has a single, well-defined responsibility
        self.proxy_handler = ProxyHandler(config=self.config)          # Proxy country logic
        self.missing_data_handler = MissingDataHandler(config=self.config)  # Missing data logic
        self.validator = DataValidator(config=self.config)             # Validation logic
```

Instead of one massive processor class doing everything, the logic is delegated to specialized handlers.

---

### Open/Closed Principle (OCP)

> *Software entities should be open for extension but closed for modification.*

The factory functions allow adding new modules without modifying existing code:

**Example: Factory functions in processors.py and predictors.py**

```python
def get_processor(module, config=None):
    """Factory function - extend by adding new elif, without modifying existing processors"""
    if module == "education":
        return EducationProcessor(module_type=module, config=config)
    elif module == "age_gender":
        return AgeGenderProcessor(module_type=module, config=config)
    # To add a new module, just add: elif module == "new_module": return NewModuleProcessor(...)
    else:
        raise ValueError(f"Unknown module: {module}")


def get_predictor(module_type, mode="local", pred_year=None, last_year=None, loader=None, config=None):
    """Factory function for predictors"""
    if module_type == "education":
        return EducationPredictor(mode=mode, pred_year=pred_year, ...)
    elif module_type == "age_gender":
        return AgeGenderPredictor(mode=mode, pred_year=pred_year, ...)
    else:
        raise ValueError(f"Unknown module type: {module_type}")
```

**How main.py benefits from OCP:**

```python
# main.py doesn't need to know the internal implementation of processors/predictors
processor = get_processor(module_name, config=config)
predictor = get_predictor(module_name, mode=mode, pred_year=pred_year, ...)

# Adding a new forecasting module requires:
# 1. Creating a new Processor class
# 2. Creating a new Predictor class  
# 3. Adding them to the factory functions
# main.py remains unchanged!
```

---

### Liskov Substitution Principle (LSP)

> *Objects of a superclass should be replaceable with objects of its subclasses without affecting program correctness.*

While Python doesn't enforce interfaces, the predictors and processors follow consistent method signatures:

**Example: Predictor classes share common interface**

```python
class EducationPredictor:
    def __init__(self, mode="local", pred_year=2025, last_year=2023, loader=None, config=None):
        ...
    
    def predict(self, df, last_year_of_observations, year_of_predictions):
        """Main prediction method"""
        ...


class AgeGenderPredictor:
    def __init__(self, mode="local", pred_year=2025, last_year=2023, loader=None, config=None):
        ...
    
    def run_with_data(self, df_age_gender, last_year_of_observations):
        """Main prediction method (different name, but same purpose)"""
        ...
```

Both predictors can be used interchangeably by the main pipeline, with the processor adapting to their specific interfaces.

---

### Interface Segregation Principle (ISP)

> *Clients should not be forced to depend on interfaces they do not use.*

The `Loader` class provides separate methods for different loading needs:

```python
class Loader:
    def read_csv(self, filename):
        """Simple CSV reading for known filenames"""
        ...
    
    def read_csv_timestamped(self, base_filename):
        """Reading CSV files with timestamp patterns (e.g., 'data_2024-01-01T12:00:00.csv')"""
        ...
    
    def _read_local(self, filename):
        """Internal: local file reading"""
        ...
    
    def _read_gcs(self, filename):
        """Internal: GCS file reading"""
        ...
```

Clients only call the methods they need (`read_csv` or `read_csv_timestamped`), not a single bloated method.

---

### Dependency Inversion Principle (DIP)

> *High-level modules should not depend on low-level modules. Both should depend on abstractions.*

**Example: Configuration injection throughout the codebase**

```python
# High-level module (main.py) doesn't depend on concrete config loading
config = load_config()  # Abstraction: config is a dict

# All classes receive config as a parameter (dependency injection)
loader = Loader(namespace=namespace, mode=mode, config=config)
writer = FileWriter(namespace=namespace, mode=mode, pred_year=pred_year, config=config)
processor = get_processor(module_name, config=config)
predictor = get_predictor(module_name, mode=mode, ..., config=config)
```

**Example: Loader injection into processors**

```python
class AgeGenderProcessor:
    def run_age_gender_pipeline(self, loader, writer, predictor, ...):
        # Depends on abstractions (loader, writer), not concrete implementations
        df_age_gender = loader.read_csv_timestamped(df_age_gender_base)
        ...
        writer.save_predictions(frame, ...)
```

---

## Design Patterns

### Factory Pattern

Factory functions create objects without exposing instantiation logic:

```python
# In main.py - client code
processor = get_processor("education", config=config)  # Returns EducationProcessor
predictor = get_predictor("education", ...)            # Returns EducationPredictor

# The main.py doesn't need to know about EducationProcessor or AgeGenderProcessor classes
```

**Benefits:**

- Decouples object creation from usage
- Makes it easy to add new module types
- Centralizes instantiation logic

---

### Strategy Pattern

Different algorithms (strategies) for the same operation, selected at runtime:

**Example: Module-specific pipelines**

```python
# In main.py
if module_name == "education":
    processor.run_education_pipeline_for_all_age_groups(loader, writer, predictor, ...)
elif module_name == "age_gender":
    processor.run_age_gender_pipeline(loader, writer, predictor, ...)
```

**Example: Local vs Production storage strategy in Loader**

```python
class Loader:
    def read_csv(self, filename):
        if self.mode == "local":
            return self._read_local(filename)      # Strategy 1
        else:
            return self._read_gcs(filename)        # Strategy 2
```

**Example: Multiple forecasting strategies combined**

```python
class Adoption_Forecaster:
    def combine_forecasts(self):
        # Multiple prediction strategies combined with weights
        self.merged_all[f"{year}_combined"] = (
            logistic_weight * self.merged_all[f"{year}_logistic"] +
            theta_weight * self.merged_all[f"{year}_theta"] +
            random_walk_weight * self.merged_all[f"{year}_rw"]
        )
```

---

### Composition over Inheritance

The codebase favors composing objects over inheritance hierarchies:

**Example: EducationProcessor composes handlers**

```python
class EducationProcessor:
    def __init__(self, module_type="education", config=None):
        # Composition: using handler objects instead of inheriting from them
        self.proxy_handler = ProxyHandler(config=self.config)
        self.missing_data_handler = MissingDataHandler(config=self.config)
        self.validator = DataValidator(config=self.config)
    
    def interpolation_for_missing_data(self, df, last_year_of_observations):
        # Delegate to composed handler
        return self.proxy_handler.interpolation_for_missing_data(df, last_year_of_observations)
    
    def load_and_process_education_data(self, loader, regex_patterns, country_mapping, age_group):
        # Delegate to composed handler
        return self.missing_data_handler.load_and_process_education_data(...)
```

**Benefits:**

- Flexible: handlers can be swapped or mocked for testing
- Reusable: `ProxyHandler` could be used by other processors
- Testable: each handler can be unit tested independently

---

### Adapter Pattern

The `FileWriter` class adapts between different storage backends:

```python
class FileWriter:
    def save_predictions(self, predictions, year_of_prediction, output_dir, filename, ...):
        """Unified interface for saving predictions"""
        if self.mode == "local":
            return self._save_local_predictions(predictions, ...)   # Adapts to local filesystem
        else:
            return self._save_gcp_predictions(predictions, ...)     # Adapts to GCS

    def _save_local_predictions(self, predictions, ...):
        """Adapt to local filesystem operations"""
        local_path = self._get_local_path(output_dir, filename)
        predictions.to_csv(local_path, encoding="utf-8-sig")

    def _save_gcp_predictions(self, predictions, ...):
        """Adapt to Google Cloud Storage operations"""
        blob_path = self._get_gcs_path(output_dir, filename)
        blob = self.bucket_int.blob(blob_path)
        blob.upload_from_string(csv_content, content_type="text/csv")
```

The client code uses `save_predictions()` without knowing whether it writes locally or to GCS.

---

### Layered Architecture

The Eurostat data fetching follows a three-layer architecture:

```
┌──────────────────────────────────────────────────────────────────┐
│                     APPLICATION LAYER                             │
│                   EurostatDataFetcher                            │
│     High-level orchestrator: fetch_data() → process → upload    │
└─────────────────────────────┬────────────────────────────────────┘
                              │
┌─────────────────────────────▼────────────────────────────────────┐
│                      SERVICE LAYER                                │
│                       EurostatApp                                 │
│     Business logic: process_total_population(), upload_to_gcs() │
└─────────────────────────────┬────────────────────────────────────┘
                              │
┌─────────────────────────────▼────────────────────────────────────┐
│                      UTILITY LAYER                                │
│                     EurostatFetcher                              │
│     Low-level API client: fetch_dataset(), filter_by_education() │
└──────────────────────────────────────────────────────────────────┘
```

**Code example:**

```python
# Application Layer - orchestrates the flow
class EurostatDataFetcher:
    def __init__(self, last_year_of_observations=None, config=None):
        self.app = EurostatApp(last_year_of_observations=last_year_of_observations, config=self.config)
    
    def fetch_data(self):
        self.app.run()                    # Delegate to service layer
        self.app.upload_all_to_gcs()      # Delegate to service layer

# Service Layer - business logic
class EurostatApp:
    def __init__(self, last_year_of_observations, config):
        self.fetcher = EurostatFetcher(...)  # Uses utility layer
    
    def run(self):
        self.process_total_population_by_age_range()
        self.process_internet_by_education()
        ...

# Utility Layer - API interactions
class EurostatFetcher:
    def fetch_dataset(self, code):
        url = self.config["eurostat"]["api"]["data_url_template"].format(code=code)
        resp = requests.get(url)
        return pd.DataFrame(results[0]), updated_datetime
```

---

### Pipeline Pattern

The forecasting modules follow a pipeline pattern with distinct stages:

```
┌─────────┐   ┌────────────┐   ┌────────────┐   ┌──────────┐   ┌────────┐
│  Load   │ → │  Process   │ → │  Predict   │ → │  Validate │ → │  Save  │
│  Data   │   │  & Clean   │   │            │   │           │   │        │
└─────────┘   └────────────┘   └────────────┘   └──────────┘   └────────┘
   Loader      Processor        Predictor        Processor     FileWriter
```

**Example: Education pipeline**

```python
def run_education_pipeline_for_all_age_groups(self, loader, writer, predictor, ...):
    # Stage 1: Load
    education_df = self.load_and_process_education_data(loader, ...)
    
    # Stage 2: Process (fill missing data)
    education_df_merged = self.missing_data_handler.merge_education_levels(education_df_filled)
    education_df_merged = self.interpolation_for_missing_data(education_df_merged, ...)
    
    # Stage 3: Predict
    education_df_merged = predictor.predict(education_df_merged, ...)
    
    # Stage 4 & 5: Validate and Save
    self._save_education_predictions(education_data, writer, ...)
```

---

## Summary

| Principle/Pattern | Where Applied | Benefit |
|-------------------|---------------|---------|
| **SRP** | Separate handler classes | Focused, maintainable code |
| **OCP** | Factory functions | Easy to add new modules |
| **LSP** | Processor/Predictor interfaces | Interchangeable components |
| **ISP** | Loader methods | Clients use only what they need |
| **DIP** | Config/Loader injection | Testable, decoupled code |
| **Factory** | `get_processor()`, `get_predictor()` | Decoupled object creation |
| **Strategy** | Local/GCS modes, forecasting methods | Runtime algorithm selection |
| **Composition** | Processors with handlers | Flexible, reusable components |
| **Adapter** | FileWriter | Unified interface for storage |
| **Layered** | Eurostat fetching | Separation of concerns |
| **Pipeline** | Forecasting flow | Clear, sequential processing |

---

## Related documentation

- **[AI_PROJECTION_CORRECTIONS.md](./AI_PROJECTION_CORRECTIONS.md)** — LLM-based projection post-processing (`AI_adjustments`): agent loop, tools, workflow enforcement, and limitations (separate from the main forecasting pipeline).
