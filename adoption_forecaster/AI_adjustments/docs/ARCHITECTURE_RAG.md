# RAG Architecture Schema

**See also:** [AI_PROJECTION_CORRECTIONS.md](./AI_PROJECTION_CORRECTIONS.md) — technical reference for the `AI_adjustments` LLM post-processor (problem, solution, limits).

## High-Level Flow

```
main.py
  │
  ├─► Load feedback CSV → FeedbackRAGRetriever
  │                        │
  │                        ├─► Build embeddings index (OpenAI text-embedding-3-small)
  │                        └─► Store: {country, level/individual, comment, embedding}
  │
  └─► Initialize CorrectionAgent (with feedback_retriever)
      │
      └─► Tool Execution Loop
          │
          ├─► analyze_country_trend_*()
          │   └─► feedback_retriever.get_feedback_*()
          │       │
          │       ├─► Direct Match Check
          │       │   └─► Exact country/individual match
          │       │
          │       └─► Semantic Search (if no match)
          │           ├─► Generate query embedding
          │           ├─► Cosine similarity vs. index
          │           └─► Return if similarity > 0.75
          │
          └─► Feedback included in analysis → AI adjustment decisions
```

## Component Interactions

```
┌─────────────────┐
│    main.py      │
│                 │
│ • Load CSV      │──┐
│ • Init RAG      │  │ feedback_data
└────────┬────────┘  │
         │           │
         │           ▼
         │   ┌──────────────────┐
         │   │FeedbackRAGRetriever│
         │   │                   │
         │   │ _build_index()    │──► OpenAI Embeddings API
         │   │   • Pre-compute    │    (text-embedding-3-small)
         │   │     embeddings    │
         │   └─────────┬─────────┘
         │             │
         │             │ instance
         │             │
         ▼             │
┌─────────────────┐   │
│ CorrectionAgent │◄──┘
│                 │
│ • Stores        │
│   feedback_     │
│   retriever     │
└────────┬────────┘
         │
         │ passes to tools
         │
         ▼
┌─────────────────┐
│    tools.py     │
│                 │
│ analyze_*()     │──► feedback_retriever.get_feedback_*()
│                 │    └─► Returns feedback text
│                 │        └─► Included in analysis result
└─────────────────┘
```

## RAG Retrieval Process

### Education Module

```
get_feedback_education(country, level, projection_value, previous_value)
  │
  ├─► 1. Direct Match: Check feedback_data[country]["secondary"]
  │
  ├─► 2. Build Query: "{country} secondary education projection {value}%"
  │
  ├─► 3. Generate Embedding: OpenAI API (text-embedding-3-small)
  │
  ├─► 4. Semantic Search: Cosine similarity vs. embeddings_index
  │
  └─► 5. Return: comment if similarity > 0.75, else None
```

### Age/Gender Module

```
get_feedback_age_gender(country, individual, column, projection_value, previous_value)
  │
  ├─► 1. Direct Match: Check feedback_data[country]["age_gender"][individual]
  │
  ├─► 2. Build Query: "{country} {individual} age gender {column} projection {value}"
  │
  ├─► 3. Generate Embedding: OpenAI API
  │
  ├─► 4. Semantic Search: Filter by module="age_gender", cosine similarity
  │
  └─► 5. Return: comment if similarity > 0.75, else None
```

## Data Structures

### FeedbackRAGRetriever

- `feedback_data`: dict - Loaded from CSV via `load_feedback()`
- `embeddings_index`: List[dict] - Pre-computed embeddings

  ```python
  {
    "country": str,
    "level" | "individual": str,
    "comment": str,
    "embedding": List[float],
    "module": str (optional, for age_gender)
  }
  ```

### Tool Analysis Result

```python
{
  "analyses": {
    "column_name": {
      "previous_year_value": float,
      "current_value": float,
      "change_pct": float,
      "is_outlier": bool,
      "feedback": str | None  # ← RAG-retrieved feedback
    }
  }
}
```

## Key Features

- **Dual Retrieval**: Direct match → Semantic search fallback
- **Similarity Threshold**: 0.75 minimum cosine similarity
- **Module Support**: Education (secondary only) and Age/Gender
- **Pre-computed Index**: Embeddings built once during initialization
- **Graceful Degradation**: Works without feedback files
