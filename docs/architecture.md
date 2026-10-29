# System Architecture & Design Specification

`LLM Evaluation Lab` is an open-source AI engineering platform designed for reproducible prompt evaluation, model benchmarking, and statistical regression detection.

```
                           ┌────────────────────────┐
                           │   JSONL Test Dataset   │
                           └───────────┬────────────┘
                                       │
                                       ▼
┌──────────────────┐       ┌────────────────────────┐       ┌──────────────────────┐
│ Model Provider   │ ◄───► │    Evaluation Runner   │ ───►  │ Evaluator Pipeline   │
│ (Mock / OpenAI)  │       │     (llm_eval_lab)     │       │ (EM, F1, JSON, etc.) │
└──────────────────┘       └───────────┬────────────┘       └──────────────────────┘
                                       │
                                       ▼
                           ┌────────────────────────┐
                           │  SQLite Storage Layer  │
                           │     (eval_lab.db)      │
                           └───────────┬────────────┘
                                       │
             ┌─────────────────────────┴─────────────────────────┐
             ▼                                                   ▼
┌────────────────────────┐                           ┌────────────────────────┐
│ Statistical Comparison │                           │ API & Frontend / CLI   │
│ (Paired Bootstrap CI)  │                           │ (FastAPI, React, Typer)│
└────────────────────────┘                           └────────────────────────┘
```

---

## 1. Core Principles

1. **Empirical Evidence First:** Subjective prompt engineering is replaced with automated scoring on fixed test datasets.
2. **Statistical Rigor:** Comparisons compute paired bootstrap confidence intervals (default 2,000 resamples) rather than naive point-estimate deltas.
3. **Reproducibility:** Every experiment captures the complete prompt template, provider configuration, dataset digest, latency metrics, and timestamp.
4. **Zero Secret Leakage:** API keys and sensitive tokens are strictly redacted from database rows, CLI outputs, and REST responses.
5. **Decoupled Architecture:** The evaluation runner works as a standalone Python library, a CLI (`llm-eval`), a REST API (`/api/*`), or a React web application.

---

## 2. Component Architecture

### 2.1 Schemas & Data Contracts (`llm_eval_lab.schemas`)
All domain models are validated through Pydantic v2:
- **`TestCase`**: Raw test case containing unique `id`, `input`, optional `reference`, `context`, `metadata`, and evaluator-specific expectations (`expected_json_schema`, `expected_keywords`).
- **`ModelResponse`**: Standardized provider output including generated `text`, `latency_ms`, `token_usage`, `provider`, `model`, and error flags.
- **`MetricResult`**: Output for one metric on one test case, with `score`, `passed`, `details`, and `direction` (`higher_is_better` vs `lower_is_better`).
- **`CaseResult`**: Composite record joining a `TestCase` with its `ModelResponse` and all associated `MetricResult` records.
- **`ExperimentSummary`**: High-level run metrics (success rate, latency percentiles, aggregate means, token consumption).
- **`ComparisonReport`**: Statistical comparison output between baseline and candidate, holding `MetricComparison` arrays and overall `RegressionDecision`.

### 2.2 Providers (`llm_eval_lab.providers`)
- **`BaseProvider`**: Abstract base class defining `generate(prompt: str, **kwargs) -> ModelResponse`.
- **`MockProvider`**: Deterministic offline provider for tests, CI pipelines, and local development. Generates predictable responses, simulates latency, and matches fixtures based on prompt keywords without network requests.
- **`OpenAICompatibleProvider`**: Connects via `httpx` to any OpenAI-compatible API endpoint (OpenAI, Ollama, vLLM, LM Studio, Azure, Groq). Handles retries, timeout configurations, and token usage parsing.

### 2.3 Evaluators (`llm_eval_lab.evaluators`)
Evaluators implement `BaseEvaluator` and compute numeric scores and pass/fail verdicts:
- **Deterministic**:
  - `ExactMatchEvaluator`: Binary string equality.
  - `NormalizedExactMatchEvaluator`: Lowercase, stripped punctuation, normalized whitespace.
  - `TokenOverlapF1Evaluator`: Token-level precision, recall, and harmonic mean F1.
  - `KeywordCoverageEvaluator`: Percentage of required keywords present in model output.
  - `LatencyEvaluator`: Latency in milliseconds (`lower_is_better`).
- **Structural**:
  - `JSONValidityEvaluator`: Validates whether output parses as valid JSON.
  - `JSONSchemaEvaluator`: Validates output against a specified JSON Schema.
- **Semantic & Model-Based**:
  - `SemanticSimilarityEvaluator`: Cosine similarity of sentence embeddings.
  - `JudgeEvaluator`: LLM-as-a-judge scoring with structured rubric prompts.

### 2.4 Statistical Comparison Engine (`llm_eval_lab.statistics` & `llm_eval_lab.regression`)
- **Paired Scoring**: Case IDs are aligned between baseline and candidate. Cases missing from either run or failing during execution are cleanly separated into audit lists.
- **Paired Bootstrap Resampling**: Resamples difference vectors $\Delta_i = C_i - B_i$ with replacement over $B = 2,000$ iterations to compute empirical percentiles:
  $$\text{CI}_{95\%} = [\text{percentile}_{2.5\%}, \text{percentile}_{97.5\%}]$$
- **Decision Engine State Machine**:
  - `IMPROVED`: Lower bound of CI $> 0$ (or $< 0$ for lower-is-better metrics) and exceeds practical significance threshold $\tau$.
  - `REGRESSED`: Upper bound of CI $< 0$ (or $> 0$ for lower-is-better metrics) and exceeds practical significance threshold $\tau$.
  - `INCONCLUSIVE`: Confidence interval spans across zero.
  - `INSUFFICIENT_DATA`: Paired sample count $n < 5$.

### 2.5 Storage Layer (`llm_eval_lab.storage`)
Lightweight, audit-friendly SQLite database (`eval_lab.db`):
- `experiments`: High-level experiment configurations and summaries stored as JSON text.
- `results`: Case-level results with `(experiment_id, case_id)` unique constraint and `ON DELETE CASCADE`.
- `comparisons`: Stored statistical comparison reports with foreign key indexes.
- Explicit connection management (`_tx()` and `_read_conn()`) ensuring immediate lock release on Windows and POSIX filesystems.

### 2.6 REST API (`llm_eval_lab.api`)
FastAPI application with explicit lifecycle management:
- `POST /api/datasets/validate`: Validates uploaded JSONL files with 10MB limits.
- `POST /api/experiments/`: Synchronous and asynchronous experiment runner.
- `GET /api/experiments/`: Paginated experiment list.
- `GET /api/experiments/{id}`: Detailed experiment retrieval.
- `DELETE /api/experiments/{id}`: Cascading deletion.
- `POST /api/comparisons/`: Statistical regression analysis between any two experiments.
- `GET /api/reports/...`: Markdown, JSON, and CSV report export.

### 2.7 Frontend Web Application (`frontend/`)
React 19 + TypeScript + Vite:
- Obsidian dark theme with glassmorphism and subtle micro-animations.
- Experiment tracking dashboard with drill-down modals.
- Real-time evaluation launch console.
- Interactive statistical comparison view with visual CI error-bar charts.
- JSONL drag-and-drop dataset validator.
- Export center for Markdown/JSON/CSV evidence packages.
