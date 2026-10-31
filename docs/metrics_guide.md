# Comprehensive Evaluators & Metrics Guide

`LLM Evaluation Lab` provides a layered suite of evaluators categorized by execution cost, determinism, and semantic depth.

> [!IMPORTANT]
> All metric implementations are designed to report `score=None` when they cannot compute a
> result (e.g., missing reference, missing dependency, provider error) rather than silently
> returning a misleading score. Always inspect the `error` field alongside the `score`.

---

## 1. Evaluator Taxonomy

```
                          ┌────────────────────────┐
                          │  Evaluator Categories  │
                          └───────────┬────────────┘
         ┌────────────────────────────┼────────────────────────────┐
         ▼                            ▼                            ▼
┌──────────────────┐         ┌──────────────────┐         ┌──────────────────┐
│  Deterministic   │         │    Structural    │         │     Semantic     │
│  Fast, Free,     │         │ JSON / Schema    │         │ Embeddings &     │
│  Mathematical    │         │ Validation       │         │ LLM-as-a-Judge   │
└──────────────────┘         └──────────────────┘         └──────────────────┘
```

---

## 2. Deterministic Evaluators

All deterministic evaluators are **free, fast, and require no external services**.

### 2.1 Exact Match (`exact_match`)

- **Type:** Binary indicator {0.0, 1.0}.
- **Formula:**
  $$\text{score} = \begin{cases} 1.0 & \text{if } \hat{y} = y^* \\ 0.0 & \text{otherwise} \end{cases}$$
- **Normalization:** None. Comparison is byte-exact including case and whitespace.
- **When to use:** Strict classification codes, IDs, enum values, mathematical expressions.
- **Returns unavailable when:** reference is `None` or provider returned an error.

### 2.2 Normalized Exact Match (`normalized_exact_match`)

- **Type:** Binary indicator {0.0, 1.0}.
- **Normalization steps applied (in order):**
  1. Strip leading/trailing whitespace.
  2. Collapse internal whitespace runs to a single space.
  3. Fold to lowercase.
- **Important:** Article stripping (a/an/the) is **not** applied. Use token F1 for looser matching.
- **When to use:** Short-answer QA where minor formatting differences should not penalize.
- **Returns unavailable when:** reference is `None` or provider returned an error.

### 2.3 Token Overlap F1 (`token_overlap_f1`)

- **Type:** Continuous score in [0.0, 1.0].
- **Tokenization:** Lowercase, strip punctuation, split on whitespace.
- **Formula:** Let $T_{\text{pred}}$ and $T_{\text{ref}}$ be the multisets of tokens:
  $$\text{Precision} = \frac{|T_{\text{pred}} \cap T_{\text{ref}}|}{|T_{\text{pred}}|}, \quad \text{Recall} = \frac{|T_{\text{pred}} \cap T_{\text{ref}}|}{|T_{\text{ref}}|}$$
  $$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
- **Edge cases:**
  - Both empty → F1 = 1.0 (vacuous match).
  - One empty → F1 = 0.0.
- **Limitation:** Token overlap measures word-level overlap, NOT semantic equivalence. A response with high F1 may still be factually wrong.
- **Returns unavailable when:** reference is `None` or provider returned an error.

### 2.4 Keyword Coverage (`keyword_coverage`)

- **Type:** Continuous score in [0.0, 1.0].
- **Formula:**
  $$\text{score} = \frac{|\{k \in K \mid k \in \hat{y}\}|}{|K|}$$
  where $K$ is `case.expected_keywords` (defined per test case).
- **Matching:** Case-insensitive whole-word match by default (configurable).
- **Returns unavailable when:** no `expected_keywords` are defined or provider returned an error.

### 2.5 Latency Evaluator (`latency_ms`)

- **Type:** Continuous real number ≥ 0.0 milliseconds.
- **Direction:** `lower_is_better`.
- **Measurement:** Wall-clock time around the provider network call.
- **Notes:** This metric is always computed (never returns unavailable). It is classified as **informational** in the regression engine and does NOT directly influence the overall experiment decision (see Regression Engine documentation).

---

## 3. Structural & Schema Evaluators

### 3.1 JSON Validity (`json_validity`)

- **Type:** Binary indicator {0.0, 1.0}.
- **Behavior:** Attempts `json.loads(output)` on the raw model output string.
- **Important:** Markdown code fences (` ```json ... ``` `) are **not** automatically stripped.
  If your model wraps JSON in code fences, pre-process or use a custom prompt template.
- **Returns unavailable when:** provider returned an error.

### 3.2 JSON Schema Compliance (`json_schema_compliance`)

- **Type:** Binary indicator {0.0, 1.0}.
- **Behavior:**
  1. Parses output as JSON (same as `json_validity`).
  2. Validates the parsed object against `case.expected_json_schema` using the `jsonschema` library.
- **Schema support:** JSON Schema Draft-07 and later (whatever `jsonschema` supports).
- **Installation required:** `pip install 'llm-eval-lab[json-schema]'` or `pip install jsonschema`.
- **Returns unavailable when:**
  - `expected_json_schema` is not defined for the test case.
  - `jsonschema` package is not installed.
  - Provider returned an error.

---

## 4. Semantic & Model Evaluators

### 4.1 Semantic Similarity (`semantic_similarity`)

- **Type:** Continuous score in **[-1.0, 1.0]** (cosine similarity of L2-normalized embeddings).
- **Model:** Sentence-Transformers (default: `all-MiniLM-L6-v2`, ~80 MB, CPU-compatible).
- **Formula:**
  $$\text{score} = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$
  where $\mathbf{u}$ and $\mathbf{v}$ are unit-normalized embedding vectors. Score is clamped to [-1.0, 1.0] to remove numerical noise.
- **Score interpretation:**
  - 1.0 → identical/paraphrase
  - 0.0 → orthogonal (no relationship)
  - negative → dissimilar or opposing
- **Default `passed` threshold:** 0.8 (configurable).
- **Critical limitation:** Semantic similarity measures **embedding distance, not factual correctness**. A response can score 0.9 while containing factual errors, and score 0.3 while being factually correct but phrased differently.
- **Installation required:** `pip install 'llm-eval-lab[semantic]'`
- **Returns unavailable when:**
  - `sentence-transformers` is not installed.
  - Reference is `None`.
  - Provider returned an error.

### 4.2 LLM-as-a-Judge (`judge_score`)

- **Type:** Continuous score in **[0.0, 1.0]** (normalized from raw 0–10 rubric).
- **Raw rubric:** The default prompt asks the judge model to score from 0 (completely wrong) to 10 (perfect).
- **Normalization:** `normalized_score = raw_score / 10.0`. The stored `score` field always holds the normalized value.
- **`passed` threshold:** `score >= 0.5` (equivalent to raw score ≥ 5).
- **Stored details:** `raw_score`, `score_range`, `judge_provider`, `judge_model`, `judge_response` (truncated to 500 chars), `judge_latency_ms`.
- **Critical limitation:** Judge scores are **subjective estimates produced by a language model, not objective ground truth**. The judge may be biased, inconsistent, or wrong. Always use alongside deterministic metrics. Results are auditable via stored `judge_response`.
- **Security note:** Judge prompt instructions are formatted as system instructions. User input is interpolated after the rubric template — not as free-form injection.
- **Installation required:** A configured `judge_provider` (e.g., `openai_compatible`) with a valid API key.

---

## 5. Metric Roles in Regression Testing

| Metric | Direction | Regression Role |
|--------|-----------|----------------|
| `exact_match` | higher_is_better | Quality (participates in overall decision) |
| `normalized_exact_match` | higher_is_better | Quality |
| `token_overlap_f1` | higher_is_better | Quality |
| `keyword_coverage` | higher_is_better | Quality |
| `json_validity` | higher_is_better | Quality |
| `json_schema_compliance` | higher_is_better | Quality |
| `semantic_similarity` | higher_is_better | Quality |
| `judge_score` | higher_is_better | Quality |
| `latency_ms` | lower_is_better | **Informational** (per-metric decision computed but excluded from overall experiment verdict) |

The overall experiment decision is determined by quality metrics only. Latency regression is reported per-metric and visible in the comparison report, but does not by itself block a release decision. Configure a dedicated latency threshold if latency gates are required.

---

## 6. Groundedness

**Groundedness is not currently implemented** as a standalone evaluator.

The `TestCase` schema includes an `expected_facts` field which is reserved for future use. A groundedness evaluator would verify that the model's output is supported by the provided `context` and does not contradict the `expected_facts`.

Reliable groundedness evaluation requires either:
- An NLI (Natural Language Inference) model, or
- An LLM judge specifically prompted for claim verification.

Neither approach is currently implemented. Use `keyword_coverage` as a limited proxy for checking that specific factual terms appear in the output, combined with `semantic_similarity` for general relevance. Do not treat either as a groundedness measure.
