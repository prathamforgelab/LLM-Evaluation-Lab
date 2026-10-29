# Comprehensive Evaluators & Metrics Guide

`LLM Evaluation Lab` provides a layered suite of evaluators categorized by execution cost, determinism, and semantic depth.

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

### 2.1 Exact Match (`exact_match`)
- **Type:** Binary indicator $\{0.0, 1.0\}$.
- **Formula:**
  $$\text{score} = \begin{cases} 1.0 & \text{if } \hat{y} = y^* \\ 0.0 & \text{otherwise} \end{cases}$$
- **When to use:** Strict classification codes, IDs, enum values, mathematical expressions.
- **Edge cases:** Trailing whitespace, case sensitivity.

### 2.2 Normalized Exact Match (`normalized_exact_match`)
- **Type:** Binary indicator $\{0.0, 1.0\}$.
- **Normalization Pipeline:**
  1. Lowercase conversion: $t \to \text{lower}(t)$.
  2. Strip punctuation: removes `[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]`.
  3. Strip leading/trailing articles: `a`, `an`, `the`.
  4. Collapse multiple whitespaces to single spaces.
- **When to use:** Short-answer QA, entity extraction, factual recall where formatting minor variations shouldn't penalize the model.

### 2.3 Token Overlap F1 (`token_overlap_f1`)
- **Type:** Continuous score in $[0.0, 1.0]$.
- **Formula:**
  Let $T_{\text{pred}}$ and $T_{\text{ref}}$ be the multiset of whitespace-delimited tokens in output and reference.
  $$\text{Precision} = \frac{|T_{\text{pred}} \cap T_{\text{ref}}|}{|T_{\text{pred}}|}, \quad \text{Recall} = \frac{|T_{\text{pred}} \cap T_{\text{ref}}|}{|T_{\text{ref}}|}$$
  $$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
- **Edge cases:** If both prediction and reference are empty, $F_1 = 1.0$. If prediction is empty while reference is non-empty, $F_1 = 0.0$.

### 2.4 Keyword Coverage (`keyword_coverage`)
- **Type:** Continuous score in $[0.0, 1.0]$.
- **Formula:**
  $$\text{score} = \frac{|\{k \in K \mid k \in \hat{y}\}|}{|K|}$$
  where $K$ is the list of expected keywords defined in test case `metadata["expected_keywords"]`.
- **When to use:** Verifying that mandatory policy terms, legal clauses, or key entity names appear in the output.

### 2.5 Latency Evaluator (`latency_ms`)
- **Type:** Continuous real number $> 0.0$ milliseconds.
- **Direction:** `lower_is_better`.
- **Measurement:** High-resolution wall-clock execution time around provider network calls:
  $$t_{\text{latency}} = (t_{\text{end}} - t_{\text{start}}) \cdot 1000.0$$

---

## 3. Structural & Schema Evaluators

### 3.1 JSON Validity (`json_validity`)
- **Type:** Binary indicator $\{0.0, 1.0\}$.
- **Behavior:** Attempts to parse model output using Python's `json.loads()`. Output wrapped in markdown code blocks (` ```json ... ``` `) is automatically unwrapped before parsing.

### 3.2 JSON Schema Compliance (`json_schema`)
- **Type:** Binary indicator $\{0.0, 1.0\}$.
- **Behavior:** Parses JSON output and validates it against `expected_json_schema` using standard JSON Schema Draft-07/2020-12 specifications. Validation errors are detailed in the `details["schema_errors"]` payload.

---

## 4. Semantic & Model Evaluators

### 4.1 Semantic Similarity (`semantic_similarity`)
- **Type:** Continuous score in $[0.0, 1.0]$.
- **Model:** Sentence-Transformers (e.g. `all-MiniLM-L6-v2`).
- **Formula:**
  $$\text{score} = \max\left(0.0, \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}\right)$$
- **When to use:** Paraphrase matching, summarization, long-form answer evaluation where token overlap is too brittle.

### 4.2 LLM-as-a-Judge (`judge`)
- **Type:** Continuous score or rubric scale $[1, 5]$.
- **Prompt Structure:** Evaluates model answer against reference and grounding context using structured rubrics (Faithfulness, Coherence, Completeness).
- **Security:** Judge instructions are explicitly framed as system instructions; user inputs are strictly enclosed in delimiter tags to prevent prompt injection.
